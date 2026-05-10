import os
import chromadb
from google import genai
from google.genai import types
import config
from src.database import DatabaseManager
import json
import re

# 1. API 클라이언트 초기화
client = genai.Client(api_key=config.GEMINI_API_KEY)

# 2. DB 및 ChromaDB 연동
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'tracker.db')
db = DatabaseManager(DB_PATH)

CHROMA_DB_PATH = os.path.join(BASE_DIR, 'data', 'chroma_db')
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="boj_problems")

GEMINI_MODEL_ID = os.environ.get("GEMINI_MODEL_ID", "gemini-3.1-flash-lite-preview")
EMBEDDING_MODEL_ID = "gemini-embedding-001"
RECOMMENDATION_RESULT_START = "[RECOMMENDATION_RESULT]"
RECOMMENDATION_RESULT_END = "[/RECOMMENDATION_RESULT]"

RECOMMENDATION_STEPS = [
    {"id": "profile", "label": "solved.ac 프로필 데이터를 확인하고 있습니다."},
    {"id": "patterns", "label": "최근 풀이 난이도와 태그 패턴을 분석하고 있습니다."},
    {"id": "candidates", "label": "추천 후보 문제를 생성하고 있습니다."},
    {"id": "ranking", "label": "Top-3 추천 문제와 추천 이유를 정리하고 있습니다."},
]

# 💡 태영님만을 위한 초개인화 시스템 프롬프트
SYSTEM_INSTRUCTION = """당신은 김태영님의 전담 알고리즘 및 코딩 튜터입니다.
학습자는 현재 은행의 AI데이터전략부에서 근무 중인 데이터 기획자이자 엔지니어이며,
기회비용과 정확성, 실무적 운영성을 매우 중요하게 생각하는 분석적이고 계획적인 성향입니다.
항상 신사적이고 매너 있는 어조를 유지하세요.

[중요 지시사항]
1. 사용자의 질문에 답하되, 제공된 로컬 문제 데이터와 현재 문제 맥락을 우선 참고하세요.
2. 추천 요청은 별도의 구조화 추천 파이프라인에서 처리되므로, 일반 답변에서는 문제 이동 태그를 만들지 마세요.
3. 사용자가 볼 수 있는 간결한 설명만 작성하고, 숨겨진 사고 과정이나 장문 추론을 노출하지 마세요.
"""

def analyze_intent_and_rewrite(message: str, history: list) -> tuple[bool, str, str]:
    """
    과거 대화를 바탕으로 사용자의 의도를 파악하고, ChromaDB에 던질 검색어를 정제합니다.
    """
    # 1차 방어선 (명시적 키워드 우선순위)
    review_keywords = ["풀었던", "복습", "다시", "기존에", "풀어본", "저번에", "아까"]
    is_explicit_review = any(keyword in message for keyword in review_keywords)
    
    # 최근 대화 6개 정도만 집중적으로 참고
    history_text = "\n".join([f"{msg['role']}: {msg['text']}" for msg in history[-6:]])
    
    prompt = f"""
    당신은 사용자의 대화 맥락을 추적하고 최적의 검색 쿼리를 생성하는 의도 분석기입니다.
    
    [핵심 지침: 세션 모드 유지]
    1. 사용자가 이전 대화에서 '복습(review)'을 원했다면, "다른 거", "더 추천해줘"라는 질문에도 **반드시 'review' 상태를 유지**해야 합니다.
    2. 명시적으로 "이제 안 푼 거 줘", "새로운 문제", "다른 유형의 신규 문제"라고 해야만 'new'로 바뀝니다.
    3. 근거 없는 구체화(예: 말하지 않은 골드 5 티어 추가)는 절대 금지입니다.
    4. 내부 추론 과정을 쓰지 말고, 사용자에게 보여줄 수 있는 한 문장 상태 요약만 작성하세요.
    
    [대화 이력]
    {history_text}
    
    [현재 질문]
    user: {message}
    
    [작업 지침]
    - `status_summary`에는 사용자에게 노출해도 안전한 진행 요약만 작성하세요.
    
    [출력 형식 (반드시 JSON만)]
    {{
        "status_summary": "대화 맥락을 확인해 복습 추천 모드로 검색어를 정리했습니다.",
        "intent": "review",
        "search_query": "그리디 알고리즘"
    }}
    """
    
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1 # 일관성을 위해 낮은 온도로 설정
            )
        )
        result = json.loads(response.text)
        
        status_summary = result.get('status_summary', '대화 맥락을 확인해 추천 검색어를 정리했습니다.')
        print(f"🧭 [추천 상태 요약]: {status_summary}")
        
        # 명시적 키워드가 있거나 LLM이 review라고 판단한 경우 복습 모드 유지
        if is_explicit_review:
            exclude_solved = False
        else:
            exclude_solved = (result.get("intent") != "review")
            
        refined_query = result.get("search_query", message)
        return exclude_solved, refined_query, status_summary
        
    except Exception as e:
        print(f"⚠️ 의도 분석 실패, 기본값 진행: {e}")
        return not is_explicit_review, message, "대화 이력을 참고하여 추천을 진행합니다."

def is_recommendation_request(message: str) -> bool:
    keywords = ["추천", "문제 줘", "문제 찾아", "풀 문제", "복습", "약점", "오늘의 추천"]
    return any(keyword in message for keyword in keywords)

def emit_step(step_id: str) -> str:
    return f"[RECOMMENDATION_STEP:{step_id}]\n"

def parse_tags(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(tag) for tag in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(tag) for tag in parsed]
        except json.JSONDecodeError:
            return [tag.strip(" '\"") for tag in value.strip("[]").split(",") if tag.strip()]
    return []

def normalize_problem_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value

def tier_to_name(tier) -> str:
    try:
        level = int(tier)
    except (TypeError, ValueError):
        return str(tier) if tier else ""
    if level <= 0:
        return "Unrated"
    names = [
        ("Bronze", 1, 5),
        ("Silver", 6, 10),
        ("Gold", 11, 15),
        ("Platinum", 16, 20),
        ("Diamond", 21, 25),
        ("Ruby", 26, 30),
    ]
    for label, start, end in names:
        if start <= level <= end:
            return f"{label} {end - level + 1}"
    return str(level)

def problem_from_metadata(metadata: dict, document: str = "") -> dict:
    problem_id = normalize_problem_id(metadata.get("problem_id"))
    tier = metadata.get("tier")
    tags = parse_tags(metadata.get("tags"))
    return {
        "problemId": problem_id,
        "title": metadata.get("title") or "제목 없음",
        "tier": tier_to_name(tier),
        "level": int(tier) if str(tier).isdigit() else None,
        "tags": tags,
        "reason": "",
        "learningEffect": "",
        "solvedAcUrl": f"https://www.acmicpc.net/problem/{problem_id}" if problem_id else "",
        "document": document[:1200],
    }

def enrich_candidates_from_db(candidates: list[dict]) -> list[dict]:
    ids = [candidate.get("problemId") for candidate in candidates if candidate.get("problemId")]
    if not ids:
        return candidates

    placeholders = ",".join(["?"] * len(ids))
    try:
        with db.get_connection() as conn:
            cursor = conn.execute(
                f"SELECT problem_id, title, tier, tags FROM problems WHERE problem_id IN ({placeholders})",
                ids,
            )
            metadata_by_id = {
                row[0]: {
                    "title": row[1],
                    "tier": tier_to_name(row[2]),
                    "level": row[2],
                    "tags": parse_tags(row[3]),
                }
                for row in cursor.fetchall()
            }
    except Exception as e:
        print(f"⚠️ 문제 메타데이터 보강 실패: {e}")
        return candidates

    enriched = []
    for candidate in candidates:
        problem_id = candidate.get("problemId")
        db_meta = metadata_by_id.get(problem_id, {})
        enriched.append({**candidate, **{k: v for k, v in db_meta.items() if v}})
    return enriched

def get_user_tag_snapshot() -> dict:
    try:
        with db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT tag_id, solved_count
                FROM user_tag_stats
                WHERE date = (SELECT MAX(date) FROM user_tag_stats)
                ORDER BY solved_count DESC
            """)
            rows = cursor.fetchall()
    except Exception as e:
        print(f"⚠️ 태그 통계 조회 실패: {e}")
        return {"strongTags": [], "weakTags": []}

    strong_tags = [{"tag": row[0], "solvedCount": row[1]} for row in rows[:8]]
    weak_tags = [{"tag": row[0], "solvedCount": row[1]} for row in rows[-8:]]
    return {"strongTags": strong_tags, "weakTags": weak_tags}

def extract_recommended_problem_ids(text: str) -> list[int]:
    ids = [int(match) for match in re.findall(r"\[LOAD_PROBLEM:(\d+)\]", text)]
    for raw_json in re.findall(
        rf"{re.escape(RECOMMENDATION_RESULT_START)}(.*?){re.escape(RECOMMENDATION_RESULT_END)}",
        text,
        flags=re.DOTALL,
    ):
        try:
            data = json.loads(raw_json)
            for item in data.get("recommendations", []):
                problem_id = normalize_problem_id(item.get("problemId") or item.get("problem_id"))
                if isinstance(problem_id, int):
                    ids.append(problem_id)
        except json.JSONDecodeError:
            continue
    return ids

def query_similar_problems(query: str, solved_ids: list, exclude_solved: bool, blacklist_ids: list | None = None, top_k: int = 3):
    blacklist_ids = blacklist_ids or []
    try:
        # 💡 [핵심 예외 처리] 복습을 원하는데 tracker.db에 푼 문제가 없을 때!
        if not exclude_solved and len(solved_ids) == 0:
            return "학습자님이 아직 시스템에 기록한 푼 문제(Solved) 이력이 없습니다. 먼저 새로운 문제를 풀고 기록을 남겨주셔야 복습 추천이 가능합니다.", []

        # 질문 임베딩 (기존 로직)
        response = client.models.embed_content(model=EMBEDDING_MODEL_ID, contents=query)
        query_embedding = response.embeddings[0].values
        
        # ChromaDB 필터링 (동적 조건 적용)
        conditions = []
        
        # 1. 푼 문제 관련 조건 (복습 vs 신규)
        if solved_ids:
            if exclude_solved:
                conditions.append({"problem_id": {"$nin": solved_ids}}) # 안 푼 문제만
            else:
                conditions.append({"problem_id": {"$in": solved_ids}})  # 푼 문제만
        
        # 2. 최근 추천 리스트 제외 (블랙리스트)
        if blacklist_ids:
            conditions.append({"problem_id": {"$nin": blacklist_ids}})
            
        # 조건 결합 (ChromaDB $and 연산)
        where_clause = None
        if len(conditions) > 1:
            where_clause = {"$and": conditions}
        elif len(conditions) == 1:
            where_clause = conditions[0]
                
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause
        )
        
        # 검색 결과가 아예 비어있을 때의 환각 방지
        if not results or not results['documents'] or not results['documents'][0]:
            return "조건에 맞는 문제를 데이터베이스에서 찾을 수 없습니다.", []

        retrieved_docs = results['documents'][0]
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        candidates = []
        for index, document in enumerate(retrieved_docs):
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            candidates.append(problem_from_metadata(metadata, document))
        candidates = enrich_candidates_from_db(candidates)
        context = "\n\n---\n\n".join(retrieved_docs)
        return context, candidates
        
    except Exception as e:
        print(f"❌ 검색 에러: {e}")
        return "", []

def retrieve_similar_problems(query: str, solved_ids: list, exclude_solved: bool, blacklist_ids: list | None = None, top_k: int = 3):
    context, _ = query_similar_problems(query, solved_ids, exclude_solved, blacklist_ids, top_k)
    return context

def parse_structured_json(text: str) -> dict:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise

def normalize_recommendation_response(response: dict | None, candidates: list[dict], fallback_summary: str = "") -> dict:
    response = response or {}
    raw_items = response.get("recommendations") or response.get("topRecommendations") or response.get("top_3") or []
    if isinstance(raw_items, dict):
        raw_items = [raw_items]

    candidate_by_id = {
        str(candidate.get("problemId")): candidate
        for candidate in candidates
        if candidate.get("problemId") is not None
    }

    normalized = []
    seen_ids = set()
    for index, item in enumerate(raw_items[:3]):
        if not isinstance(item, dict):
            continue
        problem_id = normalize_problem_id(item.get("problemId") or item.get("problem_id") or item.get("id"))
        candidate = candidate_by_id.get(str(problem_id), {})
        if not problem_id and candidate:
            problem_id = candidate.get("problemId")
        if not problem_id:
            continue

        tags = item.get("tags") if isinstance(item.get("tags"), list) else candidate.get("tags", [])
        tier = item.get("tier") or candidate.get("tier") or tier_to_name(item.get("level"))
        level = item.get("level") or candidate.get("level")
        normalized.append({
            "rank": int(item.get("rank") or index + 1),
            "problemId": problem_id,
            "title": item.get("title") or candidate.get("title") or "제목 없음",
            "tier": tier,
            "level": level,
            "tags": [str(tag) for tag in tags[:5]],
            "reason": item.get("reason") or "현재 검색 조건과 학습 상태에 맞는 후보 문제입니다.",
            "learningEffect": item.get("learningEffect") or item.get("learning_effect") or "핵심 알고리즘 패턴을 안정적으로 연습할 수 있습니다.",
            "solvedAcUrl": item.get("solvedAcUrl") or item.get("url") or candidate.get("solvedAcUrl") or f"https://www.acmicpc.net/problem/{problem_id}",
        })
        seen_ids.add(str(problem_id))

    if len(normalized) < 3:
        for candidate in candidates:
            problem_id = candidate.get("problemId")
            if not problem_id or str(problem_id) in seen_ids:
                continue
            normalized.append({
                "rank": len(normalized) + 1,
                "problemId": problem_id,
                "title": candidate.get("title") or "제목 없음",
                "tier": candidate.get("tier") or "",
                "level": candidate.get("level"),
                "tags": candidate.get("tags", [])[:5],
                "reason": "Gemini 구조화 응답이 부족해 로컬 검색 후보를 기준으로 보완했습니다.",
                "learningEffect": "현재 검색 의도와 가까운 문제를 풀며 관련 태그 감각을 점검할 수 있습니다.",
                "solvedAcUrl": candidate.get("solvedAcUrl") or f"https://www.acmicpc.net/problem/{problem_id}",
            })
            seen_ids.add(str(problem_id))
            if len(normalized) >= 3:
                break

    for index, item in enumerate(normalized):
        item["rank"] = index + 1

    return {
        "summary": response.get("summary") or fallback_summary or "현재 학습 상태와 검색 후보를 바탕으로 추천을 정리했습니다.",
        "processSteps": response.get("processSteps") or RECOMMENDATION_STEPS,
        "recommendations": normalized,
    }

def build_recommendation_result(message: str, user_stats: dict | None, solved_ids: list[int], refined_query: str, retrieved_context: str, candidates: list[dict], status_summary: str) -> dict:
    tag_snapshot = get_user_tag_snapshot()
    user_profile = {
        "tier": user_stats.get("tier") if user_stats else None,
        "rating": user_stats.get("rating") if user_stats else None,
        "solvedCount": user_stats.get("solved_count") if user_stats else len(solved_ids),
        "streak": user_stats.get("streak") if user_stats else None,
        "strongTags": tag_snapshot["strongTags"],
        "weakTags": tag_snapshot["weakTags"],
    }

    candidate_payload = [
        {key: value for key, value in candidate.items() if key != "document"}
        for candidate in candidates[:10]
    ]

    prompt = f"""
당신은 solved.ac 문제 추천 결과를 프론트엔드에서 바로 렌더링 가능한 JSON으로 만드는 추천 엔진입니다.
숨겨진 사고 과정은 출력하지 말고, 사용자가 볼 수 있는 요약과 추천 이유만 작성하세요.

[함수형 처리 단계]
1. analyze_user_profile: solved.ac 사용자 프로필, 티어, 풀이 수, 태그 분포를 요약합니다.
2. generate_problem_candidates: 제공된 후보 문제 안에서 학습 목적에 맞는 후보를 고릅니다.
3. rank_recommendations: 학습 효율, 난이도 적합성, 태그 보완성, 풀이 가능성으로 정렬합니다.
4. format_recommendation_result: Top-3 추천 카드를 JSON으로 정리합니다.

[사용자 프로필]
{json.dumps(user_profile, ensure_ascii=False)}

[사용자 요청]
{message}

[정제된 검색어]
{refined_query}

[검색 후보 메타데이터]
{json.dumps(candidate_payload, ensure_ascii=False)}

[검색 후보 본문 요약]
{retrieved_context[:6000]}

[출력 JSON 스키마]
{{
  "summary": "사용자에게 보여줄 추천 요약 한 문장",
  "processSteps": [
    {{"id": "profile", "label": "solved.ac 프로필 데이터를 확인했습니다."}},
    {{"id": "patterns", "label": "최근 풀이 난이도와 태그 패턴을 분석했습니다."}},
    {{"id": "candidates", "label": "추천 후보 문제를 생성했습니다."}},
    {{"id": "ranking", "label": "Top-3 추천 문제와 추천 이유를 정리했습니다."}}
  ],
  "recommendations": [
    {{
      "rank": 1,
      "problemId": 1234,
      "title": "문제 제목",
      "tier": "Gold IV",
      "level": 12,
      "tags": ["dp", "graphs"],
      "reason": "카드에서 읽기 좋은 1-2문장 추천 이유",
      "learningEffect": "기대 학습 효과 1문장",
      "solvedAcUrl": "https://www.acmicpc.net/problem/1234"
    }}
  ]
}}

반드시 검색 후보에 있는 문제만 추천하고, recommendations는 최대 3개로 제한하세요.
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.35,
            ),
        )
        parsed = parse_structured_json(response.text)
        return normalize_recommendation_response(parsed, candidates, status_summary)
    except Exception as e:
        print(f"⚠️ 구조화 추천 생성 실패, fallback 사용: {e}")
        return normalize_recommendation_response(None, candidates, "Gemini 응답을 구조화하지 못해 검색 후보를 기준으로 추천을 보완했습니다.")

def stream_chat_response(message: str, history: list=[], current_problem_id: int=None):
    """RAG 파이프라인을 실행하고 결과를 실시간으로 반환합니다."""
    print(f"🚀 [RAG 시작] 질문: {message} (현재 문제: {current_problem_id})")
    recommendation_mode = is_recommendation_request(message)
    
    # 0. 사용자 정보 및 의도 파악
    solved_ids = db.get_solved_problem_ids()
    user_stats = db.get_latest_user_stats()
    if recommendation_mode:
        yield emit_step("profile")
    
    # 0.2. 현재 보고 있는 문제 정보 가져오기 (세션 컨텍스트)
    current_problem_context = ""
    if current_problem_id:
        try:
            with db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT p.title, p.tier, pd.description 
                    FROM problems p 
                    JOIN problem_details pd ON p.problem_id = pd.problem_id 
                    WHERE p.problem_id = ?
                """, (current_problem_id,))
                row = cursor.fetchone()
                if row:
                    current_problem_context = f"\n[현재 학습자가 보고 있는 문제]: {current_problem_id}. {row[0]} (티어: {row[1]})\n[문제 설명 요약]: {row[2][:500]}..."
        except Exception as e:
            print(f"⚠️ 현재 문제 정보 로드 실패: {e}")

    # 사용자가 '다시', '복습', '풀었던' 등의 키워드를 사용했는지 확인
    review_keywords = ["복습", "다시", "풀었던", "이미 푼", "review", "again", "solved"]
    is_review_request = any(keyword in message for keyword in review_keywords)
    
    user_context = ""
    if user_stats:
        user_context = f"\n[학습자 현재 상태]: 티어 {user_stats['tier']}, 레이팅 {user_stats['rating']}, 해결한 문제 수 {user_stats['solved_count']}, 스트릭 {user_stats['streak']}일"

    # 0.5. 최근 대화에서 추천된 문제 번호 추출 (블랙리스트)
    # 프론트엔드에서 태그를 파싱하고 지울 수 있으므로, DB에 저장된 원본 메시지에서 추출하는 것이 가장 정확합니다.
    blacklist_ids = []
    
    # DB에서 최근 추천된 모든 문제 ID 추출
    try:
        raw_history = db.get_chat_history(limit=50)
        for msg in raw_history:
            if msg['role'] == 'ai':
                blacklist_ids.extend(extract_recommended_problem_ids(msg['text']))
    except Exception as e:
        print(f"⚠️ 블랙리스트 추출 중 오류 (기존 history 참고): {e}")
        # DB 실패 시 전달받은 history에서라도 추출 시도
        for msg in history:
            if msg['role'] == 'ai':
                blacklist_ids.extend(extract_recommended_problem_ids(msg['text']))
    
    # 중복 제거
    blacklist_ids = list(set(blacklist_ids))
    
    # 1. 문서 검색 (Retrieval)
    exclude_solved, refined_query, status_summary = analyze_intent_and_rewrite(message, history)

    if recommendation_mode:
        yield emit_step("patterns")
        yield f"[SAFE_PROGRESS]\n{status_summary}\n[/SAFE_PROGRESS]\n"

    status_msg = "푼 문제 제외" if exclude_solved else "푼 문제 포함(복습)"
    if blacklist_ids:
        print(f"🚫 최근 추천된 {len(blacklist_ids)}개 문제(ID: {blacklist_ids})는 검색에서 제외합니다.")

    print(f"🔍 [Query Rewritten] 정제된 검색어: '{refined_query}' ({status_msg})")
    
    if recommendation_mode:
        yield emit_step("candidates")
        retrieved_context, candidates = query_similar_problems(
            query=refined_query,
            solved_ids=solved_ids,
            exclude_solved=exclude_solved,
            blacklist_ids=blacklist_ids,
            top_k=10,
        )
        yield emit_step("ranking")
        recommendation_result = build_recommendation_result(
            message=message,
            user_stats=user_stats,
            solved_ids=solved_ids,
            refined_query=refined_query,
            retrieved_context=retrieved_context,
            candidates=candidates,
            status_summary=status_summary,
        )
        yield f"\n{RECOMMENDATION_RESULT_START}{json.dumps(recommendation_result, ensure_ascii=False)}{RECOMMENDATION_RESULT_END}\n"
        return

    retrieved_context = retrieve_similar_problems(
        query=refined_query,
        solved_ids=solved_ids,
        exclude_solved=exclude_solved,
        blacklist_ids=blacklist_ids
    )
    
    # 2. 프롬프트 증강 (Augmentation)
    prompt_instruction = "목록에 있는 문제는 사용자가 아직 풀지 않은 유사한 문제들입니다." if exclude_solved else "목록에 있는 문제는 사용자가 이미 풀었거나 관련 있는 복습용 문제들입니다."
    
    augmented_prompt = f"""사용자의 질문에 답변하세요.
{user_context}
{current_problem_context}

필요하다면 아래의 <검색된_문제_목록>을 참고하여 추천을 진행하세요.
{prompt_instruction}

<검색된_문제_목록>
{retrieved_context}
</검색된_문제_목록>

사용자 질문: {message}
"""
    
    # 3. 모델 생성 (Generation)
    generation_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.7,
    )

    try:
        response = client.models.generate_content_stream(
            model=GEMINI_MODEL_ID,
            contents=augmented_prompt,
            config=generation_config
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text
                
    except Exception as e:
        yield f"\n[AI 엔진 연결 오류]: {str(e)}"
