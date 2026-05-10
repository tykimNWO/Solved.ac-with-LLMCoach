import os
import json
import time
import sqlite3
import subprocess
import tempfile
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from fastapi.responses import StreamingResponse
from src.recommender import stream_chat_response
from src.database import DatabaseManager

from typing import List, Optional

# 💡 DB 경로 설정 (에러 방지를 위한 절대 경로)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'data', 'tracker.db')
db = DatabaseManager(DB_PATH)

# FastAPI 앱 생성
app = FastAPI(title="Solved.ac-with-LLMCoach API", description="Solved.ac-with-LLMCoach 백엔드 서버")

# 💡 CORS 설정 (React의 5173 포트에서 오는 요청을 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# 📌 1. 데이터 모델 정의 (Pydantic) - 들어오는 요청의 형태를 검증
# ----------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    text: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = Field(default_factory=list)
    current_problem_id: Optional[int] = None

class JudgeRequest(BaseModel):
    problem_id: int
    code: str

# ----------------------------------------------------
# 📌 2. API 엔드포인트 구현
# ----------------------------------------------------

@app.get("/")
def read_root():
    return {"status": "ok", "message": "AI Tutor API Server is running."}

@app.post("/api/chat/stream")
async def chat_with_ai_stream(req: ChatRequest):
    """Gemini의 응답을 실시간으로 프론트엔드에 스트리밍합니다."""
    print(f"📥 [User Message]: {req.message} (History: {len(req.history)} items)")
    
    # 1. 사용자 메시지 DB 저장
    db.save_chat_message("user", req.message)
    
    # 2. history를 dict list로 변환하여 전달
    history_dict = [{"role": msg.role, "text": msg.text} for msg in req.history]
    
    async def wrapped_stream():
        full_response = ""
        # stream_chat_response가 동기 제너레이터라면 아래와 같이 사용
        for chunk in stream_chat_response(req.message, history_dict, req.current_problem_id):
            full_response += chunk
            yield chunk
        
        # 3. AI 응답 완료 후 DB 저장
        if full_response:
            db.save_chat_message("ai", full_response)

    return StreamingResponse(wrapped_stream(), media_type="text/plain")

@app.get("/api/chat/history")
async def get_chat_history():
    """DB에서 이전 대화 내역을 가져옵니다."""
    return {"status": "success", "history": db.get_chat_history(limit=50)}

@app.delete("/api/chat/history")
async def clear_chat_history():
    """대화 내역을 초기화합니다."""
    db.clear_chat_history()
    return {"status": "success", "message": "History cleared"}

@app.get("/api/problem/{problem_id}")
async def get_problem(problem_id: int):
    """DB에서 문제 상세 정보를 가져오는 API"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT pd.description, pd.input_desc, pd.output_desc, pd.sample_inputs, pd.sample_outputs, pd.problem_limit,
                   p.title, p.tier, p.tags
            FROM problem_details pd
            JOIN problems p ON pd.problem_id = p.problem_id
            WHERE pd.problem_id = ?
        """, (problem_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            tags_list = []
            if row[8]:
                try:
                    tags_list = json.loads(row[8])
                except json.JSONDecodeError:
                    pass

            # Check if the problem is solved
            solved_ids = db.get_solved_problem_ids()
            is_solved = problem_id in solved_ids

            return {
                "status": "success",
                "data": {
                    "description": row[0],
                    "input_desc": row[1],
                    "output_desc": row[2],
                    "sample_inputs": json.loads(row[3]) if row[3] else [],
                    "sample_outputs": json.loads(row[4]) if row[4] else [],
                    "problem_limit": row[5] or "",
                    "title": row[6] or "",
                    "tier": row[7] or 0,
                    "tags": tags_list,
                    "is_solved": is_solved
                }
            }
        else:
            raise HTTPException(status_code=404, detail="DB에 해당 문제가 없습니다.")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/judge")
async def run_judge(req: JudgeRequest):
    """사용자의 파이썬 코드를 채점하는 API"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT sample_inputs, sample_outputs FROM problem_details WHERE problem_id = ?", (req.problem_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise HTTPException(status_code=404, detail="채점 데이터를 찾을 수 없습니다.")
        
        sample_inputs = json.loads(row[0])
        sample_outputs = json.loads(row[1])
        results = []

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w', encoding='utf-8') as f:
            f.write(req.code)
            temp_code_path = f.name

        for i in range(len(sample_inputs)):
            start_time = time.time()
            try:
                process = subprocess.run(
                    ["python3", temp_code_path],
                    input=sample_inputs[i],
                    capture_output=True,
                    text=True,
                    timeout=2.0 
                )
                
                elapsed_time = (time.time() - start_time) * 1000
                actual_output = process.stdout.strip()
                expected_output = sample_outputs[i].strip()

                if process.returncode != 0:
                    res = {"case": i+1, "result": "Runtime Error", "error": process.stderr}
                elif actual_output == expected_output:
                    res = {"case": i+1, "result": "Success", "time": f"{elapsed_time:.1f}ms"}
                else:
                    res = {"case": i+1, "result": "Wrong Answer", "actual": actual_output, "expected": expected_output}
                    
            except subprocess.TimeoutExpired:
                res = {"case": i+1, "result": "Time Limit Exceeded"}
            
            results.append(res)

        os.remove(temp_code_path)
        return {"status": "success", "results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# main.py 에 추가 및 수정할 내용

# 1. 메모 요청 모델 추가
class MemoRequest(BaseModel):
    problem_id: int
    content: str

# 2. 서버 시작 시 테이블 생성 로직 (앱 하단이나 초기화 부분에 추가)
@app.on_event("startup")
def setup_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # 메모 테이블 생성 (문제 번호를 PK로 하여 1:1 대응)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memos (
            problem_id INTEGER PRIMARY KEY,
            content TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # problem_details에 제한(시간/메모리) HTML 컬럼이 없으면 추가 (기존 DB 호환)
    try:
        cursor.execute("ALTER TABLE problem_details ADD COLUMN problem_limit TEXT")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

# 3. 메모 조회 API
@app.get("/api/memo/{problem_id}")
async def get_memo(problem_id: int):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM memos WHERE problem_id = ?", (problem_id,))
        row = cursor.fetchone()
        conn.close()
        return {"status": "success", "content": row[0] if row else ""}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 4. 메모 저장 API
@app.post("/api/memo")
async def save_memo(req: MemoRequest):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # INSERT OR REPLACE를 사용하여 기존 메모가 있으면 업데이트
        cursor.execute("""
            INSERT OR REPLACE INTO memos (problem_id, content, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, (req.problem_id, req.content))
        conn.commit()
        conn.close()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # main.py 파일 안의 app 객체를 실행하며, 코드가 바뀔 때마다 자동 재시작(reload=True)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
