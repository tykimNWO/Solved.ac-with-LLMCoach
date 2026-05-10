import React, { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, Brain, Code2, MessageSquare, FileCode2, Search, Code, Notebook, CheckCircle2, Clock3, ListChecks, Loader2, AlertCircle, ExternalLink, Circle } from 'lucide-react';
import CodeMirror from '@uiw/react-codemirror';
import { python } from '@codemirror/lang-python';
import { markdown } from '@codemirror/lang-markdown';
import { vscodeDark } from '@uiw/codemirror-theme-vscode';

const API_BASE_URL = 'http://localhost:8000/api';

type RecommendationStatus = 'idle' | 'loading' | 'reasoning' | 'success' | 'error';
type RecommendationStepStatus = 'pending' | 'running' | 'completed' | 'error';

interface RecommendationStep {
  id: string;
  label: string;
  status: RecommendationStepStatus;
  timestamp?: number;
}

interface RecommendedProblem {
  rank: number;
  problemId: number | string;
  title: string;
  tier?: string;
  level?: number;
  tags?: string[];
  reason: string;
  learningEffect?: string;
  solvedAcUrl?: string;
}

interface RecommendationResult {
  recommendations: RecommendedProblem[];
  elapsedMs: number;
  summary?: string;
}

interface ProblemData {
  description: string;
  input_desc: string;
  output_desc: string;
  sample_inputs: string[];
  sample_outputs: string[];
  problem_limit?: string;
  title?: string;
  tier: number;
  tags: string[];
  is_solved?: boolean;
}

interface JudgeResult {
  case: number;
  result: string;
  time?: string;
  error?: string;
  actual?: string;
  expected?: string;
}

interface ChatMessage {
  role: 'user' | 'ai';
  text: string;
  elapsed?: number;
  recommendationResult?: RecommendationResult;
}

const RECOMMENDATION_RESULT_START = '[RECOMMENDATION_RESULT]';
const RECOMMENDATION_RESULT_END = '[/RECOMMENDATION_RESULT]';

const RECOMMENDATION_STEPS: RecommendationStep[] = [
  { id: 'profile', label: 'solved.ac 프로필 데이터를 확인하고 있습니다.', status: 'pending' },
  { id: 'patterns', label: '최근 풀이 난이도와 태그 패턴을 분석하고 있습니다.', status: 'pending' },
  { id: 'candidates', label: '추천 후보 문제를 생성하고 있습니다.', status: 'pending' },
  { id: 'ranking', label: 'Top-3 추천 문제와 추천 이유를 정리하고 있습니다.', status: 'pending' },
];

const parseTierLevel = (tier?: number | string) => {
  if (typeof tier === 'number') return tier;
  if (!tier) return 0;
  const match = tier.match(/^(Bronze|Silver|Gold|Platinum|Diamond|Ruby)\s+([1-5])$/i);
  if (!match) {
    const numericTier = Number.parseInt(tier, 10);
    return Number.isFinite(numericTier) ? numericTier : 0;
  }
  const rank = Number.parseInt(match[2], 10);
  const baseByGroup: Record<string, number> = {
    bronze: 0,
    silver: 5,
    gold: 10,
    platinum: 15,
    diamond: 20,
    ruby: 25,
  };
  return baseByGroup[match[1].toLowerCase()] + (6 - rank);
};

const getTierInfo = (tier?: number | string) => {
  const level = parseTierLevel(tier);
  if (!level || level === 0) return { name: "Unrated", color: "text-gray-400", bg: "bg-gray-400/10", border: "border-gray-400/30" };
  if (level <= 5) return { name: `Bronze ${6 - level}`, color: "text-[#ad5600]", bg: "bg-[#ad5600]/10", border: "border-[#ad5600]/30" };
  if (level <= 10) return { name: `Silver ${11 - level}`, color: "text-[#435f7a]", bg: "bg-[#435f7a]/10", border: "border-[#435f7a]/30" };
  if (level <= 15) return { name: `Gold ${16 - level}`, color: "text-[#ec9a00]", bg: "bg-[#ec9a00]/10", border: "border-[#ec9a00]/30" };
  if (level <= 20) return { name: `Platinum ${21 - level}`, color: "text-[#27e2a4]", bg: "bg-[#27e2a4]/10", border: "border-[#27e2a4]/30" };
  if (level <= 25) return { name: `Diamond ${26 - level}`, color: "text-[#00b4fc]", bg: "bg-[#00b4fc]/10", border: "border-[#00b4fc]/30" };
  return { name: `Ruby ${31 - level}`, color: "text-[#ff0062]", bg: "bg-[#ff0062]/10", border: "border-[#ff0062]/30" };
};

const formatElapsed = (ms: number) => `${(ms / 1000).toFixed(1)}초`;

const createInitialRecommendationSteps = (): RecommendationStep[] =>
  RECOMMENDATION_STEPS.map((step) => ({ ...step, status: 'pending' }));

const isRecommendationPrompt = (text: string) =>
  ['추천', '문제 줘', '문제 찾아', '풀 문제', '복습', '약점', '오늘의 추천'].some((keyword) => text.includes(keyword));

const stripRecommendationMarkup = (text: string) =>
  text
    .replace(/\[RECOMMENDATION_STEP:[^\]]+\]\n?/g, '')
    .replace(/\[SAFE_PROGRESS\][\s\S]*?\[\/SAFE_PROGRESS\]\n?/g, '')
    .replace(/\[THOUGHT\][\s\S]*?\[\/THOUGHT\]\n?/g, '')
    .replace(/\[LOAD_PROBLEM:\d+\]/g, '')
    .replace(/\[RECOMMENDATION_RESULT\][\s\S]*?\[\/RECOMMENDATION_RESULT\]/g, '')
    .trim();

const normalizeRecommendationResponse = (response: unknown, elapsedMs: number): RecommendationResult | null => {
  if (!response || typeof response !== 'object') return null;
  const data = response as Record<string, unknown>;
  const rawItems = Array.isArray(data.recommendations) ? data.recommendations : [];

  const recommendations = rawItems
    .slice(0, 3)
    .map((item, index): RecommendedProblem | null => {
      if (!item || typeof item !== 'object') return null;
      const raw = item as Record<string, unknown>;
      const problemId = raw.problemId ?? raw.problem_id ?? raw.id;
      if (typeof problemId !== 'string' && typeof problemId !== 'number') return null;

      const tags = Array.isArray(raw.tags) ? raw.tags.map(String).slice(0, 5) : [];
      return {
        rank: typeof raw.rank === 'number' ? raw.rank : index + 1,
        problemId,
        title: typeof raw.title === 'string' && raw.title.trim() ? raw.title : '제목 없음',
        tier: typeof raw.tier === 'string' ? raw.tier : undefined,
        level: typeof raw.level === 'number' ? raw.level : undefined,
        tags,
        reason: typeof raw.reason === 'string' && raw.reason.trim() ? raw.reason : '현재 학습 조건에 맞는 후보 문제입니다.',
        learningEffect: typeof raw.learningEffect === 'string' ? raw.learningEffect : undefined,
        solvedAcUrl: typeof raw.solvedAcUrl === 'string' ? raw.solvedAcUrl : undefined,
      };
    })
    .filter((item): item is RecommendedProblem => item !== null)
    .map((item, index) => ({ ...item, rank: index + 1 }));

  if (recommendations.length === 0) return null;
  return {
    recommendations,
    elapsedMs,
    summary: typeof data.summary === 'string' ? data.summary : undefined,
  };
};

const parseRecommendationResult = (text: string, elapsedMs: number): RecommendationResult | null => {
  const start = text.indexOf(RECOMMENDATION_RESULT_START);
  const end = text.indexOf(RECOMMENDATION_RESULT_END);
  if (start === -1 || end === -1 || end <= start) return null;

  const rawJson = text.substring(start + RECOMMENDATION_RESULT_START.length, end).trim();
  try {
    return normalizeRecommendationResponse(JSON.parse(rawJson), elapsedMs);
  } catch (error) {
    console.error('추천 결과 파싱 실패:', error);
    return null;
  }
};

function App() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const sendTimeRef = useRef<number>(0);
  const recommendationStartRef = useRef<number>(0);
  const recommendationTimerRef = useRef<number | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [recommendationStatus, setRecommendationStatus] = useState<RecommendationStatus>('idle');
  const [recommendationSteps, setRecommendationSteps] = useState<RecommendationStep[]>(createInitialRecommendationSteps);
  const [recommendationElapsedMs, setRecommendationElapsedMs] = useState(0);
  const [recommendationError, setRecommendationError] = useState('');
  const [isProcessOpen, setIsProcessOpen] = useState(true);

  const [activeTab, setActiveTab] = useState<'chat' | 'problem' | 'editor' | 'memo'>('chat');
  const [searchProblemId, setSearchProblemId] = useState('');
  const [problemData, setProblemData] = useState<ProblemData | null>(null);
  const [userCode, setUserCode] = useState('# 여기에 파이썬 코드를 작성하세요\n\nimport sys\n\ndef solution():\n    # input = sys.stdin.readline\n    pass\n\nif __name__ == "__main__":\n    solution()');
  const [judgeResults, setJudgeResults] = useState<JudgeResult[] | null>(null);
  
  // 메모 전용 상태
  const [memo, setMemo] = useState('');
  const [isSavingMemo, setIsSavingMemo] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // 1. 초기 로딩 시 DB에서 히스토리 가져오기
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/chat/history`);
        const data = await res.json();
        if (data.status === 'success') {
          const history = Array.isArray(data.history)
            ? data.history.map((msg: { role: string; text: string }) => ({
              role: msg.role === 'user' ? 'user' as const : 'ai' as const,
              text: stripRecommendationMarkup(msg.text || ''),
              recommendationResult: msg.role === 'ai' ? parseRecommendationResult(msg.text || '', 0) ?? undefined : undefined,
            }))
            : [];
          setMessages(history);
        }
      } catch (err) {
        console.error("히스토리 로드 실패:", err);
      }
    };
    fetchHistory();
  }, []);

  // 메시지 추가 시 또는 탭 전환 시 자동 스크롤
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, activeTab]);

  useEffect(() => {
    return () => {
      if (recommendationTimerRef.current) {
        window.clearInterval(recommendationTimerRef.current);
      }
    };
  }, []);

  const startRecommendationTimer = () => {
    recommendationStartRef.current = Date.now();
    setRecommendationElapsedMs(0);
    if (recommendationTimerRef.current) {
      window.clearInterval(recommendationTimerRef.current);
    }
    recommendationTimerRef.current = window.setInterval(() => {
      setRecommendationElapsedMs(Date.now() - recommendationStartRef.current);
    }, 100);
  };

  const stopRecommendationTimer = () => {
    if (recommendationTimerRef.current) {
      window.clearInterval(recommendationTimerRef.current);
      recommendationTimerRef.current = null;
    }
    if (recommendationStartRef.current) {
      setRecommendationElapsedMs(Date.now() - recommendationStartRef.current);
    }
  };

  const beginRecommendationFlow = () => {
    setRecommendationStatus('loading');
    setRecommendationSteps(createInitialRecommendationSteps());
    setRecommendationError('');
    setIsProcessOpen(true);
    startRecommendationTimer();
  };

  const completeRecommendationFlow = (status: RecommendationStatus) => {
    stopRecommendationTimer();
    setRecommendationStatus(status);
    setRecommendationSteps((prev) => prev.map((step) => {
      if (status === 'success') return { ...step, status: 'completed' };
      if (step.status === 'running') return { ...step, status: 'error' };
      return step;
    }));
  };

  const markRecommendationStepFromChunk = (chunk: string) => {
    const matches = [...chunk.matchAll(/\[RECOMMENDATION_STEP:([^\]]+)\]/g)];
    matches.forEach((match) => {
      const stepId = match[1];
      const stepIndex = RECOMMENDATION_STEPS.findIndex((step) => step.id === stepId);
      if (stepIndex === -1) return;
      setRecommendationStatus('reasoning');
      setRecommendationSteps((prev) => prev.map((step, index) => {
        if (index < stepIndex) return { ...step, status: 'completed' };
        if (index === stepIndex) return { ...step, status: 'running', timestamp: Date.now() };
        return step.status === 'completed' ? step : { ...step, status: 'pending' };
      }));
    });
  };

  const suggestions = [
    { icon: <Sparkles className="w-5 h-5 text-yellow-400" />, title: "오늘의 추천", text: "나의 현재 실력에 딱 맞는 플래티넘 도약용 문제 추천해줘." },
    { icon: <Brain className="w-5 h-5 text-purple-400" />, title: "약점 집중 보완", text: "최근에 많이 틀린 태그 위주로 어려운 문제 찾아줘." },
    { icon: <Code2 className="w-5 h-5 text-blue-400" />, title: "가벼운 두뇌 회전", text: "오늘은 머리 식히게 골드 하위 구현 문제 줘." },
  ];

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;
    const userText = input;
    const requestIsRecommendation = isRecommendationPrompt(userText);
    setInput('');
    setMessages(prev => [...prev, { role: 'user', text: userText }, { role: 'ai', text: '' }]);
    setIsLoading(true);
    sendTimeRef.current = Date.now();
    if (requestIsRecommendation) {
      beginRecommendationFlow();
    } else {
      setRecommendationStatus('idle');
      setRecommendationError('');
    }

    const currentProblemId = Number.parseInt(searchProblemId, 10);
    const chatPayload = {
      message: userText,
      history: messages.map(msg => ({role: msg.role, text: msg.text})),
      current_problem_id: Number.isFinite(currentProblemId) ? currentProblemId : undefined,
    };

    try {
      const res = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(chatPayload),
      });
      if (!res.body) throw new Error("스트리밍 오류");
      if (!res.ok) throw new Error("서버 응답 오류");
      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let done = false;
      setIsLoading(true);
      let streamedText = '';
      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        if (value) {
          const chunk = decoder.decode(value, { stream: true });
          streamedText += chunk;
          if (requestIsRecommendation) {
            markRecommendationStepFromChunk(chunk);
          }
          setMessages(prev => {
            const newMessages = [...prev];
            const lastIdx = newMessages.length - 1;
            const updatedText = newMessages[lastIdx].text + chunk;
            newMessages[lastIdx] = { ...newMessages[lastIdx], text: updatedText };
            return newMessages;
          });
        }
      }
      // 응답 시간 계산 후 마지막 AI 메시지에 기록
      const elapsedMs = Date.now() - sendTimeRef.current;
      const elapsed = Math.round(elapsedMs / 1000);
      const recommendationResult = requestIsRecommendation
        ? parseRecommendationResult(streamedText, elapsedMs)
        : null;
      setMessages(prev => {
        const updated = [...prev];
        const lastMessage = updated[updated.length - 1];
        updated[updated.length - 1] = {
          ...lastMessage,
          text: stripRecommendationMarkup(lastMessage.text),
          elapsed,
          recommendationResult: recommendationResult ?? undefined,
        };
        return updated;
      });
      if (requestIsRecommendation) {
        if (recommendationResult) {
          completeRecommendationFlow('success');
        } else {
          setRecommendationError('추천 결과를 카드로 정리하지 못했습니다. 다시 시도해 주세요.');
          completeRecommendationFlow('error');
        }
      }
      setIsLoading(false);
    } catch (error) {
      console.error("통신 오류:", error);
      setMessages(prev => {
        const updated = [...prev];
        const lastIdx = updated.length - 1;
        if (updated[lastIdx]?.role === 'ai') {
          updated[lastIdx] = {
            ...updated[lastIdx],
            text: '서버와 통신하는 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요.',
          };
        }
        return updated;
      });
      if (requestIsRecommendation) {
        setRecommendationError('추천 요청이 실패했습니다. 네트워크 상태나 Gemini API 설정을 확인한 뒤 다시 시도해 주세요.');
        completeRecommendationFlow('error');
      }
      setIsLoading(false);
    }
  };

  const handleClearHistory = async () => {
    if (!confirm("모든 대화 내역을 삭제하시겠습니까?")) return;
    try {
      const res = await fetch(`${API_BASE_URL}/chat/history`, { method: 'DELETE' });
      if (res.ok) {
        setMessages([]);
      }
    } catch (err) {
      console.error("히스토리 삭제 실패:", err);
    }
  };

  const fetchMemo = async (pid: number) => {
    try {
      const res = await fetch(`${API_BASE_URL}/memo/${pid}`);
      const data = await res.json();
      setMemo(data.content || '');
    } catch (error) { console.error("메모 로드 실패:", error); }
  };

  const loadProblem = async (specificId?: string) => {
    const targetId = specificId || searchProblemId;
    if (!targetId || !targetId.trim()) return;
    try {
      const res = await fetch(`${API_BASE_URL}/problem/${targetId}`);
      if (!res.ok) throw new Error("문제를 찾을 수 없습니다.");
      const response = await res.json();
      if (response.status === 'success') {
        setProblemData(response.data);
        fetchMemo(parseInt(targetId));
        if (specificId) setSearchProblemId(targetId);
      }
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : '알 수 없는 오류';
      alert("문제 로드 실패: " + message);
    }
  };

  const openProblem = (problem: RecommendedProblem) => {
    const problemId = String(problem.problemId);
    setActiveTab('problem');
    loadProblem(problemId);
  };

  const handleJudge = async () => {
    if (!searchProblemId || isLoading) return;
    setIsLoading(true);
    setJudgeResults(null);
    try {
      const res = await fetch(`${API_BASE_URL}/judge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ problem_id: parseInt(searchProblemId), code: userCode }),
      });
      const response = await res.json();
      if (response.status === 'success') setJudgeResults(response.results);
      else alert(response.detail || "채점 중 오류 발생");
    } catch { alert("서버 연결 실패"); } finally { setIsLoading(false); }
  };

  const handleSaveMemo = async () => {
    if (!searchProblemId) return;
    setIsSavingMemo(true);
    try {
      await fetch(`${API_BASE_URL}/memo`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ problem_id: parseInt(searchProblemId), content: memo }),
      });
      alert("학습 내용이 저장되었습니다.");
    } catch { alert("메모 저장 실패"); } finally { setIsSavingMemo(false); }
  };

  const renderStepIcon = (status: RecommendationStepStatus) => {
    if (status === 'completed') return <CheckCircle2 className="w-4 h-4 text-green-400" />;
    if (status === 'running') return <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />;
    if (status === 'error') return <AlertCircle className="w-4 h-4 text-red-400" />;
    return <Circle className="w-4 h-4 text-gray-600" />;
  };

  const renderRecommendationProgress = () => {
    if (recommendationStatus === 'idle') return null;
    const isFinished = recommendationStatus === 'success' || recommendationStatus === 'error';
    const activeStep = recommendationSteps.find((step) => step.status === 'running');
    const title = recommendationStatus === 'success'
      ? `추천 완료 · 총 ${formatElapsed(recommendationElapsedMs)}`
      : recommendationStatus === 'error'
        ? `추천 실패 · ${formatElapsed(recommendationElapsedMs)} 경과`
        : `추천 분석 중 · ${formatElapsed(recommendationElapsedMs)} 경과`;

    return (
      <div className="w-full max-w-4xl mb-4 rounded-2xl border border-blue-500/20 bg-[#1A1B1E] p-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="flex items-center gap-2 text-sm font-semibold text-blue-300">
              <Clock3 className="w-4 h-4" />
              {title}
            </div>
            <p className="mt-1 text-xs text-gray-400">
              {recommendationError || activeStep?.label || '추천 결과를 카드로 정리하고 있습니다.'}
            </p>
          </div>
          <button
            type="button"
            onClick={() => setIsProcessOpen((prev) => !prev)}
            className="inline-flex items-center gap-2 rounded-xl border border-white/10 px-3 py-2 text-xs text-gray-300 hover:bg-white/5"
          >
            <ListChecks className="w-4 h-4" />
            추천 과정 보기
          </button>
        </div>
        {isProcessOpen && (
          <div className="mt-4 space-y-2 border-t border-white/5 pt-4">
            {recommendationSteps.map((step) => (
              <div key={step.id} className="flex items-start gap-3 text-xs">
                <div className="mt-0.5">{renderStepIcon(step.status)}</div>
                <div className={step.status === 'pending' ? 'text-gray-500' : 'text-gray-300'}>
                  {step.label}
                </div>
              </div>
            ))}
            {isFinished && recommendationStatus === 'success' && (
              <p className="pt-1 text-xs text-green-400">추천 결과를 확인한 뒤 원하는 문제를 직접 선택하세요.</p>
            )}
          </div>
        )}
      </div>
    );
  };

  const renderRecommendationCards = (result?: RecommendationResult): React.ReactNode => {
    if (!result) return null;
    return (
      <div className="mt-4 space-y-3">
        <div>
          <div className="text-sm font-semibold text-white">오늘의 추천 문제 Top {result.recommendations.length}</div>
          <div className="text-xs text-gray-500">추천 완료 · 총 {formatElapsed(result.elapsedMs)}</div>
          {result.summary && <p className="mt-2 text-xs leading-relaxed text-gray-400">{result.summary}</p>}
        </div>
        <div className="grid grid-cols-1 gap-3">
          {result.recommendations.map((problem) => {
            const tierInfo = getTierInfo(problem.level || problem.tier);
            return (
              <div key={`${problem.rank}-${problem.problemId}`} className="rounded-2xl border border-gray-800 bg-black/20 p-4">
                <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                  <div className="min-w-0">
                    <div className="mb-2 flex flex-wrap items-center gap-2">
                      <span className="rounded-lg border border-blue-400/30 bg-blue-400/10 px-2.5 py-1 text-xs font-bold text-blue-300">
                        {problem.rank}순위
                      </span>
                      <span className={`rounded-lg border px-2.5 py-1 text-xs font-bold ${tierInfo.bg} ${tierInfo.color} ${tierInfo.border}`}>
                        {tierInfo.name}
                      </span>
                    </div>
                    <h3 className="break-words text-base font-bold text-gray-100">
                      {problem.problemId}. {problem.title}
                    </h3>
                  </div>
                  <button
                    type="button"
                    onClick={() => openProblem(problem)}
                    className="inline-flex shrink-0 items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-blue-500 via-purple-500 to-purple-600 px-3 py-2 text-xs font-bold text-white shadow-lg shadow-purple-500/20 hover:opacity-90"
                  >
                    <ExternalLink className="w-4 h-4" />
                    이 문제 풀러가기
                  </button>
                </div>
                {problem.tags && problem.tags.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {problem.tags.map((tag) => (
                      <span key={`${problem.problemId}-${tag}`} className="rounded-full border border-purple-500/20 bg-purple-500/10 px-2.5 py-1 text-[11px] text-purple-200">
                        #{tag}
                      </span>
                    ))}
                  </div>
                )}
                <p className="mt-3 text-xs leading-relaxed text-gray-300">{problem.reason}</p>
                {problem.learningEffect && (
                  <p className="mt-2 text-xs leading-relaxed text-gray-500">기대 학습 효과: {problem.learningEffect}</p>
                )}
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className="h-[100dvh] flex flex-col bg-[#131314] text-gray-100 font-sans overflow-hidden">
      <div className="flex-1 overflow-y-auto custom-scrollbar pb-24">
        
        {/* Tab 1: AI 채팅 */}
        {activeTab === 'chat' && (
          <div className="max-w-4xl mx-auto p-6 flex flex-col items-center">
            {messages.length === 0 ? (
              <div className="mt-10 md:mt-20 w-full animate-fade-in-up">
                <div className="flex flex-col items-center text-center mb-10">
                  <div>
                    <h1 className="text-3xl font-semibold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-purple-400 to-purple-600">안녕하세요, 태영님</h1>
                    <p className="text-transparent bg-clip-text bg-gradient-to-r from-gray-400 to-gray-500 text-sm">최상의 알고리즘 퍼포먼스를 위해 무엇을 도와드릴까요?</p>
                  </div>
                  {messages.length > 0 && (
                    <button onClick={handleClearHistory} className="px-4 py-2 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 rounded-xl text-red-400 text-xs transition-all">
                      대화 초기화
                    </button>
                  )}
                </div>
                <div className="grid grid-cols-1 gap-3 w-full max-w-2xl mx-auto">
                  {suggestions.map((item, idx) => (
                    <button key={idx} onClick={() => setInput(item.text)} className="flex items-center p-4 bg-[#1E1F20] rounded-2xl border border-gray-800/50 hover:bg-[#2A2B2F] transition-all text-left">
                      <div className="mr-4 p-2 bg-[#131314] rounded-xl">{item.icon}</div>
                      <div>
                        <div className="font-semibold text-sm text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">{item.title}</div>
                        <div className="text-xs text-gray-500">{item.text}</div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="w-full max-w-4xl">
                <div className="flex justify-end mb-4">
                  <button onClick={handleClearHistory} className="px-3 py-1 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 rounded-lg text-red-400 text-xs transition-all">
                    대화 초기화
                  </button>
                </div>
                {renderRecommendationProgress()}
                <div ref={scrollRef} className="w-full space-y-6 py-4 overflow-y-auto max-h-[70vh] scroll-smooth custom-scrollbar">
                  {messages.map((msg, i) => {
                    const contentText = msg.role === 'ai' ? stripRecommendationMarkup(msg.text) : msg.text;
                    const isStreaming = isLoading && i === messages.length - 1;

                    return (
                      <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className={`max-w-[85%] rounded-2xl ${msg.role === 'user' ? 'bg-gradient-to-br from-blue-500 via-purple-500 to-purple-600 text-white shadow-lg p-4' : ''}`}>
                          {msg.role === 'ai' && isStreaming && !contentText ? (
                            <div className="p-4 bg-[#1A1B1E] border border-blue-500/20 rounded-2xl animate-pulse-subtle">
                              <div className="flex items-center gap-2 mb-2 text-blue-400 font-semibold text-sm">
                                <Brain className="w-4 h-4 animate-spin" style={{ animationDuration: '3s' }} /> 분석 중...
                              </div>
                              <p className="text-xs text-gray-400 leading-relaxed">질문을 분석하고 필요한 데이터를 확인하고 있습니다.</p>
                              <div className="flex gap-1 items-center mt-3">
                                <div className="w-1.5 h-1.5 bg-blue-400/50 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                                <div className="w-1.5 h-1.5 bg-blue-400/50 rounded-full animate-bounce" style={{ animationDelay: '200ms' }}></div>
                                <div className="w-1.5 h-1.5 bg-blue-400/50 rounded-full animate-bounce" style={{ animationDelay: '400ms' }}></div>
                              </div>
                            </div>
                          ) : msg.role === 'ai' ? (
                            <div className="p-4 bg-[#1E1F20] text-gray-200 border border-gray-800 rounded-2xl">
                              {contentText && <p className="text-sm leading-relaxed whitespace-pre-wrap">{contentText}</p>}
                              {renderRecommendationCards(msg.recommendationResult)}
                              {msg.elapsed !== undefined && (
                                <p className="text-[10px] text-gray-600 mt-3 text-right">응답 시간: {msg.elapsed}초</p>
                              )}
                            </div>
                          ) : (
                            // 사용자 메시지
                            <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.text}</p>
                          )}
                        </div>
                      </div>
                    );
                  })}
              </div>
          </div>
            )}
          </div>
        )}

        {/* Tab 2: 워크스페이스 */}
        {activeTab === 'problem' && (
          <div className="p-6">
            <div className="flex gap-2 mb-8 bg-[#1E1F20] p-1.5 rounded-2xl border border-gray-800 focus-within:border-purple-500/50 transition-all">
              <input type="number" value={searchProblemId} onChange={(e) => setSearchProblemId(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && loadProblem()} placeholder="문제 번호 입력" className="flex-1 bg-transparent p-2 outline-none text-white text-sm" />
              <button onClick={() => loadProblem()} className="bg-gradient-to-r from-blue-500 via-purple-500 to-purple-600 hover:opacity-90 p-2.5 rounded-xl transition-all shadow-lg shadow-purple-500/20"><Search className="w-4 h-4 text-white" /></button>
            </div>
            {problemData && (
              <div className="space-y-8 text-sm">
                
                {/* 문제 상단 정보 (제목, 티어, 태그) */}
                <div className="mb-6 space-y-4">
                  <div className="flex flex-wrap items-center gap-3">
                    <span className={`px-2.5 py-1 text-xs font-bold rounded-lg border shadow-sm ${getTierInfo(problemData.tier).bg} ${getTierInfo(problemData.tier).color} ${getTierInfo(problemData.tier).border}`}>
                      {getTierInfo(problemData.tier).name}
                    </span>
                    <h2 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-white to-gray-300">
                      {searchProblemId}. {problemData.title || '문제'}
                    </h2>
                    {problemData.is_solved && (
                      <span className="flex items-center gap-1.5 px-3 py-1 text-xs font-bold text-green-400 bg-green-500/10 border border-green-500/30 rounded-full shadow-sm">
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        Solved!
                      </span>
                    )}
                  </div>
                  
                  {problemData.tags && problemData.tags.length > 0 && (
                    <div className="flex flex-wrap gap-2 mt-2">
                      {problemData.tags.map((tag: string, idx: number) => (
                        <span key={idx} className="px-3 py-1 text-xs text-purple-300 bg-purple-500/10 border border-purple-500/30 rounded-full shadow-sm shadow-purple-500/10 backdrop-blur-sm">
                          #{tag}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                <div className="space-y-4">
                  <h3 className="text-lg font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400 inline-block border-b border-purple-500/30 pb-1">문제 설명</h3>
                  <div className="prose prose-invert prose-sm max-w-none text-gray-300 leading-relaxed" dangerouslySetInnerHTML={{ __html: problemData.description }} />
                </div>
                <div className="grid grid-cols-1 gap-4">
                  <div className="bg-[#1E1F20] p-5 rounded-2xl border border-gray-800">
                    <h3 className="text-lg font-bold mb-2 text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">입력</h3>
                    <div className="prose prose-invert" dangerouslySetInnerHTML={{ __html: problemData.input_desc }} />
                  </div>
                  <div className="bg-[#1E1F20] p-5 rounded-2xl border border-gray-800">
                    <h3 className="text-lg font-bold mb-2 text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">출력</h3>
                    <div className="prose prose-invert" dangerouslySetInnerHTML={{ __html: problemData.output_desc }} />
                  </div>
                </div>
                {/* 제한 입출력 */}
                {/* App.tsx 내의 문제 설명 렌더링 부분 상단에 추가 */}
                {problemData.problem_limit && (
                  <div className="mb-6 p-4 bg-purple-500/5 border border-purple-500/20 rounded-2xl">
                    <h3 className="text-sm font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400 mb-2 flex items-center gap-2">
                      <Brain className="w-4 h-4" /> 제약 조건
                    </h3>
                    <div 
                      className="prose prose-invert prose-xs text-gray-400" 
                      dangerouslySetInnerHTML={{ __html: problemData.problem_limit }} 
                    />
                  </div>
                )}
                {/* 예제 입출력 */}
                <div className="space-y-4">
                  <h3 className="text-lg font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400 ml-1">예제 입출력</h3>
                  {problemData.sample_inputs && problemData.sample_inputs.map((sampleIn: string, idx: number) => (
                    <div key={idx} className="flex flex-col gap-2">
                      <div className="flex gap-2">
                        <div className="flex-1 bg-black/30 p-3 rounded-lg border border-gray-800 font-mono text-xs whitespace-pre-wrap">
                          <div className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400/80 to-purple-400/80 mb-2 font-sans">예제 입력 {idx + 1}</div>
                          {sampleIn}
                        </div>
                        <div className="flex-1 bg-black/30 p-3 rounded-lg border border-gray-800 font-mono text-xs whitespace-pre-wrap">
                          <div className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400/80 to-purple-400/80 mb-2 font-sans">예제 출력 {idx + 1}</div>
                          {problemData.sample_outputs[idx]}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab 3: 코드 에디터 */}
        {activeTab === 'editor' && (
          <div className="p-6 space-y-4 h-full flex flex-col">
            <div className="flex justify-between items-center">
              <h3 className="text-lg font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400 inline-block">코드 에디터</h3>
              <button onClick={handleJudge} disabled={isLoading} className="px-4 py-2 bg-gradient-to-r from-blue-500 via-purple-500 to-purple-600 rounded-xl text-xs font-bold hover:opacity-90 transition-all shadow-lg shadow-purple-500/20 disabled:opacity-50 text-white">
                {isLoading ? '채점 중...' : '코드 채점하기'}
              </button>
            </div>
            <div className="flex-1 bg-[#1E1F20] rounded-2xl border border-gray-800 overflow-hidden font-mono shadow-inner min-h-[400px]">
              <CodeMirror
                value={userCode}
                height="100%"
                theme={vscodeDark}
                extensions={[python()]}
                onChange={(value) => setUserCode(value)}
                className="text-sm h-full"
                basicSetup={{
                  lineNumbers: true,
                  foldGutter: true,
                  dropCursor: true,
                  allowMultipleSelections: true,
                  indentOnInput: true,
                  syntaxHighlighting: true,
                  bracketMatching: true,
                  closeBrackets: true,
                  autocompletion: true,
                  rectangularSelection: true,
                  crosshairCursor: true,
                  highlightActiveLine: true,
                  highlightSelectionMatches: true,
                  closeBracketsKeymap: true,
                  defaultKeymap: true,
                  searchKeymap: true,
                  historyKeymap: true,
                  foldKeymap: true,
                  completionKeymap: true,
                  lintKeymap: true,
                }}
              />
            </div>
            {judgeResults && (
              <div className="mt-4 space-y-2">
                {judgeResults.map((res, idx) => (
                  <div key={idx} className={`p-4 rounded-xl border ${res.result === 'Success' ? 'border-green-500/30 bg-green-500/5' : 'border-red-500/30 bg-red-500/5'}`}>
                    <div className="flex justify-between items-center text-xs font-bold">
                      <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">케이스 #{res.case}</span>
                      <span className={res.result === 'Success' ? 'text-green-400' : 'text-red-400'}>{res.result === 'Success' ? '✅ 맞았습니다' : '❌ ' + res.result}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Tab 4: 학습 노트 (디자인 업데이트된 버튼) */}
        {activeTab === 'memo' && (
          <div className="p-6 h-full flex flex-col space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="text-lg font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400 inline-block">학습 노트</h3>
              <button 
                onClick={handleSaveMemo} 
                disabled={isSavingMemo || !searchProblemId} 
                className="px-4 py-2 bg-gradient-to-r from-blue-500 via-purple-500 to-purple-600 rounded-xl text-xs font-bold hover:opacity-90 transition-all shadow-lg shadow-purple-500/20 disabled:opacity-50 text-white"
              >
                {isSavingMemo ? '저장 중...' : '메모 저장'}
              </button>
            </div>
            <div className="flex-1 bg-[#1E1F20] rounded-2xl border border-gray-800 overflow-hidden shadow-inner">
              <CodeMirror
                value={memo}
                height="100%"
                theme={vscodeDark}
                extensions={[markdown()]}
                onChange={(value) => setMemo(value)}
                className="text-sm h-full"
                placeholder="이 문제의 핵심 아이디어를 기록하세요..."
              />
            </div>
          </div>
        )}
      </div>

      {/* 플로팅 하단 바 */}
      <div className="fixed bottom-6 left-0 right-0 px-6 pointer-events-none flex justify-center">
        <div className={`bg-[#1E1F20]/80 backdrop-blur-xl border border-white/10 rounded-3xl p-2 shadow-2xl flex items-center pointer-events-auto transition-all duration-300 ${activeTab === 'chat' ? 'w-full max-w-3xl' : 'w-fit'}`}>
          {activeTab === 'chat' && (
            <div className="mr-2 flex h-14 min-w-0 flex-1 items-center rounded-2xl border border-white/5 bg-[#131314]/50 px-3">
              <textarea ref={textareaRef} value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSend())} placeholder="메시지 입력..." className="h-10 max-h-10 min-w-0 flex-1 resize-none overflow-y-auto bg-transparent py-1.5 pr-2 text-xs leading-5 text-gray-300 outline-none custom-scrollbar" rows={2} />
              <button onClick={handleSend} disabled={!input.trim() || isLoading} className="shrink-0 p-2 hover:scale-110 transition-all disabled:opacity-50"><div className="bg-gradient-to-r from-blue-400 via-purple-400 to-purple-600 rounded-full p-1.5 shadow-lg"><Send className="w-3.5 h-3.5 text-white" /></div></button>
            </div>
          )}
          <div className="flex gap-1">
            <button onClick={() => setActiveTab('chat')} className={`p-3 rounded-2xl relative transition-all ${activeTab === 'chat' ? 'text-white' : 'text-gray-500 hover:text-gray-300'}`}>{activeTab === 'chat' && <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 via-purple-500/20 to-purple-600/20 rounded-2xl animate-pulse" />}<MessageSquare className="w-5 h-5 relative z-10" /></button>
            <button onClick={() => setActiveTab('problem')} className={`p-3 rounded-2xl relative transition-all ${activeTab === 'problem' ? 'text-white' : 'text-gray-500 hover:text-gray-300'}`}>{activeTab === 'problem' && <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 via-purple-500/20 to-purple-600/20 rounded-2xl animate-pulse" />}<FileCode2 className="w-5 h-5 relative z-10" /></button>
            <button onClick={() => setActiveTab('editor')} className={`p-3 rounded-2xl relative transition-all ${activeTab === 'editor' ? 'text-white' : 'text-gray-500 hover:text-gray-300'}`}>{activeTab === 'editor' && <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 via-purple-500/20 to-purple-600/20 rounded-2xl animate-pulse" />}<Code className="w-5 h-5 relative z-10" /></button>
            <button onClick={() => setActiveTab('memo')} className={`p-3 rounded-2xl relative transition-all ${activeTab === 'memo' ? 'text-white' : 'text-gray-500 hover:text-gray-300'}`}>{activeTab === 'memo' && <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 via-purple-500/20 to-purple-600/20 rounded-2xl animate-pulse" />}<Notebook className="w-5 h-5 relative z-10" /></button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
