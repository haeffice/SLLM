"""POST /chat — 시뮬레이션 분석 결과에 대한 LLM 기반 QA.

FE가 만든 압축 분석 요약(fe/analysis.py to_summary_json)과 질문을 받아,
CHAT_LLM_* env가 설정돼 있으면 OpenAI-호환 chat/completions로 답하고
(mode="llm"), 미설정이거나 호출이 실패하면 utils.chat_fallback의 rule-based
답변으로 강등한다(mode="fallback", error에 사유). LLM 실패는 5xx가 아니라
in-band 폴백 — 데모가 외부 API 상태에 좌우되지 않게 (predict/simulate의
LIVE/DUMMY 이원 구조와 같은 철학).

mesh 모델 레지스트리와 무관 — 모델 로딩 중에도 동작한다. blocking HTTP 호출은
routers/predict.py의 CPU-bound 규약과 동일하게 run_in_executor로 뺀다.
"""

from __future__ import annotations

import asyncio
import json
import logging

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config import chat_llm_config
from utils.chat_fallback import fallback_answer

log = logging.getLogger("be.chat")
router = APIRouter()

MAX_HISTORY = 12  # 방어적 상한 — FE는 최근 6턴만 보내지만 서버도 캡을 둔다.

_SYSTEM_PROMPT = (
    "너는 물리 충격 시뮬레이션 결과를 설명하는 분석 어시스턴트다. "
    "아래 분석 JSON만 근거로 한국어로 간결하게 답하라. JSON에 없는 내용은 "
    "추측하지 말고 모른다고 답하라. 수치는 4자리 유효숫자로 표기하라.\n"
    "분석 JSON:\n{analysis_json}"
)


class ChatRequest(BaseModel):
    question: str = Field(..., description="분석 결과에 대한 자연어 질문")
    analysis: dict = Field(
        default_factory=dict,
        description="FE AnalysisResult.to_summary_json() 압축 요약 (없으면 빈 dict)",
    )
    history: list[dict] = Field(
        default_factory=list,
        description='선택: 최근 대화 [{"role": "user"|"assistant", "content": str}]',
    )


class ChatResponse(BaseModel):
    success: bool
    answer: str
    mode: str  # "llm" | "fallback"
    model: str | None = None
    error: str | None = None  # LLM 실패로 폴백된 경우 사유


def _build_messages(question: str, analysis: dict, history: list[dict]) -> list[dict]:
    """system(분석 JSON 임베드) + 최근 history + user 질문."""
    analysis_json = json.dumps(analysis or {}, ensure_ascii=False)
    messages = [{"role": "system", "content": _SYSTEM_PROMPT.format(analysis_json=analysis_json)}]
    for turn in history[-MAX_HISTORY:]:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def _call_llm(cfg: dict, messages: list[dict]) -> str:
    """OpenAI-호환 chat/completions 동기 호출. 실패는 RuntimeError로 통일."""
    headers = {"Content-Type": "application/json"}
    if cfg["api_key"]:
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    body = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 512,
    }
    resp = requests.post(
        f"{cfg['base_url']}/chat/completions",
        json=body,
        headers=headers,
        timeout=cfg["timeout"],
    )
    if resp.status_code != 200:
        raise RuntimeError(f"LLM API HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        answer = resp.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"LLM API malformed response: {e}") from e
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("LLM API returned empty answer")
    return answer.strip()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="empty question")

    cfg = chat_llm_config()
    if cfg is None:
        log.info("chat fallback (LLM not configured): %.60s", question)
        return ChatResponse(
            success=True, answer=fallback_answer(question, req.analysis), mode="fallback"
        )

    messages = _build_messages(question, req.analysis, req.history)
    log.info("chat start: model=%s, history=%d, %.60s", cfg["model"], len(req.history), question)
    try:
        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(None, _call_llm, cfg, messages)
    except Exception as e:  # 네트워크/타임아웃/응답 형식 — 전부 in-band 폴백
        log.warning("chat LLM call failed → rule-based fallback: %s", e)
        return ChatResponse(
            success=True,
            answer=fallback_answer(question, req.analysis),
            mode="fallback",
            error=str(e),
        )
    log.info("chat done: model=%s, answer_len=%d", cfg["model"], len(answer))
    return ChatResponse(success=True, answer=answer, mode="llm", model=cfg["model"])
