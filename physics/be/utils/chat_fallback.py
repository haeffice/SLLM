"""분석 결과 JSON 기반 rule-based QA 폴백 (LLM 미설정/실패/서버 미연결 시 사용).

이 파일은 BE(be/utils/chat_fallback.py)와 FE(fe/chat_fallback.py)에 바이트
단위로 동일하게 존재한다 — fe/app.py의 metal_dent 수식 미러와 같은 규약으로,
한쪽만 고치면 안 된다(온라인/오프라인 답변이 달라짐). 수정 시 두 파일을 함께
갱신할 것. stdlib만 사용(numpy 금지) — 미러 유지가 쉽도록.

입력 analysis는 fe/analysis.py AnalysisResult.to_summary_json()의 압축 요약:
  {scenario, sim_mode, num_frames, num_nodes, single_frame, fallback_whole_mesh,
   overall: {max_disp, node, verdict},
   components: [{id, name, material, max_disp, threshold, ratio, status,
                 max_node, score, notes}, ...]}

라우팅(결정적 — 같은 질문+분석이면 항상 같은 답):
  1) 질문에 부품명(name/id) 포함 → 해당 부품 상세
  2) 파손/손상 계열 키워드 → FAIL/WARN 부품 나열
  3) 최대/최악 계열 키워드 → 최악 부품(ratio 기준) + 위치
  4) 변위/수치 계열 키워드 → 전체 통계
  5) 그 외 → 종합 요약
"""

from __future__ import annotations

_NO_ANALYSIS_MSG = (
    "아직 시뮬레이션 결과가 없습니다. 메쉬에서 충격점을 클릭해 고른 뒤 "
    "Simulate를 먼저 실행해 주세요."
)

_DAMAGE_KEYWORDS = ("파손", "손상", "깨", "부서", "부러", "균열", "괜찮", "damage", "fail", "broke", "crack")
_WORST_KEYWORDS = ("최대", "가장", "제일", "최악", "어디", "worst", "max", "where")
_STAT_KEYWORDS = ("변위", "수치", "얼마", "통계", "displacement", "how much", "stats")


def _fmt(value) -> str:
    """숫자를 4 유효자리로 포맷. 숫자가 아니면 그대로 문자열화."""
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def _comp_line(comp: dict) -> str:
    return (
        f"{comp.get('name', '?')}: 최대 변위 {_fmt(comp.get('max_disp'))}"
        f" / 임계값 {_fmt(comp.get('threshold'))}"
        f" (비율 {_fmt(comp.get('ratio'))}) → {comp.get('status', '?')}"
    )


def _comp_detail(comp: dict) -> str:
    lines = [_comp_line(comp)]
    if comp.get("material"):
        lines.append(f"재질: {comp['material']}")
    node = comp.get("max_node")
    if isinstance(node, (int, float)) and not isinstance(node, bool) and node >= 0:
        lines.append(f"최대 충격 위치: 노드 #{int(node)}")
    if comp.get("score") is not None:
        lines.append(f"충격 점수: {_fmt(comp['score'])}/100")
    if comp.get("notes"):
        lines.append(f"비고: {comp['notes']}")
    return "\n".join(lines)


def _worst_comp(comps: list[dict]) -> dict:
    def ratio(c: dict) -> float:
        try:
            return float(c.get("ratio", 0.0))
        except (TypeError, ValueError):
            return 0.0

    return max(comps, key=ratio)


def _summary(analysis: dict, comps: list[dict]) -> str:
    overall = analysis.get("overall") or {}
    fails = [c for c in comps if c.get("status") == "FAIL"]
    warns = [c for c in comps if c.get("status") == "WARN"]
    verdict = overall.get("verdict") or ("FAIL" if fails else "WARN" if warns else "OK")

    lines = [
        f"종합 판정: {verdict} — 부품 {len(comps)}개 중 "
        f"FAIL {len(fails)}개, WARN {len(warns)}개."
    ]
    if overall.get("max_disp") is not None:
        lines.append(
            f"전체 최대 변위 {_fmt(overall['max_disp'])}"
            + (f" (노드 #{overall['node']})" if overall.get("node") is not None else "")
        )
    if comps:
        lines.append("최악 부품 — " + _comp_line(_worst_comp(comps)))
    if analysis.get("single_frame"):
        lines.append("(단일 프레임 결과 — 시간에 따른 속도 지표는 없습니다.)")
    if analysis.get("fallback_whole_mesh"):
        lines.append("(부품 정의가 없는 메쉬 — 전체를 하나의 부품으로 간주한 자동 임계값 분석입니다.)")
    return "\n".join(lines)


def fallback_answer(question: str, analysis: dict) -> str:
    """질문 + 압축 분석 요약 → 결정적 rule-based 한국어 답변."""
    q = (question or "").strip().lower()
    if not isinstance(analysis, dict) or not analysis.get("components"):
        return _NO_ANALYSIS_MSG
    comps = [c for c in analysis["components"] if isinstance(c, dict)]
    if not comps:
        return _NO_ANALYSIS_MSG

    # 1) 부품명 직접 언급 → 해당 부품 상세
    for comp in comps:
        name = str(comp.get("name", "")).strip().lower()
        cid = str(comp.get("id", "")).strip().lower()
        if (name and name in q) or (cid and cid in q):
            return _comp_detail(comp)

    # 2) 파손/손상 → 임계 초과(FAIL)·경고(WARN) 부품 나열
    if any(k in q for k in _DAMAGE_KEYWORDS):
        fails = [c for c in comps if c.get("status") == "FAIL"]
        warns = [c for c in comps if c.get("status") == "WARN"]
        if not fails and not warns:
            return "모든 부품이 임계값 이내입니다 (전 부품 OK). 파손 위험은 낮습니다."
        lines = []
        if fails:
            lines.append("임계값을 초과한(파손 위험) 부품:")
            lines += ["  - " + _comp_line(c) for c in fails]
        if warns:
            lines.append("경고 수준(임계값 근접) 부품:")
            lines += ["  - " + _comp_line(c) for c in warns]
        return "\n".join(lines)

    # 3) 최대/최악 → ratio 최대 부품 + 위치
    if any(k in q for k in _WORST_KEYWORDS):
        return "충격이 가장 심한 부품 —\n" + _comp_detail(_worst_comp(comps))

    # 4) 변위/수치 → 전체 통계
    if any(k in q for k in _STAT_KEYWORDS):
        lines = [f"프레임 {analysis.get('num_frames', '?')}개, 노드 {analysis.get('num_nodes', '?')}개 분석."]
        lines += ["  - " + _comp_line(c) for c in comps]
        return "\n".join(lines)

    # 5) 기본 — 종합 요약
    return _summary(analysis, comps)
