# Spatial Audio + LLM 계열

다채널/공간 오디오를 **멀티모달 LLM**이 이해·추론할 수 있는 표현(spatial audio token)
으로 인코딩하는 논문 구현 모음.

| 디렉터리 | 논문 / 내용 |
|---|---|
| `PhaseCoder` | 마이크 배열 형상 무관 공간 오디오 인코더 → spatial token → 방위/고도/거리 + Gemma 3n 주입. PhaseCoder (Google DeepMind & Google AR, 2026, arXiv:2601.21124). |

`paper/AudioSpaceMap/`(오디오 → 공간 맵 생성)와 달리, 여기서는 음원의 **방향·거리**를
추정하고 그 표현을 **LLM에 연결**하는 데 초점을 둔다.
