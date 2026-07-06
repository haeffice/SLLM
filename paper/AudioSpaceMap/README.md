# Audio Space Map 계열

오디오(음향 에코·공간 음향)에 기반하여 주어진 공간의 **space map**(floorplan,
방 기하 등)을 생성·추론하는 논문 구현 모음.

| 디렉터리 | 논문 / 내용 |
|---|---|
| `EchoScan` | 다채널 RIR(음향 에코) → 2-D floorplan + 1-D height map. EchoScan (IEEE/ACM TASLP 2024, arXiv:2310.11728). |
| `BatVision` | 양이(binaural) 에코 → 정면 시야 depth map. BatVision (ICRA 2020 arXiv:1912.07011 / IROS 2023 데이터셋 arXiv:2303.07257). 실측 공개 데이터셋. |

두 접근의 상보성: **EchoScan**은 *시뮬레이션 다채널 RIR → top-down floorplan*,
**BatVision**은 *실측 양이 녹음 → 정면 depth map*. 데이터(시뮬 vs 실측)·출력(평면도 vs
깊이)·백본(1-D ResNet+Multi-Aggregation vs 2-D U-Net)이 모두 다르다.
