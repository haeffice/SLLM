# Physics Impact Simulator — FE (Windows desktop)

PySide6 + PyVista 데스크톱 시뮬레이터. 로컬 메쉬를 열어 3D로 보고, 노드를
클릭해 충격점을 고른 뒤 force를 주면 BE `/predict`로 변형 메쉬를 받아
원본(wireframe) 위에 결과(solid)를 겹쳐 보여준다.

**모델 연결 없이도 동작한다.** 시작 시 내장 더미 메쉬(21×21 grid)가 자동으로 떠서
노드 피킹·force 입력·변형 표시를 바로 확인할 수 있다. BE/모델이 준비되지 않은
동안에는 **Simulate가 로컬에서 동일 수식(선형 감쇠)으로 변형을 계산**해 표시하며
(상단 배너에 `DUMMY 모드` 명시), 모델이 연결되면 자동으로 **BE가 돌려준 메쉬**를
표시한다(`LIVE` 배너).

## 실행 (Windows)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

> Linux/macOS도 동일하게 동작한다 (`source .venv/bin/activate`).
> BE 주소는 환경변수 `PHYSICS_BE_URL`로 바꾼다 (기본 `http://127.0.0.1:9003`).

```bat
set PHYSICS_BE_URL=http://192.168.0.42:9003
python app.py
```

## 사용법

1. 시작 시 **내장 더미 메쉬**가 떠 있다(중앙 노드가 기본 충격점). 또는 **Load mesh…**로
   `.vtk` / `.obj` / `.stl` / `.ply` / `.off` 파일을 연다.
2. 뷰포트에서 노드를 **클릭** → `Impact node`가 설정되고 주황 구로 표시.
3. **Force X/Y/Z**(충격 방향·세기), 선택적으로 **Radius**(0=자동) / **Scale** 입력.
4. **Simulate** → 변형 결과(빨강 solid)가 원본(파랑 wireframe) 위에 표시.
5. **Reset** → 원본만 다시 표시.

### 동작 모드 배너 (상태 점 아래)

| 배너 | 조건 | Simulate 동작 |
|---|---|---|
| 🟢 `LIVE · BE 연결됨` | `/health` ready 모델 존재 | BE `/predict` 결과 메쉬 표시 |
| 🟣 `BE 로딩 중` | 모델 LOADING | 로컬 더미 반응 (임시) |
| 🟠 `DUMMY 모드` | 서버 미연결/모델 미준비 | **로컬** 선형 감쇠 변형 (로그에 `(로컬 더미 반응)`) |

로컬 더미 반응은 BE `models/dummy/DummyLinearDeformer`와 **동일한 수식**이라, 모델
연결 전후로 같은 dummy 거동을 보인다. 상태 점/모델 드롭다운은 `/health` 기준이며
`Refresh` 또는 매 Simulate 직전에 갱신된다.

## 노드 인덱스 일관성

FE도 BE(`utils/mesh_handler.py`)와 동일하게 **meshio**로 메쉬를 파싱해
vertices/faces를 얻고 그 순서대로 렌더한다. 클릭 지점은 그 vertices 배열의
최근접 인덱스로 환산되고, Simulate 시 **원본 파일 바이트를 그대로** 전송하므로
BE가 같은 순서로 재파싱 → `impact_node`가 정확히 일치한다.

## 버전

버전은 `physics/fe/VERSION`(예: `0.1.0`)에 있고 창 제목/패키징 exe에 표시된다.
GitHub Actions가 **빌드마다 patch를 자동으로 +1** 한다(아래 CI 참고).

## Windows 단일 실행파일 패키징 (선택)

```bat
pip install pyinstaller
pyinstaller --noconfirm --windowed --name PhysicsSimulator --add-data "VERSION;." app.py
# dist\PhysicsSimulator\PhysicsSimulator.exe
```
PyVista/VTK 데이터가 누락되면 `--collect-all pyvista --collect-all pyvistaqt
--collect-all vtkmodules --collect-all meshio`를 추가한다.

## CI (GitHub Actions)

`.github/workflows/fe-windows-build.yml` — `physics/fe/**` push(또는 수동 실행) 시
`windows-latest`에서 자동으로:
1. `VERSION` patch +1 후 `[skip ci]` 커밋 push (버전 자동 증가, 루프 방지),
2. PyInstaller로 `.exe` 빌드 → zip,
3. **GitHub Release `fe-vX.Y.Z`**(zip 첨부) + Actions artifact 업로드.

> CI가 커밋/릴리스를 만들려면 repo Settings → Actions → "Read and write permissions"가
> 필요하다. 보호 브랜치라면 github-actions[bot]의 push를 허용해야 한다.
