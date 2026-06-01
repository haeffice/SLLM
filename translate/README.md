# 실시간 음성 번역 (Windows 클라이언트 + 서버)

실시간 번역용 Speech LLM을 로드하는 **HTTPS FastAPI 서버**와, 시스템/마이크
오디오를 16kHz mono PCM16으로 변환해 20ms 단위 바이너리 프레임으로 **WebSocket**
스트리밍하고 번역 결과(confirmed/prediction)를 화면 하단 오버레이로 표시하는
**Windows 데스크톱 앱(Electron)** 으로 구성된다.

```
translate/
├── be/                         # 번역 서버 (FastAPI + uvicorn, HTTPS/WSS)
│   ├── main.py                 #   /health(HTTP) + /ws(WebSocket), lifespan 비동기 로드
│   ├── translator.py           #   모델 레지스트리 + 장치/체크포인트 해석
│   ├── models/
│   │   ├── base.py             #   Translator 추상 인터페이스 (stream_step)
│   │   └── mock.py             #   MockTranslator (실모델 교체 지점)
│   ├── routers/ws.py           #   WS /ws (로딩/실패 가드 포함)
│   ├── preprocess.py           #   PCM16 → float32 (pcm16_to_float)
│   ├── gen-cert.sh             #   자체서명 인증서 생성
│   ├── run.sh                  #   체크포인트 인자 + TLS uvicorn 기동
│   └── requirements.txt
└── app/                        # Windows 클라이언트 (Electron)
    ├── package.json            #   start / build:win 스크립트
    ├── electron-builder.yml    #   NSIS (Windows x64) 빌드 설정
    ├── scripts/gen-config.js   #   빌드 시 서버 URL 주입
    └── src/{main,preload,renderer}/
```

---

## 1. 서버 실행

### 1) 자체서명 인증서 생성 (1회)

PC의 LAN IP 확인 후(`hostname -I`), 해당 IP로 인증서를 생성한다. 이하 예시는
LAN IP를 `192.168.0.42`로 가정.

```bash
cd translate/be
bash gen-cert.sh 192.168.0.42
```

→ `translate/be/cert.pem`, `key.pem` 생성. SAN에 `localhost`, `127.0.0.1`,
`192.168.0.42` 포함. (LAN IP가 바뀌면 인증서 재생성 필요.)

### 2) 의존성 설치

```bash
cd translate/be
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3) 서버 기동 (모델 체크포인트를 인자로)

```bash
# 기본 모델(mock)에 체크포인트 경로를 인자로 전달 → MOCK_CKPT 로 export
./run.sh /path/to/checkpoint

# 또는 환경변수로 직접 지정
MOCK_CKPT=/path/to/checkpoint ./run.sh
```

- 기본 바인딩: `https://0.0.0.0:9001` (`BE_HOST`, `BE_PORT`로 변경 가능)
- `AUDIO_LLM_ENABLED` / `AUDIO_LLM_DEFAULT` 로 모델 등록/기본값 지정 (기본: `mock`)
- 모델은 백그라운드에서 비동기 로드되며, 로드 상태는 `/health`로 노출

### 4) 동작 확인

```bash
# 상태 + 모델 태그 (HTTP)
curl -k https://localhost:9001/health
```

(`-k`는 자체서명 인증서 검증 우회)

번역은 WebSocket(`wss://<host>:9001/ws?src=en&tgt=ko&task=translate`)으로
동작한다. 오디오는 **바이너리 프레임**으로 보내므로 `websocat`로는 컨트롤 메시지만
확인 가능하다.

### WebSocket 프로토콜

같은 소켓 위에 두 종류의 **Client → Server** 프레임이 흐른다.

- **바이너리 프레임**: raw little-endian **PCM16, mono, 16kHz** 오디오.
  - 클라이언트는 20ms(=640바이트)마다 한 프레임씩 전송(저지연). 서버가 누적해
    언제 결과를 낼지 결정.
- **텍스트 프레임(JSON 컨트롤)**: 재연결 없이 세션을 즉시 변경.
  - `{"type":"directionchange","direction":"en2ko"|"ko2en"}` — 번역 방향 전환.
  - `{"type":"taskchange","task":"translate"|"transcribe"}` — 번역/전사 전환.
  - `{"type":"micoff"}` — 마이크 끔, 서버 디코딩 버퍼 리셋.
  - `{"type":"optionschanged", ...AIOptions}` — 파라미터(AIOptions)를
    평탄한 형태로 전달. 소켓 연결 직후와 다이얼로그 저장 시 전송.
  - (방향·task 전환과 micoff는 모두 누적 디코딩 컨텍스트를 비운다.)

**Server → Client**: `{"confirmed": "<검정 표시>", "prediction": "<회색 표시>"}`
  - `confirmed`는 확정 번역(누적), `prediction`은 현재 잠정 결과.
  - 가드 실패 시 `{"error": "..."}` 전송 후 소켓 종료.

---

## 2. Windows 앱 빌드

> **Windows에서 빌드하는 것을 권장**한다 (시스템 오디오 루프백·`<webview>`는
> Windows에서만 정상 동작/검증 가능). 사전에 **Node.js LTS**(npm 포함)만 설치하면
> 되고, Wine·docker 등은 불필요하다.
>
> 클라이언트는 빌드 시 서버 URL을 **고정**한다. 소스 기본값은 비어 있으며,
> `SLLM_SERVER_URL` 환경변수로 빌드 시점에 주입된다. **셸마다 환경변수 설정
> 문법이 다르니 주의** (cmd에서 bash 문법을 쓰면 값이 안 잡혀 서버 URL이 빈 채로
> 빌드된다).

### 설치 파일(.exe) 빌드 — Windows cmd

```cmd
cd translate\app
npm install
set SLLM_SERVER_URL=https://192.168.0.42:9001
npm run build:win
```

- `set` 뒤 값에 **따옴표 금지**, `=` 양옆 **공백 금지** (cmd는 그대로 값에 포함).
- 결과물: `translate\app\dist\`에 NSIS 설치 파일
  (`SLLM Translate Setup 0.1.0.exe`).

### 설치 파일(.exe) 빌드 — PowerShell

```powershell
cd translate\app
npm install
$env:SLLM_SERVER_URL = "https://192.168.0.42:9001"
npm run build:win
```

### 개발 실행 (로컬 확인)

```cmd
cd translate\app
npm install
set SLLM_SERVER_URL=https://127.0.0.1:9001
npm start
```

> **참고**
> - 코드 서명이 없는 설치 파일은 Windows SmartScreen 경고가 표시될 수 있다.
> - (선택) Linux/CI에서 빌드해야 한다면 Wine이 필요하며, electron-builder의
>   docker 이미지(`electronuserland/builder:wine`) 또는 GitHub Actions
>   `windows-latest` 러너를 사용한다. bash 셸에서는
>   `SLLM_SERVER_URL=https://... npm run build:win` 형태로 주입한다.

---

## 3. 동작 흐름

1. 좌측 사이드바에서 **실시간 번역** 탭 → 검색 바 + 임베디드 브라우저 화면으로 전환.
   (다시 탭하면 화면 초기화)
2. 검색 바 입력 → `google.com/search`를 `<webview>`에 로드하여 브라우징.
3. **마이크 버튼** → 마이크/시스템 오디오를 `AudioContext`(16kHz)로 캡처 →
   `AudioWorklet`이 PCM16으로 변환 → 20ms 바이너리 프레임으로 **WebSocket** 전송.
   소켓은 첫 마이크 켜기에서 한 번 열려, 이후 마이크 토글·방향/task 전환에도
   재연결 없이 유지된다(컨트롤 메시지로 세션만 갱신).
4. 서버가 보내는 `confirmed`/`prediction`을 화면 하단 **자막 패널**에 표시.
   `confirmed`는 **검정**, `prediction`은 **회색**(bold). 패널은 검은 테두리 +
   `#e9ecef` 배경의 박스이며, **상단 손잡이를 드래그해 높이를 조절**(설정에 저장)
   할 수 있다. 최근 줄은 진하게, 이전 줄은 위로 갈수록 옅어지며(스크롤바 없이)
   **스크롤로 이전 자막**을 볼 수 있다. 배경 투명도는 옵션에서 조절.

---

## 4. 옵션

옵션 다이얼로그에서 조절 가능:

| 항목          | 기본값  | 설명                                        |
| ------------- | ------- | ------------------------------------------- |
| 서버 URL      | (빌드값)| 런타임 오버라이드 (빌드 시 주입값이 기본)   |
| 오디오 소스   | 마이크  | 마이크 / 시스템 음성                        |
| 투명도        | 0       | 0=가림, 100=글씨만 (`alpha = 1 - pct/100`)  |
| 표시 너비     | 720px   | 자막 패널 최대 너비                         |
| 최근 줄 수    | 3       | 옅어지기 전 진하게 유지할 최근 줄 수        |

자막 패널 **높이**(`bandHeightPx`, 기본 384px)는 옵션이 아니라 패널 상단 손잡이를
드래그해 조절하며 설정에 저장된다.

**파라미터** 버튼은 AIOptions(waitK·kvCache·mode·청크 길이·preview 관련
파라미터 등)를 편집한다(창은 스크롤되며 스크롤바는 숨김). 저장하면
`settings.hyperparameters`에 보관되고 `{"type":"optionschanged", ...}`로 서버에
전송된다(연결 직후에도 1회 전송).

설정은 `userData/settings.json`에 저장된다. 언어 방향(`EN→KO`/`KO→EN`)과
task(`번역`/`전사`)는 상단 **토글 스위치**로 전환하며(왼쪽/오른쪽으로 현재 값 표시),
캡처 중에도 재연결 없이 즉시 반영된다. **옵션**은 톱니바퀴(⚙) 아이콘 버튼이다.

---

## 5. 실모델 교체

`translate/be/models/mock.py`의
`MockTranslator.stream_step(pcm, src, tgt, task, state) -> dict | None` 본문만 실제
스트리밍 Speech LLM 추론으로 교체하면 된다. `pcm`은 raw PCM16 mono 16kHz,
`task`는 `"translate"`/`"transcribe"`, `state`는 WebSocket 연결당 유지되는
dict(누적 오디오/디코딩 컨텍스트 보관용; 방향/task 전환·micoff 시 초기화).
반환 dict의 `"confirmed"`/`"prediction"` 필드와 시그니처는 고정(클라이언트가
검정/회색으로 렌더). 별도 모델을 추가하려면 `translate/be/translator.py`의
`REGISTRY`에 `Translator` 서브클래스를 등록한다.

---

## 6. DRM (Widevine) 지원

임베디드 `<webview>`에서 DRM/EME 보호 콘텐츠(예: 보호된 스트리밍)를 재생하려면
**Widevine CDM**이 필요하다. 메인 프로세스(`src/main/main.js`)는 앱 시작 시
`components`(Widevine) 초기화를 시도하며, 이 API는 **Widevine 지원 Electron
빌드에서만** 존재한다. 일반 Electron에서는 자동으로 건너뛰어(no-op) DRM 없이 그대로
실행된다(로그: `Widevine: components API unavailable`).

실제 DRM 재생을 활성화하려면:

1. Electron 의존성을 castlabs의 Widevine 포함 빌드로 교체(현재 메이저 v31에 맞는
   태그 사용). 예:

   ```bash
   npm install --save-dev "github:castlabs/electron-releases#v31.<patch>+wvcus"
   ```

2. `npm start` / `npm run build:win`은 그대로 사용. 시작 로그에 `Widevine: ready`가
   보이면 CDM 활성화 완료.
3. **배포(설치 파일) 시에는 castlabs EVS 계정으로 VMP 서명**이 추가로 필요하다
   (개발 실행에는 불필요). 자세한 절차는 castlabs/electron-releases 문서 참고.

> 코드(`initWidevine`)는 이미 DRM-ready 상태이므로, 위 1번처럼 의존성만 교체하면
> 동작한다. 기본 의존성은 CI 빌드 호환을 위해 일반 Electron으로 유지한다.

---

## 7. 트러블슈팅

- **인증서 경고/연결 실패**: LAN IP로 `gen-cert.sh`를 다시 실행했는지 확인. 앱은
  설정된 서버 호스트에 한해서만 자체서명 인증서를 신뢰한다(WSS 포함).
- **WS가 바로 끊김 + `{"error":...}`**: 모델이 아직 로딩 중이거나 로드 실패.
  `/health`의 `models.<id>.status`로 확인(`ready`가 되면 재연결).
- **시스템 오디오가 캡처되지 않음**: 루프백 캡처는 Windows 전용. 최신 Electron
  필요.
- **포트 접속 불가**: PC 방화벽이 9001 inbound 차단 가능. (`sudo ufw allow 9001/tcp`)
