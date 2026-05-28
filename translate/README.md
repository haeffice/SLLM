# 실시간 음성 번역 (Windows 클라이언트 + 서버)

실시간 번역용 Speech LLM을 로드하는 **HTTPS FastAPI 서버**와, 시스템/마이크
오디오를 720ms 단위 WAV로 잘라 서버로 전송하고 번역 결과를 화면 하단 오버레이로
표시하는 **Windows 데스크톱 앱(Electron)** 으로 구성된다.

```
translate/
├── be/                         # 번역 서버 (FastAPI + uvicorn, HTTPS)
│   ├── main.py                 #   /health, /translate, lifespan 비동기 모델 로드
│   ├── translator.py           #   모델 레지스트리 + 장치/체크포인트 해석
│   ├── models/
│   │   ├── base.py             #   Translator 추상 인터페이스
│   │   └── mock.py             #   MockTranslator (실모델 교체 지점)
│   ├── routers/translate.py    #   POST /translate (로딩/실패 가드 포함)
│   ├── preprocess.py           #   WAV → mono 16kHz 텐서
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
# 상태 + 모델 태그
curl -k https://localhost:9001/health

# 번역 (raw WAV body)
curl -k -X POST --data-binary @sample.wav \
  -H "Content-Type: audio/wav" \
  "https://localhost:9001/translate?src=en&tgt=ko"
```

(`-k`는 자체서명 인증서 검증 우회)

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
3. **마이크 버튼** → 마이크 또는 시스템 오디오를 `AudioContext`로 캡처 →
   16kHz 리샘플 → 720ms WAV로 인코딩 → `POST /translate`.
4. 서버가 반환한 `text`를 화면 하단 **오버레이 밴드**(bold, black)에 표시.
   배경 투명도는 옵션에서 조절(0% = 화면 가림, 100% = 글씨만 표시).

---

## 4. 옵션

옵션 다이얼로그에서 조절 가능:

| 항목          | 기본값  | 설명                                        |
| ------------- | ------- | ------------------------------------------- |
| 서버 URL      | (빌드값)| 런타임 오버라이드 (빌드 시 주입값이 기본)   |
| 오디오 소스   | 마이크  | 마이크 / 시스템 음성                        |
| 샘플레이트    | 16000   | 캡처 샘플레이트                             |
| 투명도        | 0       | 0=가림, 100=글씨만 (`alpha = 1 - pct/100`)  |
| 표시 너비     | 720px   | 오버레이 텍스트 최대 너비                   |
| 최근 줄 수    | 3       | 표시할 최근 (시각적) 줄 수                  |

**하이퍼파라미터** 버튼은 자리만 마련되어 있으며 추후 사용 예정.

설정은 `userData/settings.json`에 저장된다. 언어 방향(영↔한)은 상단 토글 버튼으로 전환.

---

## 5. 실모델 교체

`translate/be/models/mock.py`의 `MockTranslator.translate(wav_bytes, src, tgt) -> dict`
본문만 실제 Speech LLM 추론으로 교체하면 된다. 반환 dict의 `"text"` 필드와
메서드 시그니처는 고정 (클라이언트가 의존). 별도 모델을 추가하려면
`translate/be/translator.py`의 `REGISTRY`에 `Translator` 서브클래스를 등록한다.

---

## 6. 트러블슈팅

- **인증서 경고/연결 실패**: LAN IP로 `gen-cert.sh`를 다시 실행했는지 확인. 앱은
  설정된 서버 호스트에 한해서만 자체서명 인증서를 신뢰한다.
- **`/translate`가 503**: 모델이 아직 로딩 중(`Retry-After: 30`)이거나 로드 실패.
  `/health`의 `models.<id>.status`로 확인.
- **시스템 오디오가 캡처되지 않음**: 루프백 캡처는 Windows 전용. 최신 Electron
  필요.
- **포트 접속 불가**: PC 방화벽이 9001 inbound 차단 가능. (`sudo ufw allow 9001/tcp`)
