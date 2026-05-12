# Sound Localization Demo (Galaxy S25 + BE)

폰 내장 stereo 마이크에서 raw PCM을 받아 GCC-PHAT 기반 TDOA로 1D 방위각(azimuth)을 추정하는 데모.

```
[Galaxy S25 App]
   │  POST http://<PC_LAN_IP>:9001/localize  (audio/wav, stereo, 48kHz, 2초)
   ▼
[BE : FastAPI (포트 9001), torchaudio + tdoa.py 유틸]
   │  processor.process_stereo(waveform, sr) → JSON
```

폰과 BE가 같은 LAN에서 직접 통신.

## 디렉터리

```
localization/
├── android/                # Kotlin 앱 (Android Studio 프로젝트)
└── be/
    ├── main.py             # FastAPI app, lifespan(LLM 로드), router 등록
    ├── llm.py              # LLM load_model / infer — placeholder, 사용자 구현
    ├── routers/
    │   ├── localize.py     # POST /localize
    │   └── inference.py    # POST /inference
    ├── processor.py        # process_stereo placeholder (사용자 구현)
    └── tdoa.py             # GCC-PHAT / azimuth 유틸 (완성)
```

## BE 구성

- **Lifespan**: `main.py`의 `lifespan` 컨텍스트에서 startup 시 `llm.load_model()`을 1회 호출 → `app.state.llm_model`에 저장. 셧다운 시 정리.
- **`/localize`** (`routers/localize.py`): raw WAV 바이트 수신 → `torchaudio.load_with_torchcodec` → `process_stereo(waveform, sr)` 호출.
- **`/inference`** (`routers/inference.py`): JSON body 수신 → `app.state.llm_model`와 함께 `llm.infer(model, payload)` 호출. 모델 미로드 시 503.
- **`/health`** (`main.py`): `{"status": "ok", "model_loaded": bool}`
- **사용자 swap-in 지점**:
  - `be/llm.py` — `load_model()`, `infer(model, payload)` 본문 작성
  - `be/processor.py` — `process_stereo()` 본문 작성 (tdoa.py 함수 활용 예시는 아래 가이드 참고)

## 실행

### 1. BE 기동
```bash
cd localization/be
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 9001
```

`--host 0.0.0.0` 필수 — 폰이 LAN에서 접근하려면 외부 인터페이스 바인딩 필요.

### 2. 안드로이드 앱 빌드 (Android Studio)

1. **Android Studio 설치**
   - https://developer.android.com/studio → 다운로드 (JDK 17 번들 포함)
   - Linux: `tar -xzf android-studio-*.tar.gz && bin/studio.sh`

2. **프로젝트 열기**
   - 첫 화면 → **Open** → `SLLM/localization/android/` 선택
   - "Trust Project" 클릭
   - Gradle Sync 자동 시작 (수 분, 의존성 다운로드)
   - 누락 SDK 항목 있으면 Studio가 설치 프롬프트 표시 — 동의

3. **S25 연결**
   - S25 설정 → 휴대전화 정보 → 소프트웨어 정보 → 빌드 번호 **7회 탭** → 개발자 옵션 활성화
   - 개발자 옵션 → **USB 디버깅 ON**
   - USB로 PC 연결, 폰에 뜨는 "USB 디버깅 허용" 수락
   - Studio 상단 디바이스 선택기에 S25 표시 확인

4. **빌드 & 실행**

   기기 연결 여부와 무관하게 **빌드 자체는 기기 없이 가능**. 기기/에뮬레이터는 "설치/실행" 단계에서만 필요.

   **CLI 빌드 사전 준비 — `JAVA_HOME` 설정**

   시스템에 별도 JDK가 없으면 Android Studio 번들 JBR(JDK 21)을 사용:
   ```bash
   export JAVA_HOME=/home/<user>/Downloads/android-studio/jbr
   export PATH="$JAVA_HOME/bin:$PATH"
   ```
   (Studio 설치 경로에 맞게 수정. 영구 적용은 `~/.bashrc` 또는 `~/.zshrc`에 추가)

   Android Studio GUI로 빌드하면 JAVA_HOME 신경 쓸 필요 없음 (Studio가 내부에서 처리).

   - **기기 없이 APK만 생성** (사전 빌드)
     ```bash
     cd localization/android
     ./gradlew assembleDebug
     # → app/build/outputs/apk/debug/Localization-v<version>-debug.apk
     ```
     Android Studio 메뉴: `Build → Generate App Bundles or APKs → Generate APKs` 도 동일 (debug APK, 서명 없음). 빌드 완료 후 우하단 알림의 **locate** 링크로 결과 파일 즉시 이동 가능.
     > Studio 버전이 옛것이라면 `Build → Build Bundle(s) / APK(s) → Build APK(s)`로 표시될 수 있음 — 동일 기능.
     >
     > `Generate Bundles`(.aab)는 Play Store 배포용이므로 데모/사이드로딩엔 부적합. APK만 사용.

   - **기기 연결 후 설치 + 실행**
     - 툴바 ▶ Run 클릭 → debug APK 빌드 + 설치 + 실행
     - 또는 CLI: `./gradlew installDebug`
     - 사전 빌드한 APK 수동 설치: `adb install -r app/build/outputs/apk/debug/Localization-v<version>-debug.apk`

   > 참고 1: 이 데모는 폰의 실제 stereo 마이크가 필수이므로 **에뮬레이터로는 검증 불가** (가상 마이크는 mono 합성). 실 S25에서만 의미가 있음.
   >
   > 참고 2: 첫 `assembleDebug`는 의존성 다운로드로 수 분 걸림. 이후엔 캐시 사용.

5. **첫 실행**
   - 마이크 권한 팝업 → 허용
   - **Server URL** 입력란에 `http://<PC_LAN_IP>:9001` (기본값은 `192.168.0.42`, 본인 IP로 수정)
   - **Start** 탭

## 검증

1. **BE 헬스 체크**
   ```bash
   curl http://localhost:9001/health
   # → {"status":"ok","model_loaded":true}
   ```
   `model_loaded`가 `false`면 `llm.load_model()`이 None을 반환했거나 startup 시 실패한 것. uvicorn 로그 확인.
2. **LAN 노출 확인**: 폰 브라우저에서 `http://<PC_LAN_IP>:9001/health` GET 응답 확인.
   안 되면 방화벽: `sudo ufw allow 9001/tcp`
3. **앱 end-to-end**
   - Start 후 약 2초 뒤 응답 로그에 placeholder JSON 추가 (`num_channels=2`, `num_samples≈96000`)
   - 앱 로그 첫 줄에 실제 사용 sample rate 표시 (`녹음 시작: 48000Hz stereo, 2s chunks → ...`)
   - `azimuth` 표시는 사용자가 `processor.py`를 구현한 뒤 채워짐

## TDOA 유틸 단위 검증
```python
import numpy as np
from tdoa import gcc_phat, tau_to_azimuth

fs = 48000
t = np.arange(fs) / fs
sig = np.sin(2*np.pi*1000*t)
left = sig
right = np.roll(sig, 10)        # right가 10샘플 지연

tau, _ = gcc_phat(left, right, fs)
print(tau * fs)                  # ≈ 10
print(tau_to_azimuth(tau, 0.14)) # 양의 각도 (왼쪽이 먼저 도달)
```

## processor.py 작성 가이드 (사용자 후속 작업)

`be/processor.py`의 `process_stereo` 본문을 다음과 같이 채우면 실제 azimuth가 응답에 포함:

```python
import numpy as np
import torch
from tdoa import gcc_phat, tau_to_azimuth, confidence_from_cc, DEFAULT_MIC_DISTANCE_M

def process_stereo(waveform: torch.Tensor, sample_rate: int) -> dict:
    arr = waveform.numpy()             # [2, N]
    left, right = arr[0], arr[1]
    tau, cc = gcc_phat(left, right, sample_rate, max_tau=DEFAULT_MIC_DISTANCE_M / 343.0 * 2)
    azimuth = tau_to_azimuth(tau, DEFAULT_MIC_DISTANCE_M)
    return {
        "sample_rate": sample_rate,
        "num_samples": int(arr.shape[-1]),
        "tdoa_ms": round(tau * 1000, 4),
        "azimuth_degrees": round(azimuth, 2),
        "confidence": round(confidence_from_cc(cc), 3),
    }
```

## 알려진 한계

- 2-mic 직선 어레이 → **front/back 모호성** (cone of confusion): 좌-우는 명확, 앞/뒤 구분 불가
- 마이크 간격 `0.14m`는 S25 추정치. 실측 후 `tdoa.py:DEFAULT_MIC_DISTANCE_M` 보정 필요
- `AudioSource.CAMCORDER`가 stereo로 노출되는지는 OEM 의존. S25에서는 stereo 확인됨 (`num_channels=2`로 검증)
