# Audio Demo

브라우저 마이크 입력을 10초 단위 WAV(16kHz mono PCM)로 잘라
FastAPI 서버로 전송하고 처리 결과를 표시하는 최소 데모.
uvicorn 하나가 FE 정적 파일과 `/infer` API를 같은 HTTPS origin으로 서빙.

## 구조

```
demo/
├── fe/
│   ├── index.html
│   └── app.js
└── be/
    ├── main.py            # FastAPI: /health, /infer + FE static mount
    ├── inference.py       # mock 추론 (LLM 교체 지점)
    ├── gen-cert.sh        # 자체서명 인증서 생성 스크립트
    └── requirements.txt
```

## 사전 준비

PC의 LAN IP 확인:
```bash
hostname -I        # 또는: ip -4 addr show
```
이하 예시는 LAN IP를 `192.168.0.42`로 가정.

## 실행

### 1. 자체서명 인증서 생성 (1회)
```bash
cd demo/be
bash gen-cert.sh 192.168.0.42
```
→ `demo/be/cert.pem`, `demo/be/key.pem` 생성. SAN에 `localhost`, `127.0.0.1`, `192.168.0.42` 포함.

### 2. 의존성 설치
```bash
cd demo/be
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. TLS uvicorn 기동
```bash
uvicorn main:app --host 0.0.0.0 --port 8001 \
  --ssl-keyfile key.pem --ssl-certfile cert.pem --reload
```

### 4. 접속
- **PC**: 브라우저에서 `https://localhost:8001/`
- **스마트폰 (동일 Wi-Fi)**: `https://192.168.0.42:8001/`
  → 자체서명이라 "안전하지 않음" 경고 표시 → **고급 → 이동(unsafe)** 클릭 → FE 페이지 로드.

Start 버튼 → 마이크 권한 허용 → 약 10초마다 응답 로그에 JSON 추가.

## 동작 흐름

1. FE(`https://...:8001/`): `getUserMedia` → `AudioContext` → `ScriptProcessorNode`로 Float32 캡처
2. FE: 10초 누적 → 16kHz 리샘플 → PCM16 WAV 인코딩 → `POST /infer` (동일 origin 상대 경로)
3. BE: raw body에서 WAV 수신 → `torchaudio.load` → `inference.infer(waveform, sample_rate)` → JSON 반환

## LLM 교체

`demo/be/inference.py`의 `infer(waveform, sample_rate) -> dict` 본문만 교체.
시그니처는 고정.

## BE 단독 검증
```bash
curl -k -X POST --data-binary @sample.wav \
  -H "Content-Type: audio/wav" \
  https://localhost:8001/infer
```
(`-k`로 자체서명 검증 우회)

## 트러블슈팅

- **폰에서 접속 불가**: PC 방화벽이 8001 inbound 차단했을 가능성. `sudo ufw allow 8001/tcp` (ufw 사용 시).
- **`navigator.mediaDevices` undefined 재발**: `http://`로 접속했거나 SAN에 IP가 빠진 경우. `gen-cert.sh`를 정확한 LAN IP로 재실행.
- **인증서 경고 무한 반복**: 다른 IP로 cert를 만들었을 가능성. LAN IP 변경 시 인증서 재생성 필요.
- **`openssl: -addext` 미인식**: openssl 1.1.1 이상 필요.

## 제약

- 자체서명이라 폰에서 매 세션 경고 수동 수락 필요 (mkcert + 폰 CA 신뢰는 비범위)
- Stop 시 10초 미만 잔여 버퍼는 폐기
- `ScriptProcessorNode`는 deprecated이지만 단일 파일 데모를 위해 사용
