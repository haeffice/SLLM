# Audio Demo

브라우저 마이크 입력을 10초 단위 WAV(16kHz mono PCM)로 잘라
FastAPI 서버로 전송하고 처리 결과를 표시하는 최소 데모.

## 구조

```
demo/
├── fe/                # 정적 페이지 (python http.server로 서빙)
│   ├── index.html
│   └── app.js
└── be/                # FastAPI 서버
    ├── main.py
    ├── inference.py   # mock 추론 (LLM 교체 지점)
    └── requirements.txt
```

## 실행

### BE (포트 8001)

```bash
cd demo/be
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### FE (포트 8000)

```bash
cd demo/fe
python -m http.server 8000
```

브라우저에서 `http://localhost:8000` 접속 → **Start** 클릭 → 마이크 권한 허용.
약 10초마다 응답이 로그 영역에 추가된다.

## 동작 흐름

1. FE: `getUserMedia` → `AudioContext`(16kHz 요청) → `ScriptProcessorNode`로 Float32 캡처
2. FE: 누적 샘플이 10초 분량에 도달하면 → 16kHz로 (필요 시) 리샘플링 → WAV 인코딩 → `POST http://localhost:8001/infer`
3. BE: raw body에서 WAV 바이트 수신 → `torchaudio.load` → `inference.infer(waveform, sample_rate)` 호출 → JSON 반환

## LLM 교체

`demo/be/inference.py`의 `infer(waveform, sample_rate) -> dict` 함수만 교체.
시그니처는 고정 — 입력은 `torch.Tensor`, 반환은 dict.

## BE 단독 검증

```bash
curl -X POST --data-binary @sample.wav \
  -H "Content-Type: audio/wav" \
  http://localhost:8001/infer
```

## 제약

- HTTPS가 아닌 `http://localhost`에서만 마이크 접근 가능 (브라우저 보안 정책)
- Stop 시 10초 미만 잔여 버퍼는 폐기됨
- `ScriptProcessorNode`는 deprecated이지만 단일 파일 데모를 위해 사용
