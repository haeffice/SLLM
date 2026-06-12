# Auto Approver (Chrome Extension)

5분마다 `api_url_1`을 GET 폴링해 응답의 `items[].id`를 추출하고,
각 id에 대해 `api_url_2`로 승인 POST(`{id, groupId, approvedBy}`)를 보내는
Manifest V3 확장 프로그램입니다.

대상 서버가 CORS를 사용하므로 모든 요청은 background service worker에서 수행하며
(`host_permissions`로 CORS 우회), Bearer 토큰은 **사용자가 열어둔 대상 페이지가
자체적으로 보내는 요청에서 자동 캡처**합니다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `manifest.json` | MV3 매니페스트 (권한, host_permissions) |
| `config.js` | **모든 `###` 기입 값이 모여 있는 설정 파일** |
| `background.js` | service worker — 토큰 캡처 + 5분 폴링 + 승인 POST |

## 빌드 / 설치 방법

별도 빌드 단계는 없습니다(plain JS). 아래 순서로 로드합니다.

1. 아래의 [`###` 기입 체크리스트](#-기입-체크리스트)를 모두 채웁니다.
2. Chrome에서 `chrome://extensions` 접속
3. 우측 상단 **개발자 모드** 토글 ON
4. **압축해제된 확장 프로그램을 로드합니다(Load unpacked)** 클릭 → 이 폴더 선택
5. 코드를 수정한 경우에는 확장 카드의 새로고침(↻) 버튼을 눌러 재로드

## `###` 기입 체크리스트

### `manifest.json`

| 위치 | 내용 |
|---|---|
| `host_permissions[0]` | 대상 페이지(토큰을 캡처할 사이트)의 origin |
| `host_permissions[1]` | api_url_1 의 origin |
| `host_permissions[2]` | api_url_2 의 origin |

origin이 서로 같으면 하나로 합쳐도 됩니다. 예: `"https://api.example.com/*"`

### `config.js`

| 상수 | 내용 |
|---|---|
| `API_URL_1` | 5분마다 GET 폴링할 URL |
| `API_URL_2` | 승인 POST를 보낼 URL |
| `CAPTURE_URL_PATTERNS` | 대상 페이지가 authorization 헤더를 붙여 호출하는 API의 URL 패턴 |
| `GROUP_ID` | POST body의 `groupId` |
| `APPROVED_BY` | POST body의 `approvedBy` |
| `X_REQUEST_GROUP_ID_FALLBACK` | `x-request-group-id` 헤더 fallback (캡처되면 캡처값 우선) |
| `ACCEPT`, `ACCEPT_LANGUAGE` | 요청 헤더 (DevTools "Copy as fetch" 값 그대로) |
| `REFERRER` | 대상 페이지 URL |

> **주의:** POST body의 `groupId`와 헤더의 `x-request-group-id`는
> 서로 **다른 값일 수 있습니다.** 각각 확인해서 기입하세요.

## 동작 방식 / 동작 조건

1. **토큰 캡처:** 대상 페이지를 열면, 페이지가 보내는 API 요청의
   `authorization`(Bearer) / `x-request-group-id` 헤더를 webRequest로 읽어
   `chrome.storage.session`(메모리 전용)에 저장합니다.
   - **확장 설치(또는 브라우저 재시작) 후 대상 페이지를 최소 한 번 열어야 동작이 시작됩니다.**
   - 브라우저를 종료하면 storage.session이 비워지므로 페이지를 다시 방문해야 합니다.
   - 토큰이 갱신(rotate)되면 페이지의 다음 요청에서 자동으로 새 값을 덮어씁니다.
2. **폴링:** `chrome.alarms`가 5분마다 service worker를 깨워 GET → POST를 수행합니다.
   - 토큰 미캡처 / 요청 오류 / 응답 오류 / `items`가 비어 있음 → 해당 주기는 skip하고 5분 대기
   - `items`에 객체가 여러 개면 **모든 id에 대해 각각 POST** (중복 방지 없이 매 주기 전송)

## 제약 사항

- `sec-ch-ua`, `sec-ch-ua-mobile`, `sec-ch-ua-platform`, `sec-fetch-dest`,
  `sec-fetch-mode`, `sec-fetch-site` 헤더는 **누락이 아니라 설정이 불가능한 것입니다.**
  `Sec-` 접두사는 Fetch 스펙상 forbidden header라 확장 프로그램(JS)에서 지정해도 무시되며,
  브라우저가 자체 값을 자동으로 부착합니다. 서버가 이 값들을 엄격하게 검증하면
  요청이 거부될 수 있습니다.
- `referrer`는 service worker에 document가 없어 best-effort로만 적용됩니다
  (Chrome이 확장 origin으로 재작성하거나 제거할 수 있음).
- 즉, DevTools "Copy as fetch"와 byte 단위로 동일한 요청 복제는 불가능하며,
  일반적으로는 `authorization` 헤더만 유효하면 정상 동작합니다.

## 디버깅

1. `chrome://extensions` → 이 확장 카드의 **"service worker"** 링크 클릭 → 콘솔 열림
2. 콘솔에서 상태 확인:
   - `chrome.alarms.getAll(console.log)` — 5분 알람 등록 확인
   - `chrome.storage.session.get(console.log)` — 캡처된 토큰 확인
3. 폴링/승인 로그는 `[poll]`, `[approve]` 접두사로 출력됩니다.
4. 즉시 테스트하려면 `config.js`의 `POLL_MINUTES`를 임시로 `0.5`(30초)로 낮춘 뒤
   확장을 재로드하세요.
