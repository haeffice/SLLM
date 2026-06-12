// 추후 기입할 값은 모두 이 파일에 모여 있으며 ### 로 표시되어 있습니다.
// (manifest.json 의 host_permissions 에도 ### 가 있으니 함께 수정하세요.)

// ### api_url_1: 5분마다 GET 폴링할 URL (응답의 items[].id 를 추출)
export const API_URL_1 = "https://###/";

// ### api_url_2: 승인 POST 를 보낼 URL
export const API_URL_2 = "https://###/";

// ### 토큰 캡처 대상 URL 패턴 (match pattern 형식, 예: "https://api.example.com/*")
// 대상 페이지가 authorization 헤더를 붙여 호출하는 API origin 을 지정합니다.
// manifest.json 의 host_permissions 에 같은 origin 이 포함되어 있어야 합니다.
export const CAPTURE_URL_PATTERNS = ["https://###/*"];

// ### POST body 값: {id, groupId, approvedBy}
export const GROUP_ID = "###";
export const APPROVED_BY = "###";

// ### x-request-group-id 헤더의 fallback 값
// 페이지 요청에서 자동 캡처되면 캡처값을 사용하고, 캡처 전에만 이 값을 사용합니다.
// (POST body 의 groupId 와는 다른 값일 수 있습니다.)
export const X_REQUEST_GROUP_ID_FALLBACK = "###";

// ### 요청 헤더 (DevTools "Copy as fetch" 에 보이는 값 그대로 기입)
export const ACCEPT = "application/json, text/plain, */*";
export const ACCEPT_LANGUAGE = "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7";

// ### referrer: 대상 페이지 URL
// (service worker 에는 document 가 없어 best-effort 로만 적용됩니다)
export const REFERRER = "https://###/";

// 폴링 주기 (분)
export const POLL_MINUTES = 5;
