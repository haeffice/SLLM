import {
  API_URL_1,
  API_URL_2,
  CAPTURE_URL_PATTERNS,
  GROUP_ID,
  APPROVED_BY,
  X_REQUEST_GROUP_ID_FALLBACK,
  ACCEPT,
  ACCEPT_LANGUAGE,
  REFERRER,
  POLL_MINUTES,
} from "./config.js";

const ALARM_NAME = "poll";

// [토큰 캡처] 대상 페이지가 자체적으로 보내는 요청에서 authorization(Bearer) /
// x-request-group-id 헤더를 읽어 chrome.storage.session 에 저장합니다.
// 비차단(observational) webRequest 는 MV3 에서 모든 확장 프로그램에 허용되며,
// "extraHeaders" 없이는 authorization 헤더가 보이지 않습니다.
chrome.webRequest.onBeforeSendHeaders.addListener(
  (details) => {
    // 이 확장 프로그램이 직접 보낸 요청은 캡처 대상에서 제외
    if (details.initiator === location.origin) return;

    const captured = {};
    for (const header of details.requestHeaders ?? []) {
      const name = header.name.toLowerCase();
      if (name === "authorization" && header.value) captured.authorization = header.value;
      if (name === "x-request-group-id" && header.value) captured.xRequestGroupId = header.value;
    }
    if (Object.keys(captured).length > 0) chrome.storage.session.set(captured);
  },
  { urls: CAPTURE_URL_PATTERNS },
  ["requestHeaders", "extraHeaders"]
);

// [알람 등록] onInstalled / onStartup 양쪽에서 생성해야
// 확장 재설치·업데이트와 브라우저 재시작 모두에서 알람이 유지됩니다.
chrome.runtime.onInstalled.addListener(() => {
  chrome.alarms.create(ALARM_NAME, { periodInMinutes: POLL_MINUTES });
});
chrome.runtime.onStartup.addListener(() => {
  chrome.alarms.create(ALARM_NAME, { periodInMinutes: POLL_MINUTES });
});

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === ALARM_NAME) return pollCycle();
});

function buildHeaders(authorization, xRequestGroupId) {
  return {
    accept: ACCEPT,
    "accept-language": ACCEPT_LANGUAGE,
    authorization,
    "x-request-group-id": xRequestGroupId || X_REQUEST_GROUP_ID_FALLBACK,
    // sec-ch-ua, sec-ch-ua-mobile, sec-ch-ua-platform,
    // sec-fetch-dest, sec-fetch-mode, sec-fetch-site:
    // "Sec-" 접두사는 Fetch 스펙상 forbidden header 라서 확장 프로그램에서 설정할 수
    // 없습니다 (지정해도 무시되며, 브라우저가 자체 값을 자동으로 부착합니다).
  };
}

async function pollCycle() {
  const { authorization, xRequestGroupId } = await chrome.storage.session.get([
    "authorization",
    "xRequestGroupId",
  ]);
  if (!authorization) {
    console.log("[poll] authorization 미캡처 — 대상 페이지를 한 번 열어 주세요. skip");
    return;
  }
  const headers = buildHeaders(authorization, xRequestGroupId);

  // 1) GET api_url_1 → items[].id 추출 (오류·빈 배열이면 skip 후 다음 주기 대기)
  let ids;
  try {
    const res = await fetch(API_URL_1, {
      headers,
      referrer: REFERRER,
      body: null,
      method: "GET",
      mode: "cors",
      credentials: "include",
    });
    if (!res.ok) {
      console.log(`[poll] GET 응답 ${res.status} — skip`);
      return;
    }
    const data = await res.json();
    ids = (Array.isArray(data.items) ? data.items : [])
      .map((item) => item.id)
      .filter((id) => typeof id === "string" && id.length > 0);
  } catch (error) {
    console.log("[poll] GET 실패 — skip:", error);
    return;
  }
  if (ids.length === 0) {
    console.log("[poll] items 비어 있음 — skip");
    return;
  }

  // 2) 각 id 마다 POST api_url_2 (한 id 의 실패가 나머지를 막지 않도록 allSettled)
  console.log(`[poll] ${ids.length}개 id 승인 요청:`, ids);
  const results = await Promise.allSettled(
    ids.map((id) =>
      fetch(API_URL_2, {
        headers: { ...headers, "content-type": "application/json" },
        referrer: REFERRER,
        body: JSON.stringify({ id, groupId: GROUP_ID, approvedBy: APPROVED_BY }),
        method: "POST",
        mode: "cors",
        credentials: "include",
      })
    )
  );
  results.forEach((result, i) => {
    if (result.status === "fulfilled") {
      console.log(`[approve] ${ids[i]} → HTTP ${result.value.status}`);
    } else {
      console.log(`[approve] ${ids[i]} 실패:`, result.reason);
    }
  });
}
