// Resolves the build-injected server URL and parses it into base/host/port.
// The default in source is empty; a fixed value is baked at build time by
// scripts/gen-config.js. The user may still override `serverUrl` at runtime
// via settings (see settings.js) — that override is read in the renderer.

let injected = { SERVER_URL: "" };
try {
  injected = require("../generated/build-config");
} catch (_) {
  // build-config not generated yet — fall back to empty.
}

function parse(url) {
  if (!url) return { base: "", host: "", port: "" };
  try {
    const u = new URL(url);
    return { base: url.replace(/\/$/, ""), host: u.hostname, port: u.port };
  } catch (_) {
    return { base: "", host: "", port: "" };
  }
}

const config = parse(injected.SERVER_URL || "");

function serverConfig() {
  return config;
}

module.exports = { serverConfig };
