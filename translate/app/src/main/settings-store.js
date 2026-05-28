// Persisted settings, stored as JSON in userData/settings.json.
// The main process owns this file so it can resolve the effective server URL
// (build-injected default, overridable by the user) for the cert handler.

const fs = require("fs");
const path = require("path");
const { app } = require("electron");

const { serverConfig } = require("./server-config");

function defaults() {
  return {
    serverUrl: serverConfig().base, // build-injected default ("" if not baked)
    audioSource: "mic", // "mic" | "system"
    sampleRate: 16000,
    languageDirection: "en2ko", // "en2ko" | "ko2en"
    bandTransparencyPct: 0, // 0 = opaque band, 100 = text only (see overlay-band.js)
    bandWidthPx: 720,
    recentLines: 3,
    hyperparameters: {}, // placeholder, wired for later use
  };
}

function filePath() {
  return path.join(app.getPath("userData"), "settings.json");
}

function load() {
  try {
    const raw = fs.readFileSync(filePath(), "utf-8");
    return { ...defaults(), ...JSON.parse(raw) };
  } catch (_) {
    return defaults();
  }
}

function save(settings) {
  const merged = { ...defaults(), ...settings };
  fs.writeFileSync(filePath(), JSON.stringify(merged, null, 2));
  return merged;
}

// Effective server host for cert scoping: prefer the user override, else the
// build-injected default.
function serverHost() {
  const url = load().serverUrl || serverConfig().base;
  if (!url) return "";
  try {
    return new URL(url).hostname;
  } catch (_) {
    return "";
  }
}

module.exports = { defaults, load, save, serverHost };
