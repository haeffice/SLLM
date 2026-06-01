const path = require("path");
const electron = require("electron");
const { app, BrowserWindow, ipcMain } = electron;

const { serverConfig } = require("./server-config");
const { installCertHandler } = require("./cert");
const { installLoopbackHandler } = require("./audio-loopback");
const settingsStore = require("./settings-store");

let mainWindow = null;

// Initialize the Widevine CDM so DRM/EME content (e.g. protected streams in the
// embedded <webview>) can play. The `components` API only exists on a
// Widevine-enabled Electron build (castlabs/electron-releases); on vanilla
// Electron this is a no-op, so the app still starts without DRM. See README.
async function initWidevine() {
  const { components } = electron;
  if (!components || typeof components.whenReady !== "function") {
    console.log("Widevine: components API unavailable (non-DRM Electron build)");
    return;
  }
  try {
    await components.whenReady();
    console.log("Widevine: ready —", components.status && components.status());
  } catch (e) {
    console.warn("Widevine: initialization failed, continuing without DRM:", e);
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 820,
    title: "SLLM Translate",
    webPreferences: {
      preload: path.join(__dirname, "..", "preload", "preload.js"),
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      webviewTag: true,
    },
  });

  // Keep any attached <webview> locked down: no node integration, normal
  // verification, and popups loaded in-place rather than spawning windows.
  mainWindow.webContents.on("did-attach-webview", (_event, webContents) => {
    webContents.setWindowOpenHandler(({ url }) => {
      webContents.loadURL(url);
      return { action: "deny" };
    });
  });

  mainWindow.loadFile(path.join(__dirname, "..", "renderer", "index.html"));
}

app.whenReady().then(async () => {
  // Must run before any window loads DRM content.
  await initWidevine();

  installCertHandler();
  installLoopbackHandler();

  ipcMain.handle("config:get", () => serverConfig());
  ipcMain.handle("settings:load", () => settingsStore.load());
  ipcMain.handle("settings:save", (_event, settings) => settingsStore.save(settings));

  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
