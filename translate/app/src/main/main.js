const path = require("path");
const { app, BrowserWindow, ipcMain } = require("electron");

const { serverConfig } = require("./server-config");
const { installCertHandler } = require("./cert");
const { installLoopbackHandler } = require("./audio-loopback");
const settingsStore = require("./settings-store");

let mainWindow = null;

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

app.whenReady().then(() => {
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
