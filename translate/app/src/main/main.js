const path = require("path");
const electron = require("electron");
const { app, BrowserWindow, ipcMain, Menu } = electron;

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

// Application menu: the standard roles (File/Edit/View/Window/Help) plus a
// View → 디버그 패널 checkbox that toggles the renderer's in-app WS debug panel.
// The checkbox and the panel's own ✕ button stay in sync over IPC (menu:debug /
// debug:state); the renderer can read the current state via debug:get.
function installMenu() {
  const isMac = process.platform === "darwin";
  const template = [
    ...(isMac ? [{ role: "appMenu" }] : []),
    { role: "fileMenu" },
    { role: "editMenu" },
    {
      label: "View",
      submenu: [
        {
          id: "debug-panel",
          label: "디버그 패널",
          type: "checkbox",
          checked: false,
          accelerator: "CmdOrCtrl+D",
          click: (item) => {
            if (mainWindow) mainWindow.webContents.send("menu:debug", item.checked);
          },
        },
        { type: "separator" },
        { role: "reload" },
        { role: "toggleDevTools" },
        { type: "separator" },
        { role: "resetZoom" },
        { role: "zoomIn" },
        { role: "zoomOut" },
        { type: "separator" },
        { role: "togglefullscreen" },
      ],
    },
    { role: "windowMenu" },
    { role: "help", submenu: [{ label: "SR Live Translation", enabled: false }] },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 820,
    title: "SR Live Translation",
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

  // Keep the View → 디버그 패널 checkbox in sync when the panel's ✕ closes it, and
  // let the renderer read the current state on mount.
  const debugItem = () => {
    const menu = Menu.getApplicationMenu();
    return menu && menu.getMenuItemById("debug-panel");
  };
  ipcMain.on("debug:state", (_event, val) => {
    const item = debugItem();
    if (item) item.checked = !!val;
  });
  ipcMain.handle("debug:get", () => {
    const item = debugItem();
    return item ? item.checked : false;
  });

  installMenu();
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
