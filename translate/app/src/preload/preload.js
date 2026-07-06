const { contextBridge, ipcRenderer } = require("electron");

// The View menu (main process) toggles the debug panel via "menu:debug"; route it
// to the renderer's current callback (set via api.onMenuDebug). Registered once.
let onMenuDebugCb = null;
ipcRenderer.on("menu:debug", (_event, val) => {
  if (onMenuDebugCb) onMenuDebugCb(val);
});

// Minimal, frozen bridge. No raw ipcRenderer or Node is exposed to the page.
contextBridge.exposeInMainWorld("api", {
  getServerConfig: () => ipcRenderer.invoke("config:get"),
  loadSettings: () => ipcRenderer.invoke("settings:load"),
  saveSettings: (settings) => ipcRenderer.invoke("settings:save", settings),
  // Debug panel ↔ View-menu wiring.
  onMenuDebug: (cb) => {
    onMenuDebugCb = typeof cb === "function" ? cb : null;
  },
  setDebugState: (val) => ipcRenderer.send("debug:state", !!val),
  getDebugState: () => ipcRenderer.invoke("debug:get"),
});
