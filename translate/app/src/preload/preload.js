const { contextBridge, ipcRenderer } = require("electron");

// Minimal, frozen bridge. No raw ipcRenderer or Node is exposed to the page.
contextBridge.exposeInMainWorld("api", {
  getServerConfig: () => ipcRenderer.invoke("config:get"),
  loadSettings: () => ipcRenderer.invoke("settings:load"),
  saveSettings: (settings) => ipcRenderer.invoke("settings:save", settings),
});
