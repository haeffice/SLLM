// Wires Windows system-audio (loopback) capture for getDisplayMedia().
// The renderer calls navigator.mediaDevices.getDisplayMedia({video,audio});
// this handler answers with `audio: 'loopback'` so Chromium captures the
// Windows system mix instead of a specific window/tab. video is required for
// the request to be honored — the renderer stops the video track immediately.

const { session, desktopCapturer } = require("electron");

function installLoopbackHandler() {
  session.defaultSession.setDisplayMediaRequestHandler(
    (request, callback) => {
      desktopCapturer
        .getSources({ types: ["screen"] })
        .then((sources) => {
          callback({ video: sources[0], audio: "loopback" });
        })
        .catch(() => callback({}));
    },
    { useSystemPicker: false }
  );
}

module.exports = { installLoopbackHandler };
