// Accept the server's self-signed certificate — but ONLY for the configured
// server host. Every other host (including the embedded browser's Google
// traffic, which uses a separate session) goes through Chromium's normal
// verification. Fail closed when no server host is configured.

const { session } = require("electron");
const { serverHost } = require("./settings-store");

function installCertHandler() {
  session.defaultSession.setCertificateVerifyProc((req, callback) => {
    const allowed = serverHost(); // effective host (user override or baked-in)
    if (allowed && req.hostname === allowed) {
      callback(0); // trust: our own server's self-signed cert
    } else {
      callback(-3); // use Chromium's default verification for everything else
    }
  });
}

module.exports = { installCertHandler };
