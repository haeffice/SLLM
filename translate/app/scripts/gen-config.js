// Build-time server-URL injection.
// Writes src/generated/build-config.js from $SLLM_SERVER_URL so the default in
// source stays empty. Run via `npm run gen:config` (invoked by start/build).
const fs = require("fs");
const path = require("path");

const base = process.env.SLLM_SERVER_URL || ""; // e.g. https://192.168.0.42:9001
const outDir = path.join(__dirname, "..", "src", "generated");
const outFile = path.join(outDir, "build-config.js");

fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(
  outFile,
  `module.exports = { SERVER_URL: ${JSON.stringify(base)} };\n`
);

console.log(`gen-config: SERVER_URL=${JSON.stringify(base)} -> ${outFile}`);
