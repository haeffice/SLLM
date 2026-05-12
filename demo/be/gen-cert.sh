#!/usr/bin/env bash
# Usage: ./gen-cert.sh <LAN_IP>
# Generates cert.pem / key.pem for HTTPS uvicorn with SAN covering
# localhost, 127.0.0.1, and the provided LAN IP (required by modern
# browsers when accessing via IP literal).
set -euo pipefail

LAN_IP="${1:?Usage: $0 <LAN_IP> (e.g., 192.168.0.42)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout key.pem -out cert.pem -days 365 \
  -subj "/CN=demo" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:${LAN_IP}"

chmod 600 key.pem
echo "Generated cert.pem / key.pem in $SCRIPT_DIR"
echo "SAN: DNS:localhost, IP:127.0.0.1, IP:${LAN_IP}"
