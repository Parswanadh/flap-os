#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

echo "[FLAP] Installing/configuring Tailscale..."

if ! command -v tailscale >/dev/null 2>&1; then
  curl -fsSL https://tailscale.com/install.sh | sh
fi

${SUDO} systemctl enable tailscaled
${SUDO} systemctl restart tailscaled

if [[ -n "${TAILSCALE_AUTH_KEY:-}" ]]; then
  ${SUDO} tailscale up --authkey "${TAILSCALE_AUTH_KEY}" --ssh --accept-routes
else
  echo "[FLAP] No TAILSCALE_AUTH_KEY found. Running interactive tailscale up..."
  ${SUDO} tailscale up --ssh --accept-routes
fi

echo "[FLAP] Tailscale status:"
${SUDO} tailscale status
