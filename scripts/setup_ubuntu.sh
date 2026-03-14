#!/usr/bin/env bash
set -euo pipefail

FLAP_ROOT="/home/parshu/flap-os"
SYSTEMD_DIR="/etc/systemd/system"

if [[ "${EUID}" -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

echo "[FLAP] Starting Ubuntu setup on $(hostname)"

if [[ ! -d "${FLAP_ROOT}" ]]; then
  echo "[FLAP] Expected project directory not found: ${FLAP_ROOT}"
  echo "[FLAP] Clone flap-os to ${FLAP_ROOT} before running this script."
  exit 1
fi

${SUDO} apt-get update
${SUDO} apt-get install -y ca-certificates curl gnupg lsb-release jq git build-essential python3-venv python3-pip

if ! command -v docker >/dev/null 2>&1; then
  echo "[FLAP] Installing Docker Engine..."
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | ${SUDO} gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  ${SUDO} chmod a+r /etc/apt/keyrings/docker.gpg
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "${VERSION_CODENAME}") stable" | ${SUDO} tee /etc/apt/sources.list.d/docker.list >/dev/null
  ${SUDO} apt-get update
  ${SUDO} apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[FLAP] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if ! command -v node >/dev/null 2>&1; then
  echo "[FLAP] Installing Node.js 20..."
  curl -fsSL https://deb.nodesource.com/setup_20.x | ${SUDO} -E bash -
  ${SUDO} apt-get install -y nodejs
fi

${SUDO} usermod -aG docker parshu || true

if [[ ! -f "${FLAP_ROOT}/.env" ]]; then
  cp "${FLAP_ROOT}/.env.example" "${FLAP_ROOT}/.env"
  echo "[FLAP] Created ${FLAP_ROOT}/.env (fill secrets before production use)."
fi

cd "${FLAP_ROOT}"
docker compose pull
docker compose up -d

if [[ -f "${FLAP_ROOT}/scripts/systemd/flap-compose.service" ]]; then
  ${SUDO} cp "${FLAP_ROOT}/scripts/systemd/flap-compose.service" "${SYSTEMD_DIR}/flap-compose.service"
fi
if [[ -f "${FLAP_ROOT}/scripts/systemd/screenpipe.service" ]]; then
  ${SUDO} cp "${FLAP_ROOT}/scripts/systemd/screenpipe.service" "${SYSTEMD_DIR}/screenpipe.service"
fi

${SUDO} systemctl daemon-reload
${SUDO} systemctl enable flap-compose.service
${SUDO} systemctl restart flap-compose.service
${SUDO} systemctl enable screenpipe.service
${SUDO} systemctl restart screenpipe.service

echo "[FLAP] Setup complete."
echo "[FLAP] Services:"
${SUDO} systemctl --no-pager --full status flap-compose.service | sed -n '1,12p'
${SUDO} systemctl --no-pager --full status screenpipe.service | sed -n '1,12p'
