#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DB="${IRIS_DB:-}"

if [ -z "${DB}" ]; then
  if [ -f "${APP_DIR}/../iris_users_web.json" ]; then
    DB="${APP_DIR}/../iris_users_web.json"
  else
    DB="${APP_DIR}/data/iris_users_web.json"
  fi
fi

mkdir -p "$(dirname "${DB}")"
printf '{\n  "version": 2,\n  "users": {}\n}\n' > "${DB}"
echo "Reset database: ${DB}"
