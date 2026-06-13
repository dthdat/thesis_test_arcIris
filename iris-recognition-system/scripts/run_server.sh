#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${APP_DIR}"

HOST="${IRIS_HOST:-0.0.0.0}"
PORT="${IRIS_PORT:-8000}"
DB="${IRIS_DB:-}"
THRESHOLD="${IRIS_THRESHOLD:-0.30}"

if [ -z "${DB}" ]; then
  if [ -f "../iris_users_web.json" ]; then
    DB="../iris_users_web.json"
  else
    DB="data/iris_users_web.json"
  fi
fi

ENGINE="${IRIS_ENGINE:-}"
if [ -z "${ENGINE}" ]; then
  for candidate in \
    "../iris_model_fp16.engine" \
    "./iris_model_fp16.engine" \
    "../iris_model.engine" \
    "./iris_model.engine"
  do
    if [ -f "${candidate}" ]; then
      ENGINE="${candidate}"
      break
    fi
  done
fi

META="${IRIS_META:-}"
if [ -z "${META}" ]; then
  for candidate in \
    "../iris_model.metadata.json" \
    "./iris_model.metadata.json" \
    "../iris_iresnet50_msff_embedding.metadata.json" \
    "./iris_iresnet50_msff_embedding.metadata.json"
  do
    if [ -f "${candidate}" ]; then
      META="${candidate}"
      break
    fi
  done
fi

if [ -z "${ENGINE}" ] || [ ! -f "${ENGINE}" ]; then
  echo "ERROR: TensorRT engine not found."
  echo "Expected one of:"
  echo "  ${APP_DIR}/../iris_model_fp16.engine"
  echo "  ${APP_DIR}/iris_model_fp16.engine"
  echo "Or set IRIS_ENGINE=/path/to/iris_model_fp16.engine"
  exit 1
fi

if [ -z "${META}" ] || [ ! -f "${META}" ]; then
  echo "ERROR: metadata JSON not found."
  echo "Expected one of:"
  echo "  ${APP_DIR}/../iris_model.metadata.json"
  echo "  ${APP_DIR}/../iris_iresnet50_msff_embedding.metadata.json"
  echo "Or set IRIS_META=/path/to/metadata.json"
  exit 1
fi

mkdir -p uploads data

echo "Starting Iris Recognition System"
echo "App:       ${APP_DIR}/iris_web_app.py"
echo "Engine:    ${ENGINE}"
echo "Metadata:  ${META}"
echo "Database:  ${DB}"
echo "Threshold: ${THRESHOLD}"
echo "Host:      ${HOST}"
echo "Port:      ${PORT}"

python3 iris_web_app.py \
  --engine "${ENGINE}" \
  --meta "${META}" \
  --db "${DB}" \
  --threshold "${THRESHOLD}" \
  --host "${HOST}" \
  --port "${PORT}"
