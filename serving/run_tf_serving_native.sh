#!/usr/bin/env bash
# Native TensorFlow Serving (Linux, installed via APT: `tensorflow-model-server`).
# Uses the generated config whose base_path is the absolute path on this machine.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
nohup tensorflow_model_server \
  --rest_api_port=8501 \
  --model_config_file="${HERE}/models/models.config" \
  --model_config_file_poll_wait_seconds=10 \
  --allow_version_labels_for_unavailable_models=true \
  > "${HERE}/server.log" 2>&1 &
echo "tensorflow_model_server started (pid $!), logs: ${HERE}/server.log"
