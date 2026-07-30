#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export VLA_PROTOCOL_PROFILE="mobile_joint_base"
export VLA_ARM_SIDE="both"
export VLA_BASE_URL="${VLA_BASE_URL:-http://127.0.0.1:18027}"
export VLA_PROMPT="${VLA_PROMPT:-Put the black block into the bowl with the left hand.}"
export VLA_SOURCE_ROS_HUMBLE="1"

exec "${SCRIPT_DIR}/start_real_robot_vla.sh" \
  --base-command-source provider \
  --base-motion \
  --base-velocity-frame base_link \
  --online-inference-chunk-step-mode per_tick \
  "$@"
