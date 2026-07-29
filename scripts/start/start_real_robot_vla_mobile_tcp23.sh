#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export VLA_PROTOCOL_PROFILE="mobile_tcp23"
export VLA_ARM_SIDE="both"
export VLA_PROMPT="${VLA_PROMPT:-Put the black block into the bowl with the left hand.}"
export VLA_SOURCE_ROS_HUMBLE="1"

exec "${SCRIPT_DIR}/start_real_robot_vla.sh" \
  --base-command-source provider \
  --base-motion \
  --mobile-manipulation-mode direct_ik \
  --base-velocity-frame base_link \
  "$@"
