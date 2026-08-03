#!/usr/bin/env bash
# One-command rollback for the gateway. Two levers (Day 5):
#   1. rollout undo    -> revert to the previous Deployment revision (image/config)
#   2. model revision  -> change the model version in the ConfigMap + restart
#
# Usage:
#   ./scripts/rollback.sh deploy                 # undo last gateway rollout
#   ./scripts/rollback.sh deploy 3               # roll back to revision 3
#   ./scripts/rollback.sh model v1.0.0           # switch served model version
set -euo pipefail
NS=domainbot
MODE="${1:-}"

case "$MODE" in
  deploy)
    REV="${2:-}"
    if [ -n "$REV" ]; then
      kubectl -n "$NS" rollout undo deployment/domainbot-gateway --to-revision="$REV"
    else
      kubectl -n "$NS" rollout undo deployment/domainbot-gateway
    fi
    kubectl -n "$NS" rollout status deployment/domainbot-gateway
    ;;
  model)
    VERSION="${2:?need a version, e.g. v1.0.0}"
    kubectl -n "$NS" patch configmap domainbot-config \
      --type merge -p "{\"data\":{\"DOMAINBOT_REVISION\":\"$VERSION\"}}"
    # restart so pods pick up the new env
    kubectl -n "$NS" rollout restart deployment/domainbot-gateway
    kubectl -n "$NS" rollout status deployment/domainbot-gateway
    echo ">> switched model revision to $VERSION"
    ;;
  *)
    echo "usage: $0 {deploy [revision] | model <version>}"
    exit 1
    ;;
esac
