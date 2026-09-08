#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
[[ "$#" -ge 1 ]] || { echo "Uso: $0 CONFIG [SESSIONE] [-- OPZIONI_TRAIN...]" >&2; exit 2; }
requested="$1"
if [[ "$requested" != /* ]]; then requested="$REPO/$requested"; fi
absolute="$(realpath -e -- "$requested")"
case "$absolute" in
  "$REPO"/*) CONFIG="${absolute#"$REPO"/}" ;;
  *) echo "La config deve essere nel repository montato" >&2; exit 2 ;;
esac
shift
SESSION="$(basename -- "${CONFIG%.yaml}")"
if [[ "$#" -gt 0 && "$1" != "--" ]]; then SESSION="$1"; shift; fi
if [[ "$#" -gt 0 && "$1" == "--" ]]; then shift; fi
exec bash "$SCRIPT_DIR/run_command.sh" "$SESSION" -- train --config "$CONFIG" "$@"
