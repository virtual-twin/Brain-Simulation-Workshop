#!/usr/bin/env sh
# Quarto pre-render sync: copy the demo videos produced by the tvbo-platform
# e2e demo (tests/e2e/demo-output/*.mp4) into this repo's img/videos/, so the
# slides always embed the freshest recordings.
#
# The committed copies in img/videos/ are the source of truth for any build that
# does NOT have tvbo-platform checked out beside this repo (CI, collaborators,
# the published _site). In that case this script warns and keeps them — it never
# fails the render. Set TVBO_PLATFORM_DIR to override the platform location.
#
# POSIX sh on purpose (no bashisms): Quarto may run pre-render scripts via /bin/sh.
set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PLATFORM_DIR=${TVBO_PLATFORM_DIR:-"$REPO_ROOT/../tvbo-platform"}
SRC_DIR="$PLATFORM_DIR/tests/e2e/demo-output"
DEST_DIR="$REPO_ROOT/img/videos"

# Videos that originate from tvbo-platform. (Other img/videos/*.mp4 are produced
# in this repo and are intentionally NOT touched here.)
VIDEOS="tvbo-experiment-builder-workflow.mp4 tvbo-run-in-python.mp4 tvbo-specify-a-model.mp4"

if [ ! -d "$SRC_DIR" ]; then
  echo "[sync-tvbo-videos] tvbo-platform demo-output not found at $SRC_DIR — keeping committed copies." >&2
  exit 0
fi

mkdir -p "$DEST_DIR"
for v in $VIDEOS; do
  src="$SRC_DIR/$v"
  dest="$DEST_DIR/$v"
  if [ ! -f "$src" ]; then
    echo "[sync-tvbo-videos] source missing, keeping committed copy: $v" >&2
    continue
  fi
  if cmp -s "$src" "$dest" 2>/dev/null; then
    echo "[sync-tvbo-videos] up to date: $v"
  else
    cp "$src" "$dest"
    echo "[sync-tvbo-videos] updated:    $v"
  fi
done
