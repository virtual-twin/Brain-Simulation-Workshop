#!/usr/bin/env bash
# Regenerate img/videos/tvbo-ext-ontology-simulation.mp4
# (the interactive-simulation demo of the tvb-ext-ontology JupyterLab extension).
#
# End to end: launch JupyterLab (tvbo venv) with the extension -> Playwright records
# the walkthrough (vid-tvb-ext-ontology-simulation.js) -> ffmpeg trims the static
# compile/wait segment and encodes the final MP4.
#
# Requirements:
#   - a Python env with jupyterlab + tvbo + tvboptim + the tvb-ext-ontology extension
#   - node + a node_modules that has playwright (the extension repo's node_modules)
#   - ffmpeg
#   - macOS arm64: libcairo from Homebrew (DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib)
#
# Override any of the paths below via the environment.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

VENV="${VENV:-$HOME/tools/tvbo/.venv}"
JUPYTER="${JUPYTER:-$VENV/bin/jupyter}"
EXT_NODE_MODULES="${EXT_NODE_MODULES:-$HOME/projects/TVB-O/tvb-ext-ontology/node_modules}"
DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:-/opt/homebrew/lib}"
export DYLD_FALLBACK_LIBRARY_PATH

PORT="${PORT:-8889}"
BASE="http://127.0.0.1:$PORT/lab"
WORKDIR="${WORKDIR:-/tmp/tvbo-lab-work}"          # jupyter cwd + where BIDS results are written
RECDIR="${RECDIR:-/tmp/tvbo-rec}"                 # raw recording artifacts
OUT="${OUT:-$REPO/img/videos/tvbo-ext-ontology-simulation.mp4}"

mkdir -p "$WORKDIR" "$RECDIR/video"
rm -rf "${WORKDIR:?}/"* "$RECDIR/video/"*

# --- launch JupyterLab ---
# (Note: a `interpolate_delays` coupling-kwarg mismatch between dev-tvbo and tvboptim
#  was fixed 2026-06-15. If it ever regresses, add a jupyter --config that strips the
#  kwarg in tvboptim.experimental...coupling.base.AbstractCoupling.__init__.)
echo "[1/4] launching JupyterLab on $BASE ..."
"$JUPYTER" lab --no-browser --port="$PORT" --ip=127.0.0.1 \
  --ServerApp.token='' --ServerApp.password='' --ServerApp.disable_check_xsrf=True \
  --ServerApp.open_browser=False --notebook-dir="$WORKDIR" > "$RECDIR/jupyter.log" 2>&1 &
JL_PID=$!
trap 'kill $JL_PID 2>/dev/null || true' EXIT

for i in $(seq 1 60); do
  if curl -s -o /dev/null -w '%{http_code}' "$BASE" 2>/dev/null | grep -q 200; then break; fi
  sleep 1
done

# --- record ---
echo "[2/4] recording walkthrough (the real simulation takes ~50s) ..."
MARKS="$RECDIR/video/marks.json"
NODE_PATH="$EXT_NODE_MODULES" BASE="$BASE" VIDEO_DIR="$RECDIR/video" MARKS_FILE="$MARKS" \
  node "$HERE/vid-tvb-ext-ontology-simulation.js"

# --- trim the static compile/wait segment + encode ---
echo "[3/4] trimming + encoding ..."
read -r A_END B_START B_END VIDEO < <(python3 - "$MARKS" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
a_end = round(m["run_click"] + 3.0, 2)      # interactions + click + a few seconds "running"
b_start = round(m["run_done"] - 1.5, 2)     # resume just before the success message
b_end = round(m["end"], 2)
print(a_end, b_start, b_end, m["video"])
PYEOF
)

# segment A = interactions (lightly sped up 1.25x); segment B = result + file reveal (normal).
ffmpeg -y -loglevel error -i "$VIDEO" -filter_complex \
  "[0:v]trim=0:$A_END,setpts=(PTS-STARTPTS)/1.25[a];[0:v]trim=$B_START:$B_END,setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1[v]" \
  -map "[v]" -c:v libx264 -pix_fmt yuv420p -crf 26 -movflags +faststart -an "$OUT"

echo "[4/4] done -> $OUT"
ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$OUT" | xargs printf 'duration: %ss\n'
