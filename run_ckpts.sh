#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_ckpts.sh BASE_FILENAME MAX_CKPT [STEPS=8]
[ $# -ge 2 ] || { echo "Usage: $0 BASE_FILENAME MAX_CKPT [STEPS=8]"; exit 1; }

BASE="$1"
MAX="$2"
STEPS="${3:-8}"

export PYOPENGL_PLATFORM=glx  # don't rely on your alias
# If your launcher isn't "vk_pro python", override with: GLX_CMD="mycmd" ./run_ckpts.sh ...
GLX_CMD=${GLX_CMD:-vk_pro python}

CKPTS=$(python3 - "$MAX" "$STEPS" <<'PY'
import sys, math
minv = 100
maxv = int(sys.argv[1])
steps = int(sys.argv[2])

if steps < 2:
    out = [maxv]
else:
    r = (maxv/minv)**(1/(steps-1))
    vals = [int(round((minv*(r**i))/100.0)*100) for i in range(steps)]
    out, seen = [], set()
    for v in vals:
        if v not in seen:
            out.append(v); seen.add(v)
    if out[-1] != maxv:
        out.append(maxv)

print(" ".join(map(str, out)))
PY
)

for ckpt in $CKPTS; do
  echo "=> ckpt=$ckpt"
  $GLX_CMD examples/drone/w_hover_eval.py -e "$BASE" --ckpt "$ckpt" --record
  mv "${BASE}.mp4" "${BASE}_iter${ckpt}.mp4"
done
