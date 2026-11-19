#!/bin/bash
#
# Run iterative decomposition + similarity grouping (rats_iter.py) over demo images.
# One figure per image:
#   TOP row     : IMAGE | L1 | L2 | ... | Ld (over-seg) | grouped (sim)
#   BOTTOM rows : one row per L1 mask -> isolated mask | L2 | ... | Ld
#
# Usage:
#   scripts/run_infer_rats.sh                 # all images in demo_image1/ with defaults
#   SIZE=2048 MERGE_SIM=0.5 scripts/run_infer_rats.sh   # override params via env vars
#   scripts/run_infer_rats.sh path/to/img.png # run a single image

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${SCRIPT_DIR}"

REPO="${SCRIPT_DIR}"; source "$(dirname "$0")/fetch_checkpoint.sh"   # auto-download from HF if needed

# ---- params (override via env) ----
SIZE="${SIZE:-1024}"     # input resolution
TOP_N="${TOP_N:-10}"     # branching per iteration: an int (fixed) or 'auto' (adaptive)
DEPTH="${DEPTH:-4}"      # recursive decomposition depth
MERGE_SIM="${MERGE_SIM:-0.85}"  # merge leaf segments with cosine similarity >= this
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/output/rats_iter}"

mkdir -p "${OUT_DIR}"

# ---- collect images ----
if [ "$#" -ge 1 ]; then
    IMAGES=("$@")
else
    IMAGES=()
    for img in "${SCRIPT_DIR}"/demo_image1/*.JPEG \
               "${SCRIPT_DIR}"/demo_image1/*.jpg \
               "${SCRIPT_DIR}"/demo_image1/*.png; do
        [ -f "$img" ] && IMAGES+=("$img")
    done
fi

echo "Params: size=${SIZE} top_n=${TOP_N} depth=${DEPTH} merge_sim=${MERGE_SIM}"
echo "Images: ${#IMAGES[@]}  ->  ${OUT_DIR}"

for img in "${IMAGES[@]}"; do
    [ -f "$img" ] || continue
    name=$(basename "${img%.*}")
    echo "Processing: ${name}"
    python rats_iter.py \
        --image "${img}" \
        --size "${SIZE}" \
        --top_n "${TOP_N}" \
        --depth "${DEPTH}" \
        --merge_sim "${MERGE_SIM}" \
        --out_dir "${OUT_DIR}"
done

echo "All done! Results in ${OUT_DIR}"
