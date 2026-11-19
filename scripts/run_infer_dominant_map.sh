#!/bin/bash
#
# Demo runner (no-norm variant): draw_dominant_map.py skips the model's final
# layer norm and uses the raw last-block output tokens. See the [no_norm] block
# in draw_dominant_map.py.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEMO_DIR="${SCRIPT_DIR}/demo_image1"
OUTPUT_DIR="${SCRIPT_DIR}/output/dominant_map"
mkdir -p "${OUTPUT_DIR}"

MODEL_DIR="${SCRIPT_DIR}"
REPO="${SCRIPT_DIR}"; source "$(dirname "$0")/fetch_checkpoint.sh"   # auto-download from HF if needed
WEIGHTS="${CKPT}"

for img in "${DEMO_DIR}"/*.JPEG "${DEMO_DIR}"/*.jpg "${DEMO_DIR}"/*.png; do
    [ -f "$img" ] || continue
    name=$(basename "${img%.*}")
    echo "Processing: ${name}"

    cd "${SCRIPT_DIR}" && \
    python draw_dominant_map.py \
        --model_type dino_attn_base \
        --model_dir "${MODEL_DIR}" \
        --arch region_vit_base \
        --patch_size 16 \
        --num_register_tokens 192 \
        --num_reg_for_loss_per_head 16 \
        --use_cross_attn \
        --pretrained_weights "${WEIGHTS}" \
        --checkpoint_key teacher \
        --image_size 1024 1024 \
        --image_path "${img}" \
        --output_path "${OUTPUT_DIR}/${name}.png" \
        --metric similarity \
        --top_n 10 \
        --select_method max \
        --dominant_norm spatial_smooth \
        --sim_mode raw \
        --smooth_sigma 1.5 \
        --dominant_mode all_per_head \
        --clean_min_patches_frac 0.005
done

echo "All done! Results in ${OUTPUT_DIR}"
