#!/bin/bash
#
# Learned-dictionary part retrieval on PartImageNet (region_vit_base, dict 4096).
# Auto-downloads the released dict-only checkpoint (backbone + codebook) from the Hub,
# then runs the full pipeline: build the codebook↔part dictionary on PartImageNet
# TRAIN, evaluate part purity on VAL, and render the visualizations.
#
# Needs PartImageNet (symlink under ./data/PartImageNet or set PARTIMAGENET_ROOT).
#   bash retrieval_learnable/run_base_dict4096.sh
#   MAX_IMAGES=2000 bash retrieval_learnable/run_base_dict4096.sh   # smaller/faster run

#SBATCH --job-name=retrieval_base_dict4096
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

export PYTHONUNBUFFERED=1
set -e
# source /path/to/venv/bin/activate   # <- activate your Python env

REPO="$(cd "$(dirname "$0")/.." && pwd)"
export MODEL_DIR="${REPO}"                 # demo_topk.py does sys.path.insert(0, MODEL_DIR)
cd "${REPO}/retrieval_learnable"           # so `python3 demo_topk.py` resolves

# ---- Data / output (symlink your PartImageNet under ./data, or set PARTIMAGENET_ROOT) ----
PARTIMAGENET_ROOT="${PARTIMAGENET_ROOT:-${REPO}/data/PartImageNet}"
OUTPUT_DIR="${REPO}/output/retrieval_base_dict4096"
MAX_IMAGES="${MAX_IMAGES:-20000}"

# ---- Weights: released base backbone (ep0388) + dict codebook, auto-downloaded ----
# The backbone is the released ep0388 teacher; the small codebook is stitched onto it.
# Override with a local full dict checkpoint by setting CKPT and dropping --codebook.
CKPT="${CKPT:-$(python3 -c "from huggingface_hub import hf_hub_download as d; print(d('TimingYang/RATS','base/checkpoint_teacher_ep0388.pth'))")}"
CODEBOOK="${CODEBOOK:-$(python3 -c "from huggingface_hub import hf_hub_download as d; print(d('TimingYang/RATS','dict/codebook_base_dict4096.pth'))")}"

K=4096
DICT_SIZE=4096
ARCH=region_vit_base
NUM_REGISTER_TOKENS=192
NUM_REG_FOR_LOSS_PER_HEAD=16
TOP_N=40
SCRIPT=demo_topk.py        # register matching; swap demo_topk_patch.py for patch matching

COMMON_MODEL_ARGS=(
    --arch ${ARCH}
    --num_register_tokens ${NUM_REGISTER_TOKENS}
    --num_reg_for_loss_per_head ${NUM_REG_FOR_LOSS_PER_HEAD}
    --top_n ${TOP_N}
    --K ${K}
    --dict_size ${DICT_SIZE}
    --codebook ${CODEBOOK}
)

echo "==================== STEP 1: cluster (codebook ↔ PartImageNet train GT) ===================="
python3 ${SCRIPT} cluster \
    --partimagenet_dir ${PARTIMAGENET_ROOT} \
    --pretrained_weights ${CKPT} \
    --output_dir ${OUTPUT_DIR} \
    "${COMMON_MODEL_ARGS[@]}" \
    --max_images ${MAX_IMAGES} \
    --topk_registers 20 \
    --topk_method max \
    --vote_metric attention \
    --concentration_thresh 0.5 \
    --split train \
    --image_size 512 512

EVAL_COMMON=(
    --partimagenet_dir ${PARTIMAGENET_ROOT}
    --pretrained_weights ${CKPT}
    --feature_dir ${OUTPUT_DIR}
    "${COMMON_MODEL_ARGS[@]}"
    --split val
    --image_size 512 512
    --max_images ${MAX_IMAGES}
)

echo "==================== STEP 2a: eval extract ===================="
python3 ${SCRIPT} eval_parts "${EVAL_COMMON[@]}" --eval_stage extract
echo "==================== STEP 2b: eval analyze ===================="
python3 ${SCRIPT} eval_parts "${EVAL_COMMON[@]}" --eval_stage analyze
echo "==================== STEP 2c: eval visualize ===================="
python3 ${SCRIPT} eval_parts "${EVAL_COMMON[@]}" --eval_stage visualize

echo "==================== ALL DONE ===================="
echo "Results: ${OUTPUT_DIR}"
