#!/bin/bash
# Dict-Only Training: freeze the pretrained backbone, train only the DictionaryQuantizer codebook
# Arch: region_vit_base (LxN-NxN-NxL, 3-step cross-attention, N=192)

#SBATCH --job-name=dict_only_base
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
set -e

# ============== Configuration ==============
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_PATH="./data/imagenet"

ARCH="region_vit_base"
PATCH_SIZE=16
NUM_REGISTER_TOKENS=192
NUM_REG_FOR_LOSS_PER_HEAD=16
USE_CROSS_ATTN=true

# Dictionary config
DICT_SIZE=4096
DICT_TAU=0.1
TOP_N=40
SPATIAL_SIGMA=0.8
MASK_TAU=0.1

# Pretrained backbone (no dict trained yet, dict0)
PRETRAINED_CKPT="${PROJECT_DIR}/output/region_vit_base_ps16_ep400_N192_loss16_crosstrue_dict0/checkpoint.pth"

# Training config
EPOCHS=10
BATCH_SIZE=128
LR=0.001
WARMUP_EPOCHS=0

NGPUS=4
NUM_WORKERS=10

OUTPUT="${PROJECT_DIR}/output/${ARCH}_ps${PATCH_SIZE}_dict${DICT_SIZE}_dictonly"

# ============== Setup ==============
# source /path/to/venv/bin/activate   # <- activate your Python env
cd ${PROJECT_DIR}
mkdir -p ${OUTPUT}

# ============== Dict-Only Training ==============
echo "[DICT-ONLY] ${ARCH} (dict_size=${DICT_SIZE}, tau=${DICT_TAU})"
echo "  Pretrained: ${PRETRAINED_CKPT}"
echo "  Output:     ${OUTPUT}"

torchrun --standalone --nproc_per_node=${NGPUS} main_dino.py \
    --arch ${ARCH} \
    --patch_size ${PATCH_SIZE} \
    --num_register_tokens ${NUM_REGISTER_TOKENS} \
    --num_reg_for_loss_per_head ${NUM_REG_FOR_LOSS_PER_HEAD} \
    --use_cross_attn ${USE_CROSS_ATTN} \
    --dict_size ${DICT_SIZE} \
    --dict_tau ${DICT_TAU} \
    --top_n ${TOP_N} \
    --spatial_sigma ${SPATIAL_SIGMA} \
    --mask_tau ${MASK_TAU} \
    --entropy_weight 0.1 \
    --pretrained_checkpoint ${PRETRAINED_CKPT} \
    --train_dict_only true \
    --data_path ${DATA_PATH}/train \
    --output_dir ${OUTPUT} \
    --batch_size_per_gpu ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --lr ${LR} \
    --min_lr 1e-6 \
    --weight_decay 0.04 \
    --weight_decay_end 0.4 \
    --clip_grad 3.0 \
    --out_dim 65536 \
    --local_crops_number 8 \
    --global_crops_scale 0.4 1.0 \
    --local_crops_scale 0.05 0.4 \
    --momentum_teacher 0.996 \
    --use_fp16 true \
    --num_workers ${NUM_WORKERS}

echo "Done! Output: ${OUTPUT}"
