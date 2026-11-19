#!/bin/bash
# Region DINO: Bottleneck Attention (reg_subset+L) x reg_subset x (reg_subset+L)

#SBATCH --job-name=tim
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=ccvl32
set -e


# source /path/to/venv/bin/activate   # <- activate your Python env

# ============== Configuration ==============
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${PROJECT_DIR}"
DATA_PATH="./data/imagenet"

ARCH="region_vit_base"
PATCH_SIZE=16
NUM_REGISTER_TOKENS=192  # Must be divisible by num_heads (6 for vit_small), 96/6=16 regs per head
NUM_REG_FOR_LOSS_PER_HEAD=16  # Registers per head that contribute to loss
USE_CROSS_ATTN=true     # Bottleneck: (reg_subset+L) x reg_subset x (reg_subset+L)
DICT_SIZE=0  #1024      # Dictionary codes (0 to disable)
DICT_TAU=0.1            # Temperature for codebook soft assignment
TOP_N=6                 # Top-N registers by max activation for dominant map
SPATIAL_SIGMA=0.8       # Gaussian smoothing sigma for spatial coherence
MASK_TAU=0.1            # Temperature for cross-register softmax



EPOCHS=400
BATCH_SIZE=32
LR=0.00075
WARMUP_EPOCHS=10


NGPUS=8
NUM_WORKERS=8

OUTPUT="${OUTPUT_DIR}/output/${ARCH}_ps${PATCH_SIZE}_ep${EPOCHS}_N${NUM_REGISTER_TOKENS}_loss${NUM_REG_FOR_LOSS_PER_HEAD}_cross${USE_CROSS_ATTN}_dict${DICT_SIZE}"

# ============== Setup ==============
cd ${PROJECT_DIR}
mkdir -p ${OUTPUT}

# ============== Training ==============
echo "[TRAIN] ${ARCH} (N=${NUM_REGISTER_TOKENS}, loss_per_head=${NUM_REG_FOR_LOSS_PER_HEAD}, cross_attn=${USE_CROSS_ATTN})"
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
    --data_path ${DATA_PATH}/train \
    --output_dir ${OUTPUT} \
    --batch_size_per_gpu ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --lr ${LR} \
    --min_lr 1e-6 \
    --weight_decay 0.04 \
    --weight_decay_end 0.4 \
    --clip_grad 0.3 \
    --out_dim 65536 \
    --local_crops_number 10 \
    --global_crops_scale 0.25 1.0 \
    --local_crops_scale 0.05 0.25 \
    --momentum_teacher 0.996 \
    --teacher_temp 0.07 \
    --warmup_teacher_temp_epochs 50 \
    --freeze_last_layer 3 \
    --precision bf16 \
    --num_workers ${NUM_WORKERS}

echo "Done! Output: ${OUTPUT}"
