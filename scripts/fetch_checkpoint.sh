#!/bin/bash
# Sourced by the inference scripts. Ensures ${CKPT} (a checkpoint_teacher.pth at the
# repo root) exists, downloading the released base teacher from the Hugging Face Hub
# if it is absent.
#
#   - to use your own weights: drop a checkpoint_teacher.pth at the repo root
#   - to fetch a different one: set CKPT_REPO / CKPT_FILE before calling the script
#       CKPT_FILE=small/checkpoint_teacher_ep0300.pth bash scripts/run_infer_rats.sh
#
# The caller must set REPO to the repo root.
CKPT="${REPO}/checkpoint_teacher.pth"
CKPT_REPO="${CKPT_REPO:-TimingYang/RATS}"
CKPT_FILE="${CKPT_FILE:-base/checkpoint_teacher_ep0080.pth}"

if [ ! -e "${CKPT}" ]; then
    echo "[ckpt] no ${CKPT}; fetching ${CKPT_FILE} from ${CKPT_REPO} ..."
    cached="$(python -c "from huggingface_hub import hf_hub_download as d; print(d('${CKPT_REPO}','${CKPT_FILE}'))")" || {
        echo "[ckpt] download failed. Put a checkpoint_teacher.pth at the repo root," >&2
        echo "       or set CKPT_REPO/CKPT_FILE (and HF_TOKEN for a private repo)." >&2
        exit 1
    }
    ln -sf "${cached}" "${CKPT}"
    echo "[ckpt] linked ${CKPT} -> ${cached}"
fi
