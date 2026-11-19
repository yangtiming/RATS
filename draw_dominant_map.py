"""
Draw Dominant Map from DINO Vision Transformer (Last Layer)

Input:  image + model checkpoint
Output: dominant map image (each patch colored by its dominant register/head)

================================================================================
Arguments
================================================================================

[Model args - must match training config]
  --model_type          Model type
                        choices: dino, dino_attn_base, dino_attn_base_share,
                                 dino_attn_base_slot, dino_slot, dino_slot_ori
  --arch                Architecture name, e.g. region_vit_small, vit_small, slot_vit_small
  --patch_size          Patch size, typically 8 or 16
  --num_register_tokens Number of register tokens, e.g. 6, 12, 16, 32, 96, 120
  --use_cross_attn      Whether to use cross attention (flag, default False)
  --pretrained_weights  Checkpoint path, e.g. .../checkpoint.pth
  --checkpoint_key      Key in checkpoint dict, typically 'teacher'
  --num_reg_for_loss_per_head  (optional) Number of registers per head used in loss
  --slot_attn_iterations       (optional) Slot attention iterations, default 3

[Input / Output]
  --image_path          Input image path (required)
  --image_size          Input resolution, e.g. 512 512 or 384 384
  --output_path         Output image save path, default ./dominant_map.png
  --figsize             Figure size, default 10 10
  --dpi                 Output dpi, default 150

[Dominant Map core args]
  --dominant_mode       Dominant mode:
                          head_level   -> color each patch by dominant HEAD (default, recommended for paper)
                          all_per_head -> color each patch by dominant REGISTER
  --direction           Attention direction:
                          N2L -> register attends to patches (register->patch, default)
                          L2N -> patch attends to registers (patch->register)
  --metric              Score computation method:
                          attention   -> use attention weights (default)
                          similarity  -> use cosine similarity
  --top_n               Only use top N registers, 0=all (default 0)
                        e.g. --top_n 6 keeps only the 6 most salient registers
  --select_method       Ranking criterion for top_n:
                          variance -> highest variance (most discriminative, default)
                          max      -> highest peak activation
  --temperature         Attention softmax temperature, <1.0 = sharper, default 1.0

[Similarity normalization - only effective with --metric similarity]
  --sim_mode            Per-register similarity normalization:
                          raw       -> no normalization, raw cosine similarity (default)
                          sharpness -> percentile clipping + power transform (with --sharpness)
                          softmax   -> softmax normalization (with --sim_temperature)
                          topk      -> keep only top 10% values, zero out the rest
                          threshold -> keep values above mean+std, zero out the rest
  --sharpness           Power exponent for sim_mode=sharpness, >1 = sharper, default 1.0 (recommended 2.0)
  --sim_temperature     Temperature for sim_mode=softmax, smaller = sharper, default 0.07

[Spatial smoothing - for more coherent dominant regions]
  --dominant_norm       Smoothing/normalization method before argmax:
                          none             -> no smoothing (default)
                          spatial_smooth   -> Gaussian smoothing (most common, reduces fragmentation)
                          spatial_median   -> median filtering (denoising while preserving edges)
                          spatial_max      -> max pooling (expands high-activation regions)
                          spatial_bilateral-> bilateral filtering (smooth interior, sharp boundaries)
                          spatial_dilate   -> dilation + Gaussian (expand then smooth, fills holes)
                          local_contrast   -> local contrast normalization (removes amplitude differences)
                          cross_softmax    -> cross-register softmax competition (non-spatial)
                          cross_rank       -> rank normalization (preserves relative ordering only)
  --smooth_sigma        Spatial filter strength, larger = smoother, default 1.0 (recommended 1.0~2.0)
  --dominant_temp       Temperature for cross_softmax, smaller = sharper competition, default 0.1

[Overlay]
  --overlay             Overlay dominant map on original image (default on)
  --no_overlay          Disable overlay, pure color blocks
  --overlay_alpha       Overlay transparency, 0~1, default 0.5

================================================================================
Example commands
================================================================================

cd /path/to/RATS

# 1) Register-level, top 6, Gaussian smoothing (most common) - N96+cross model
python draw_dominant_map.py \
    --model_type dino_attn_base \
    --arch region_vit_small \
    --patch_size 16 \
    --num_register_tokens 96 \
    --use_cross_attn \
    --image_path /path/to/image.jpg \
    --pretrained_weights /path/to/checkpoint.pth \
    --checkpoint_key teacher \
    --image_size 512 512 \
    --output_path ./dominant_reg_top6_smooth.png \
    --dominant_mode all_per_head \
    --top_n 6 \
    --dominant_norm spatial_smooth \
    --smooth_sigma 1.5

# 2) Head-level dominant map (for paper, shows head spatial specialization)
python draw_dominant_map.py \
    --model_type dino_attn_base \
    --arch region_vit_small \
    --patch_size 16 \
    --num_register_tokens 96 \
    --use_cross_attn \
    --image_path /path/to/image.jpg \
    --pretrained_weights /path/to/checkpoint.pth \
    --checkpoint_key teacher \
    --image_size 512 512 \
    --output_path ./dominant_head.png \
    --dominant_mode head_level \
    --dominant_norm spatial_smooth \
    --smooth_sigma 1.0

# 3) Pure color blocks without overlay
python draw_dominant_map.py \
    --model_type dino_attn_base \
    --arch region_vit_small \
    --patch_size 16 \
    --num_register_tokens 96 \
    --use_cross_attn \
    --image_path /path/to/image.jpg \
    --pretrained_weights /path/to/checkpoint.pth \
    --checkpoint_key teacher \
    --image_size 512 512 \
    --no_overlay \
    --output_path ./dominant_pure.png

# 4) Use cosine similarity instead of attention
python draw_dominant_map.py \
    --model_type dino_attn_base \
    --arch region_vit_small \
    --patch_size 16 \
    --num_register_tokens 96 \
    --use_cross_attn \
    --image_path /path/to/image.jpg \
    --pretrained_weights /path/to/checkpoint.pth \
    --checkpoint_key teacher \
    --image_size 512 512 \
    --metric similarity \
    --output_path ./dominant_sim.png

# 5) N6 model without cross_attn
python draw_dominant_map.py \
    --model_type dino_attn_base \
    --arch region_vit_small \
    --patch_size 16 \
    --num_register_tokens 6 \
    --image_path /path/to/image.jpg \
    --pretrained_weights /path/to/checkpoint.pth \
    --checkpoint_key teacher \
    --image_size 512 512 \
    --output_path ./dominant_N6.png \
    --dominant_mode all_per_head \
    --top_n 6 \
    --dominant_norm spatial_smooth \
    --smooth_sigma 1.5

# 6) N12 model (12 heads)
python draw_dominant_map.py \
    --model_type dino_attn_base \
    --arch region_vit_small \
    --patch_size 16 \
    --num_register_tokens 12 \
    --image_path /path/to/image.jpg \
    --pretrained_weights /path/to/checkpoint.pth \
    --checkpoint_key teacher \
    --image_size 512 512 \
    --output_path ./dominant_N12.png \
    --dominant_mode all_per_head \
    --dominant_norm spatial_smooth \
    --smooth_sigma 1.5

# 7) Original DINO model N32 (without cross_attn)
python draw_dominant_map.py \
    --model_type dino \
    --arch region_vit_small \
    --patch_size 16 \
    --num_register_tokens 32 \
    --image_path /path/to/image.jpg \
    --pretrained_weights /path/to/checkpoint.pth \
    --checkpoint_key teacher \
    --image_size 512 512 \
    --output_path ./dominant_dino_N32.png \
    --dominant_mode all_per_head \
    --dominant_norm spatial_smooth \
    --smooth_sigma 1.5
"""

import os
import sys
import argparse
import colorsys
from collections import Counter, OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as pth_transforms
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# Model Loading (from extract_layer_features.py)
# ============================================================

MODEL_DIRS = {
    'dino': 'dino',
    'dino_attn_base': 'dino_attn_base',
    'dino_attn_base_share': 'dino_attn_base_share',
    'dino_attn_base_slot': 'dino_attn_base_slot',
    'dino_slot': 'dino_slot',
    'dino_slot_ori': 'dino_slot_ori',
}


def get_model_module(model_type, model_dir=None):
    """Dynamically import vision_transformer module based on model type.

    Args:
        model_type: model type string (e.g. 'dino_attn_base')
        model_dir: (optional) explicit path to directory containing vision_transformer.py
                   If provided, overrides the default the model directory next to this file.
    """
    if model_dir:
        model_path = model_dir
    else:
        # Fallback when --model_dir is not given: look next to this file.
        REGION_DINOV1_BASE = os.path.dirname(os.path.abspath(__file__))
        model_dir_name = MODEL_DIRS.get(model_type, 'dino_attn_base')
        model_path = os.path.join(REGION_DINOV1_BASE, model_dir_name)

    print(f"[INFO] Model path: {model_path}")

    if model_path not in sys.path:
        sys.path.insert(0, model_path)

    if model_type == 'dino_slot_ori':
        slot_attn_path = os.path.join(model_path, 'slot-attention')
        if slot_attn_path not in sys.path:
            sys.path.insert(0, slot_attn_path)

    import importlib
    for mod_name in list(sys.modules.keys()):
        if mod_name in ['vision_transformer', 'utils', 'slot_attention'] or \
           mod_name.startswith('vision_transformer.') or mod_name.startswith('slot_attention.'):
            del sys.modules[mod_name]

    vits = importlib.import_module('vision_transformer')
    return vits


def load_model(args):
    """Load pretrained model based on model type."""
    model_dir = getattr(args, 'model_dir', None)
    vits = get_model_module(args.model_type, model_dir=model_dir)

    model_kwargs = {'patch_size': args.patch_size, 'num_classes': 0}

    if args.num_register_tokens > 0 and ('region' in args.arch or 'slot' in args.arch):
        model_kwargs['num_register_tokens'] = args.num_register_tokens
        if args.model_type in ['dino_attn_base', 'dino_attn_base_share', 'dino_attn_base_slot']:
            model_kwargs['use_cross_attn'] = args.use_cross_attn
            if args.num_reg_for_loss_per_head is not None:
                model_kwargs['num_reg_for_loss_per_head'] = args.num_reg_for_loss_per_head
        if args.model_type in ['dino_slot', 'dino_slot_ori', 'dino_attn_base_slot']:
            model_kwargs['slot_attn_iterations'] = args.slot_attn_iterations

    if hasattr(vits, args.arch):
        model = vits.__dict__[args.arch](**model_kwargs)
    else:
        available = [k for k in dir(vits) if 'vit' in k.lower()]
        print(f"Architecture {args.arch} not found. Available: {available}")
        return None, None

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.pretrained_weights and os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu", weights_only=False)
        if args.checkpoint_key and args.checkpoint_key in state_dict:
            print(f"Using key '{args.checkpoint_key}' from checkpoint")
            state_dict = state_dict[args.checkpoint_key]
        state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {args.pretrained_weights}")
        if msg.missing_keys:
            print(f"  Missing keys: {msg.missing_keys[:5]}..." if len(msg.missing_keys) > 5 else f"  Missing keys: {msg.missing_keys}")
        if msg.unexpected_keys:
            print(f"  Unexpected keys: {msg.unexpected_keys[:5]}..." if len(msg.unexpected_keys) > 5 else f"  Unexpected keys: {msg.unexpected_keys}")
    else:
        print("No pretrained weights provided, using random initialization.")

    return model, device


def load_image(image_path, image_size, patch_size):
    """Load and preprocess image."""
    img = Image.open(image_path).convert('RGB')
    transform = pth_transforms.Compose([
        pth_transforms.Resize(image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img_tensor = transform(img)
    _, h, w = img_tensor.shape
    h = h - h % patch_size
    w = w - w % patch_size
    img_tensor = img_tensor[:, :h, :w].unsqueeze(0)
    return img_tensor, img, (h, w)


# ============================================================
# Feature Extraction (from extract_layer_features.py)
# ============================================================

class LayerFeatureExtractor:
    """Hook-based feature extractor for extracting layer-wise data from ViT."""

    def __init__(self, model, model_type='dino', temperature=1.0):
        self.model = model
        self.model_type = model_type
        self.temperature = temperature
        self.layer_data = OrderedDict()
        self._hooks = []
        self._block_inputs = {}
        self._qkv_data = {}
        self._attn_data = {}
        self._slot_data = {}

    def _get_block_name(self, idx):
        return f'block_{idx}'

    def _register_hooks(self):
        for idx, block in enumerate(self.model.blocks):
            block_name = self._get_block_name(idx)
            self._hooks.append(
                block.register_forward_pre_hook(self._make_block_input_hook(block_name))
            )
            self._hooks.append(
                block.register_forward_hook(self._make_block_output_hook(block_name))
            )
            if hasattr(block, 'attn') and hasattr(block.attn, 'qkv'):
                self._hooks.append(
                    block.attn.qkv.register_forward_hook(self._make_qkv_hook(block_name))
                )
                self._attn_data[block_name] = {
                    'module': block.attn,
                    'num_heads': block.attn.num_heads,
                    'head_dim': block.attn.head_dim,
                    'num_register_tokens': getattr(block.attn, 'num_register_tokens', 0),
                }
            if self.model_type in ['dino_slot', 'dino_slot_ori', 'dino_attn_base_slot']:
                if hasattr(block, 'attn'):
                    self._patch_attention_for_capture(block_name, block.attn)

    def _make_block_input_hook(self, block_name):
        def hook(module, input):
            x = input[0] if isinstance(input, tuple) else input
            self._block_inputs[block_name] = x.detach().clone()
        return hook

    def _make_block_output_hook(self, block_name):
        def hook(module, input, output):
            if block_name not in self.layer_data:
                self.layer_data[block_name] = {}
            self.layer_data[block_name]['input_tokens'] = self._block_inputs.get(block_name)
            if isinstance(output, tuple):
                self.layer_data[block_name]['output_tokens'] = output[0].detach().clone()
            else:
                self.layer_data[block_name]['output_tokens'] = output.detach().clone()
        return hook

    def _make_qkv_hook(self, block_name):
        def hook(module, input, output):
            if block_name not in self.layer_data:
                self.layer_data[block_name] = {}

            qkv_raw = output.detach().clone()
            attn_info = self._attn_data.get(block_name, {})
            num_heads = attn_info.get('num_heads', 6)
            head_dim = attn_info.get('head_dim', 64)
            num_register_tokens = attn_info.get('num_register_tokens', 0)

            B, N_total, _ = qkv_raw.shape
            qkv = qkv_raw.reshape(B, N_total, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            self.layer_data[block_name]['q'] = q
            self.layer_data[block_name]['k'] = k
            self.layer_data[block_name]['v'] = v

            scale = head_dim ** -0.5
            temperature = self.temperature

            if self.model_type in ['dino_attn_base', 'dino_attn_base_share'] and num_register_tokens > 0:
                N_reg = num_register_tokens
                N_patch = N_total - N_reg
                regs_per_head = N_reg // num_heads

                q_reg = q[:, :, :N_reg, :]
                k_reg = k[:, :, :N_reg, :]
                q_patch = q[:, :, N_reg:, :]
                k_patch = k[:, :, N_reg:, :]

                q_reg_reshaped = q_reg.reshape(B, num_heads, num_heads, regs_per_head, head_dim)
                k_reg_reshaped = k_reg.reshape(B, num_heads, num_heads, regs_per_head, head_dim)

                head_indices = torch.arange(num_heads, device=q.device)
                q_reg_diag = q_reg_reshaped[:, head_indices, head_indices, :, :]
                k_reg_diag = k_reg_reshaped[:, head_indices, head_indices, :, :]

                attn_weights = torch.zeros(B, num_heads, N_total, N_total, device=q.device, dtype=q.dtype)

                attn_L_to_N_logits = (q_reg_diag @ k_patch.transpose(-2, -1)) * scale
                attn_L_to_N = F.softmax(attn_L_to_N_logits / temperature, dim=-1)

                attn_N_to_L_logits = (q_patch @ k_reg_diag.transpose(-2, -1)) * scale
                attn_N_to_L = F.softmax(attn_N_to_L_logits / temperature, dim=-1)

                for h in range(num_heads):
                    reg_start = h * regs_per_head
                    reg_end = (h + 1) * regs_per_head
                    attn_weights[:, h, reg_start:reg_end, N_reg:] = attn_L_to_N[:, h, :, :]
                    attn_weights[:, h, N_reg:, reg_start:reg_end] = attn_N_to_L[:, h, :, :]

                attn_logits = attn_L_to_N_logits

            elif self.model_type == 'dino_attn_base_slot' and num_register_tokens > 0:
                N_reg = num_register_tokens
                N_patch = N_total - N_reg
                regs_per_head = N_reg // num_heads

                q_reg = q[:, :, :N_reg, :]
                k_reg = k[:, :, :N_reg, :]
                q_patch = q[:, :, N_reg:, :]
                k_patch = k[:, :, N_reg:, :]

                q_reg_reshaped = q_reg.reshape(B, num_heads, num_heads, regs_per_head, head_dim)
                k_reg_reshaped = k_reg.reshape(B, num_heads, num_heads, regs_per_head, head_dim)

                head_indices = torch.arange(num_heads, device=q.device)
                q_reg_diag = q_reg_reshaped[:, head_indices, head_indices, :, :]
                k_reg_diag = k_reg_reshaped[:, head_indices, head_indices, :, :]

                attn_weights = torch.zeros(B, num_heads, N_total, N_total, device=q.device, dtype=q.dtype)

                attn_N_to_L_logits = (q_patch @ k_reg_diag.transpose(-2, -1)) * scale
                attn_N_to_L = F.softmax(attn_N_to_L_logits / temperature, dim=-1)

                attn_L_to_N_logits = (q_reg_diag @ k_patch.transpose(-2, -1)) * scale
                attn_L_to_N = F.softmax(attn_L_to_N_logits / temperature, dim=2)
                attn_L_to_N = attn_L_to_N / (attn_L_to_N.sum(dim=-1, keepdim=True) + 1e-8)

                for h in range(num_heads):
                    reg_start = h * regs_per_head
                    reg_end = (h + 1) * regs_per_head
                    attn_weights[:, h, reg_start:reg_end, N_reg:] = attn_L_to_N[:, h, :, :]
                    attn_weights[:, h, N_reg:, reg_start:reg_end] = attn_N_to_L[:, h, :, :]

                attn_logits = torch.zeros_like(attn_weights)

            else:
                attn_logits = (q @ k.transpose(-2, -1)) * scale
                attn_weights = F.softmax(attn_logits / temperature, dim=-1)

            self.layer_data[block_name]['attn_logits'] = attn_logits
            self.layer_data[block_name]['attn_weights'] = attn_weights

        return hook

    def _patch_attention_for_capture(self, block_name, attn_module):
        if not hasattr(attn_module, 'use_region_attn') or not attn_module.use_region_attn:
            return

        original_forward = attn_module.forward
        extractor = self
        model_type = self.model_type

        def patched_forward(x):
            result = original_forward(x)
            if block_name not in extractor.layer_data:
                extractor.layer_data[block_name] = {}

            if model_type == 'dino_attn_base_slot':
                if hasattr(attn_module, '_attn_L_to_N') and hasattr(attn_module, '_attn_N_to_L'):
                    attn_L_to_N = attn_module._attn_L_to_N
                    attn_N_to_L = attn_module._attn_N_to_L

                    B, num_heads, regs_per_head, L = attn_L_to_N.shape
                    N_reg = attn_module.num_register_tokens
                    N_total = N_reg + L

                    attn_weights = torch.zeros(B, num_heads, N_total, N_total,
                                               device=attn_L_to_N.device, dtype=attn_L_to_N.dtype)
                    for h in range(num_heads):
                        reg_start = h * regs_per_head
                        reg_end = (h + 1) * regs_per_head
                        attn_weights[:, h, reg_start:reg_end, N_reg:] = attn_L_to_N[:, h, :, :]
                        attn_weights[:, h, N_reg:, reg_start:reg_end] = attn_N_to_L[:, h, :, :]

                    extractor.layer_data[block_name]['attn_weights'] = attn_weights

            if hasattr(attn_module, '_captured_slot_data'):
                extractor.layer_data[block_name]['slot_data'] = attn_module._captured_slot_data

            return result

        attn_module.forward = patched_forward

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def extract(self, x):
        self.layer_data = OrderedDict()
        self._block_inputs = {}
        self._register_hooks()

        with torch.no_grad():
            output = self.model(x)

        self._remove_hooks()

        self.layer_data['final_output'] = {
            'cls_token': output if output.dim() == 2 else output[:, 0],
        }

        last_block = list(self.layer_data.keys())[-2]
        if last_block in self.layer_data and 'output_tokens' in self.layer_data[last_block]:
            last_output = self.layer_data[last_block]['output_tokens']
            num_reg = getattr(self.model, 'num_register_tokens', 0)
            if num_reg > 0:
                self.layer_data['final_output']['register_tokens'] = last_output[:, :num_reg]
                self.layer_data['final_output']['patch_tokens'] = last_output[:, num_reg:]
            else:
                self.layer_data['final_output']['cls_token'] = last_output[:, 0]
                self.layer_data['final_output']['patch_tokens'] = last_output[:, 1:]

        return self.layer_data


def extract_register_and_patch_features(layer_data, num_register_tokens=0):
    """Extract register and patch features from layer data."""
    results = {
        'register_features': [],
        'patch_features': [],
        'attention_weights': [],
    }

    for block_name, data in layer_data.items():
        if block_name == 'final_output':
            continue
        if 'output_tokens' in data:
            tokens = data['output_tokens']
            if num_register_tokens > 0:
                results['register_features'].append(tokens[:, :num_register_tokens])
                results['patch_features'].append(tokens[:, num_register_tokens:])
            else:
                results['register_features'].append(tokens[:, :1])
                results['patch_features'].append(tokens[:, 1:])
        if 'attn_weights' in data:
            results['attention_weights'].append(data['attn_weights'])

    for key in results:
        if results[key]:
            results[key] = torch.stack(results[key], dim=0)

    return results


# ============================================================
# Dominant Map Visualization (from feature_visualizer_server.py)
# ============================================================

def get_distinct_colors(n):
    """Generate n visually distinct colors."""
    base_colors = [
        (1.0, 0.0, 0.0), (0.0, 0.8, 0.0), (0.0, 0.0, 1.0),
        (1.0, 0.8, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0),
        (1.0, 0.5, 0.0), (0.5, 0.0, 1.0), (0.0, 0.5, 0.0),
        (0.5, 0.5, 0.0), (0.0, 0.5, 1.0), (1.0, 0.0, 0.5),
    ]
    if n <= len(base_colors):
        return base_colors[:n]
    colors = list(base_colors)
    for i in range(len(base_colors), n):
        hue = (i * 0.618033988749895) % 1.0
        colors.append(colorsys.hsv_to_rgb(hue, 0.9, 0.9))
    return colors


def get_owning_head(reg_idx, num_heads, num_reg):
    """Get the head that owns a specific register."""
    regs_per_head = num_reg // num_heads
    return reg_idx // regs_per_head


def get_all_scores_per_head(summary, layer_idx, num_reg, num_heads, direction='N2L', metric='attention'):
    """
    Get scores for all registers where each register uses its own head's attention.

    Returns:
        all_scores: [num_reg, L]
    """
    regs_per_head = num_reg // num_heads

    if metric == 'similarity':
        reg_feat = summary['register_features'][layer_idx, 0]  # [num_reg, C]
        patch_feat = summary['patch_features'][layer_idx, 0]  # [L, C]
        reg_norm = F.normalize(reg_feat.float(), dim=-1)
        patch_norm = F.normalize(patch_feat.float(), dim=-1)
        all_scores = (reg_norm @ patch_norm.T).numpy()  # [num_reg, L]
    else:
        attn = summary['attention_weights'][layer_idx, 0]  # [heads, N, N]
        num_patches = attn.shape[-1] - num_reg

        all_scores = np.zeros((num_reg, num_patches), dtype=np.float32)
        for reg_idx in range(num_reg):
            owning_head = reg_idx // regs_per_head
            attn_head = attn[owning_head]

            if direction == 'L2N':
                all_scores[reg_idx] = np.asarray(attn_head[num_reg:, reg_idx], dtype=np.float32)
            else:
                all_scores[reg_idx] = np.asarray(attn_head[reg_idx, num_reg:], dtype=np.float32)

    return all_scores


def get_head_level_scores(summary, layer_idx, num_reg, num_heads, direction='N2L', metric='attention'):
    """
    Get HEAD-level scores by averaging registers within each head.

    Returns:
        head_scores: [num_heads, L]
    """
    regs_per_head = num_reg // num_heads

    attn = summary['attention_weights'][layer_idx, 0]  # [heads, N, N]
    num_patches = attn.shape[-1] - num_reg

    if metric == 'similarity':
        reg_feat = summary['register_features'][layer_idx, 0]
        patch_feat = summary['patch_features'][layer_idx, 0]
        reg_norm = F.normalize(reg_feat.float(), dim=-1)
        patch_norm = F.normalize(patch_feat.float(), dim=-1)
        full_scores = (reg_norm @ patch_norm.T).numpy()

        head_scores = np.zeros((num_heads, num_patches), dtype=np.float32)
        for h in range(num_heads):
            start = h * regs_per_head
            end = (h + 1) * regs_per_head
            head_scores[h] = full_scores[start:end, :].mean(axis=0)
    else:
        head_scores = np.zeros((num_heads, num_patches), dtype=np.float32)
        for h in range(num_heads):
            attn_head = attn[h]
            start = h * regs_per_head
            end = (h + 1) * regs_per_head

            if direction == 'L2N':
                reg_scores = np.asarray(attn_head[num_reg:, start:end], dtype=np.float32)
            else:
                reg_scores = np.asarray(attn_head[start:end, num_reg:], dtype=np.float32)
                reg_scores = reg_scores.T

            head_scores[h] = reg_scores.mean(axis=1)

    return head_scores


def normalize_similarity(scores, sharpness=2.0, mode='sharpness', temperature=0.07, top_k_percent=10):
    """
    Normalize similarity scores per register (from server).

    Args:
        scores: [L] similarity scores for one register
        sharpness: power transform exponent (>1 = sharper contrast)
        mode: 'raw', 'sharpness', 'softmax', 'topk', 'threshold'
        temperature: softmax temperature (lower = sharper)
        top_k_percent: for topk mode

    Returns:
        normalized scores in [0, 1]
    """
    scores = np.asarray(scores, dtype=np.float32)

    if mode == 'softmax':
        scores_scaled = scores / temperature
        scores_exp = np.exp(scores_scaled - scores_scaled.max())
        scores = scores_exp / (scores_exp.sum() + 1e-8)
        scores = scores / (scores.max() + 1e-8)
        return scores

    if mode == 'topk':
        threshold = np.percentile(scores, 100 - top_k_percent)
        scores = np.where(scores >= threshold, scores, 0)
        s_max = scores.max()
        if s_max > 1e-8:
            scores = scores / s_max
        return scores

    if mode == 'threshold':
        mean_val = scores.mean()
        std_val = scores.std()
        threshold = mean_val + std_val
        scores = np.where(scores >= threshold, scores, 0)
        s_max = scores.max()
        if s_max > 1e-8:
            scores = scores / s_max
        return scores

    # Default: percentile clipping + power transform (sharpness mode)
    percentile_clip = 5
    if percentile_clip > 0:
        low = np.percentile(scores, percentile_clip)
        high = np.percentile(scores, 100 - percentile_clip)
        scores = np.clip(scores, low, high)

    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min > 1e-8:
        scores = (scores - s_min) / (s_max - s_min)
    else:
        scores = np.zeros_like(scores)

    if mode == 'sharpness' and sharpness != 1.0 and sharpness > 0:
        scores = np.power(scores, sharpness)

    return scores


def normalize_scores_for_dominant(all_scores, method='none', temperature=0.1, spatial_sigma=1.0):
    """
    Normalize scores for dominant map visualization.

    Args:
        all_scores: [num_regs, L] - raw scores for each register
        method:
            'none'              - no normalization
            'cross_softmax'     - softmax across registers at each patch (sharper competition)
            'cross_rank'        - rank-based normalization across registers
            'spatial_smooth'    - Gaussian smoothing per register (coherent regions)
            'spatial_median'    - median filter per register (edge-preserving)
            'spatial_max'       - max pooling per register (expand high regions)
            'spatial_bilateral' - bilateral-like filtering (smooth but preserve edges)
            'spatial_dilate'    - morphological dilation + Gaussian (expand then smooth)
            'local_contrast'    - local contrast normalization (normalize by local mean/std)
        temperature: for cross_softmax, lower = sharper competition
        spatial_sigma: sigma/size for spatial filters

    Returns:
        normalized_scores: [num_regs, L]
    """
    from scipy.ndimage import gaussian_filter, median_filter, maximum_filter, uniform_filter

    all_scores = np.asarray(all_scores, dtype=np.float32)
    num_regs, L = all_scores.shape

    h = int(np.sqrt(L))
    w = L // h
    is_square = (h * w == L)

    if method == 'cross_softmax':
        # Softmax ACROSS registers at each patch location
        # Makes competition sharper - winner takes more
        scores_scaled = all_scores / temperature
        scores_scaled = scores_scaled - scores_scaled.max(axis=0, keepdims=True)
        scores_exp = np.exp(scores_scaled)
        all_scores = scores_exp / (scores_exp.sum(axis=0, keepdims=True) + 1e-8)

    elif method == 'cross_rank':
        # Convert to ranks across registers (removes magnitude, keeps ordering)
        ranks = np.zeros_like(all_scores)
        for j in range(L):
            order = np.argsort(all_scores[:, j])
            ranks[order, j] = np.arange(num_regs) / (num_regs - 1)
        all_scores = ranks

    elif method == 'spatial_smooth' and is_square:
        # Gaussian smoothing per register -> coherent spatial regions
        for i in range(num_regs):
            score_2d = all_scores[i].reshape(h, w)
            score_2d = gaussian_filter(score_2d, sigma=spatial_sigma)
            all_scores[i] = score_2d.flatten()

    elif method == 'spatial_median' and is_square:
        # Median filter - better edge preservation than Gaussian
        kernel_size = max(3, int(spatial_sigma * 2 + 1))
        if kernel_size % 2 == 0:
            kernel_size += 1
        for i in range(num_regs):
            score_2d = all_scores[i].reshape(h, w)
            score_2d = median_filter(score_2d, size=kernel_size)
            all_scores[i] = score_2d.flatten()

    elif method == 'spatial_max' and is_square:
        # Max pooling - expands high-activation regions
        kernel_size = max(3, int(spatial_sigma * 2 + 1))
        if kernel_size % 2 == 0:
            kernel_size += 1
        for i in range(num_regs):
            score_2d = all_scores[i].reshape(h, w)
            score_2d = maximum_filter(score_2d, size=kernel_size)
            all_scores[i] = score_2d.flatten()

    elif method == 'spatial_bilateral' and is_square:
        # Bilateral-like: smooth but preserve edges
        for i in range(num_regs):
            score_2d = all_scores[i].reshape(h, w)
            smoothed = gaussian_filter(score_2d, sigma=spatial_sigma)
            diff = np.abs(score_2d - smoothed)
            edge_weight = np.exp(-diff * 10 / (diff.max() + 1e-8))
            score_2d = score_2d * (1 - edge_weight) + smoothed * edge_weight
            all_scores[i] = score_2d.flatten()

    elif method == 'spatial_dilate' and is_square:
        # Morphological dilation + Gaussian -> expand regions smoothly
        for i in range(num_regs):
            score_2d = all_scores[i].reshape(h, w)
            kernel_size = max(3, int(spatial_sigma + 1))
            dilated = maximum_filter(score_2d, size=kernel_size)
            score_2d = gaussian_filter(dilated, sigma=spatial_sigma / 2)
            all_scores[i] = score_2d.flatten()

    elif method == 'local_contrast' and is_square:
        # Local contrast normalization (normalize by local mean/std)
        for i in range(num_regs):
            score_2d = all_scores[i].reshape(h, w)
            k = int(spatial_sigma * 2 + 1)
            local_mean = uniform_filter(score_2d, size=k)
            local_std = np.sqrt(uniform_filter(score_2d**2, size=k) - local_mean**2 + 1e-8)
            score_2d = (score_2d - local_mean) / (local_std + 1e-8)
            score_2d = gaussian_filter(score_2d, sigma=spatial_sigma / 2)
            all_scores[i] = score_2d.flatten()

    return all_scores


# ============================================================
# Isolated-patch cleanup (connected-component post-processing).
# Ported from save_single_panels.py — keeps the demo's dominant
# map consistent with the eval/visualize pipeline (CLEAN_FRAC).
# ============================================================

def _connected_components(mask, connectivity=8):
    """Flood fill on a bool mask. connectivity: 4 (N/S/E/W) or 8 (also diagonals).
    Returns (label_map, n)."""
    H, W = mask.shape
    label_map = np.zeros_like(mask, dtype=np.int32)
    n = 0
    if connectivity == 8:
        offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    else:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(H):
        for j in range(W):
            if mask[i, j] and label_map[i, j] == 0:
                n += 1
                stack = [(i, j)]
                while stack:
                    y, x = stack.pop()
                    if not (0 <= y < H and 0 <= x < W):
                        continue
                    if not mask[y, x] or label_map[y, x] != 0:
                        continue
                    label_map[y, x] = n
                    for dy, dx in offsets:
                        stack.append((y + dy, x + dx))
    return label_map, n


def clean_isolated_patches(dominant_map, min_patches=3, min_patches_frac=0.0, max_iters=16):
    """Drop small connected components (< effective_min patches) by reassigning
    their patches to the mode of their 8-neighbours' labels.

    effective_min = max(min_patches, ceil(min_patches_frac * total_patches))
      — e.g. min_patches_frac=0.005 at 1024x1024/patch16 → 64*64*0.005 ≈ 21.

    8-connectivity for CC detection. Two-pass reassignment per outer iteration:
      pass 1 uses non-small neighbours (anchors) only;
      pass 2 fallback uses ALL neighbours (breaks deadlock in dense noise regions).

    `effective_min <= 1` disables cleanup.
    """
    H, W = dominant_map.shape
    total = H * W
    effective_min = max(int(min_patches), int(np.ceil(float(min_patches_frac) * total)))
    if effective_min <= 1:
        return dominant_map.copy()

    out = dominant_map.copy()
    for _ in range(max_iters):
        small_mask = np.zeros_like(out, dtype=bool)
        for lbl in np.unique(out):
            if lbl < 0:
                continue
            cc_map, ncc = _connected_components(out == lbl, connectivity=8)
            for k in range(1, ncc + 1):
                comp = (cc_map == k)
                if comp.sum() < effective_min:
                    small_mask |= comp

        if not small_mask.any():
            break

        changed_any = False
        ys, xs = np.where(small_mask)
        coords = list(zip(ys.tolist(), xs.tolist()))

        any_changed_this_pass = True
        while any_changed_this_pass:
            any_changed_this_pass = False
            remaining = []
            for (y, x) in coords:
                if not small_mask[y, x]:
                    continue
                # Pass 1: prefer non-small neighbours (real anchor labels).
                counts_anchor = Counter()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            v = out[ny, nx]
                            if v < 0:
                                continue
                            if not small_mask[ny, nx]:
                                counts_anchor[int(v)] += 1
                if counts_anchor:
                    out[y, x] = counts_anchor.most_common(1)[0][0]
                    small_mask[y, x] = False   # newly anchored
                    changed_any = True
                    any_changed_this_pass = True
                else:
                    remaining.append((y, x))
            coords = remaining
            if not coords:
                break
            # Pass 2 fallback: reassign using mode of ALL 8-neighbours.
            made_progress = False
            for (y, x) in coords[:]:
                counts = Counter()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            v = out[ny, nx]
                            if v >= 0:
                                counts[int(v)] += 1
                if counts:
                    new_lbl = counts.most_common(1)[0][0]
                    if new_lbl != out[y, x]:
                        out[y, x] = new_lbl
                        made_progress = True
                        changed_any = True
                        any_changed_this_pass = True
            if not made_progress:
                break

        if not changed_any:
            break
    return out


def draw_dominant_map(summary, metadata, image_path, output_path,
                      dominant_mode='all_per_head', direction='N2L', metric='attention',
                      overlay=True, overlay_alpha=0.5, top_n=0, select_method='variance',
                      dominant_norm='none', smooth_sigma=1.0, dominant_temp=0.1,
                      sim_mode='raw', sharpness=1.0, sim_temperature=0.07,
                      clean_min_patches=0, clean_min_patches_frac=0.0,
                      figsize=(10, 10), dpi=150):
    """
    Draw and save dominant map.

    Args:
        summary: dict with register_features, patch_features, attention_weights
        metadata: dict with model info
        image_path: path to original image
        output_path: where to save the dominant map image
        dominant_mode: 'all_per_head' (dominant register) or 'head_level' (dominant head)
        direction: 'N2L' or 'L2N'
        metric: 'attention' or 'similarity'
        overlay: whether to overlay on original image
        overlay_alpha: blend ratio for overlay
        top_n: if > 0, only use top N registers by variance
        dominant_norm: 'none', 'cross_softmax', 'spatial_smooth'
        smooth_sigma: Gaussian sigma for spatial_smooth (higher = smoother, more coherent)
        dominant_temp: temperature for cross_softmax (lower = sharper competition)
        figsize: figure size
        dpi: output dpi
    """
    h_feat, w_feat = metadata['feature_size']
    num_reg = metadata['num_register_tokens']
    num_heads = metadata['num_heads']
    patch_size = metadata['patch_size']
    regs_per_head = num_reg // num_heads
    num_layers = metadata['num_layers']

    # Use last layer
    layer_idx = num_layers - 1

    if dominant_mode == 'head_level':
        # Find dominant HEAD per patch
        head_scores = get_head_level_scores(
            summary, layer_idx, num_reg, num_heads, direction=direction, metric=metric
        )

        # Apply spatial smoothing before argmax
        if dominant_norm != 'none':
            head_scores = normalize_scores_for_dominant(
                head_scores, method=dominant_norm,
                temperature=dominant_temp, spatial_sigma=smooth_sigma
            )

        dominant_head = head_scores.argmax(axis=0)  # [L]
        dominant_map = dominant_head.reshape(h_feat, w_feat)

        # Connected-component cleanup: dissolve tiny isolated regions into neighbours
        if (clean_min_patches and clean_min_patches > 1) or \
                (clean_min_patches_frac and clean_min_patches_frac > 0):
            before = dominant_map.copy()
            dominant_map = clean_isolated_patches(
                dominant_map, min_patches=clean_min_patches,
                min_patches_frac=clean_min_patches_frac,
            )
            dominant_head = dominant_map.reshape(-1)
            print(f"[clean] reassigned {int((dominant_map != before).sum())} isolated patches "
                  f"(min_patches={clean_min_patches}, frac={clean_min_patches_frac})")

        colors = get_distinct_colors(num_heads)

        rgb_map = np.zeros((h_feat, w_feat, 3))
        for h in range(num_heads):
            mask = (dominant_map == h)
            for c in range(3):
                rgb_map[:, :, c] += mask * colors[h][c]

        # Upscale (pure mask)
        rgb_mask = np.repeat(np.repeat(rgb_map, patch_size, axis=0), patch_size, axis=1)

        # Overlay version
        rgb_overlay = rgb_mask.copy()
        if image_path and os.path.exists(image_path):
            orig_img = Image.open(image_path).convert('RGB')
            orig_img = orig_img.resize((rgb_mask.shape[1], rgb_mask.shape[0]))
            orig_np = np.array(orig_img).astype(np.float32) / 255.0
            rgb_overlay = orig_np * (1 - overlay_alpha) + rgb_mask * overlay_alpha

        # Side by side: original (left) + overlay (middle) + mask (right)
        fig, (ax_orig, ax_overlay, ax_mask) = plt.subplots(1, 3, figsize=(figsize[0] * 3, figsize[1]))

        if image_path and os.path.exists(image_path):
            orig_img_show = Image.open(image_path).convert('RGB')
            orig_img_show = orig_img_show.resize((rgb_mask.shape[1], rgb_mask.shape[0]))
            ax_orig.imshow(orig_img_show)
        ax_orig.set_title('Original')
        ax_orig.axis('off')

        ax_overlay.imshow(rgb_overlay)
        ax_overlay.set_title('Overlay')
        ax_overlay.axis('off')

        ax_mask.imshow(rgb_mask)
        ax_mask.set_title('Mask')
        ax_mask.axis('off')

        legend_patches = [
            Patch(color=colors[h], label=f'H{h}(R{h*regs_per_head}-{(h+1)*regs_per_head-1})')
            for h in range(num_heads)
        ]
        ax_mask.legend(handles=legend_patches, loc='upper right', fontsize=8, ncol=1)

        metric_str = f'Attn({direction})' if metric == 'attention' else 'Similarity'
        fig.suptitle(f'Dominant HEAD ({metric_str}): L{layer_idx}', fontsize=14)

        # Print stats
        unique, counts = np.unique(dominant_head, return_counts=True)
        print(f"\nDominant HEAD statistics:")
        for u, c in zip(unique, counts):
            print(f"  H{int(u)}: {int(c)} patches ({100*c/len(dominant_head):.1f}%)")

    else:
        # Find dominant REGISTER per patch (all_per_head mode)
        all_scores = get_all_scores_per_head(
            summary, layer_idx, num_reg, num_heads, direction=direction, metric=metric
        )

        selected_regs = list(range(num_reg))

        # Per-register similarity normalization (sharpness/softmax/topk)
        if metric == 'similarity' and sim_mode != 'raw':
            for i in range(all_scores.shape[0]):
                all_scores[i] = normalize_similarity(
                    all_scores[i], sharpness=sharpness, mode=sim_mode,
                    temperature=sim_temperature
                )

        # Filter by top N registers
        if top_n > 0 and top_n < num_reg:
            if select_method == 'max':
                rank_scores = all_scores.max(axis=1)
            else:  # variance (default)
                rank_scores = all_scores.var(axis=1)
            top_indices = np.argsort(rank_scores)[-top_n:][::-1].tolist()
            top_indices = sorted(top_indices)
            all_scores = all_scores[top_indices]
            selected_regs = [selected_regs[i] for i in top_indices]

        # Apply spatial smoothing / cross-register normalization before argmax
        if dominant_norm != 'none':
            all_scores = normalize_scores_for_dominant(
                all_scores, method=dominant_norm,
                temperature=dominant_temp, spatial_sigma=smooth_sigma
            )

        # Find dominant register per patch
        dominant_idx = all_scores.argmax(axis=0)  # [L]
        dominant_reg = np.array([selected_regs[i] for i in dominant_idx])
        dominant_map = dominant_reg.reshape(h_feat, w_feat)

        # Connected-component cleanup: dissolve tiny isolated regions into neighbours
        if (clean_min_patches and clean_min_patches > 1) or \
                (clean_min_patches_frac and clean_min_patches_frac > 0):
            before = dominant_map.copy()
            dominant_map = clean_isolated_patches(
                dominant_map, min_patches=clean_min_patches,
                min_patches_frac=clean_min_patches_frac,
            )
            print(f"[clean] reassigned {int((dominant_map != before).sum())} isolated patches "
                  f"(min_patches={clean_min_patches}, frac={clean_min_patches_frac})")

        num_selected = len(selected_regs)
        colors = get_distinct_colors(num_selected)
        color_map = {reg: colors[i] for i, reg in enumerate(selected_regs)}

        rgb_map = np.zeros((h_feat, w_feat, 3))
        for reg_idx_val in selected_regs:
            mask = (dominant_map == reg_idx_val)
            for c in range(3):
                rgb_map[:, :, c] += mask * color_map[reg_idx_val][c]

        # Upscale (pure mask)
        rgb_mask = np.repeat(np.repeat(rgb_map, patch_size, axis=0), patch_size, axis=1)

        # Overlay version
        rgb_overlay = rgb_mask.copy()
        if image_path and os.path.exists(image_path):
            orig_img = Image.open(image_path).convert('RGB')
            orig_img = orig_img.resize((rgb_mask.shape[1], rgb_mask.shape[0]))
            orig_np = np.array(orig_img).astype(np.float32) / 255.0
            rgb_overlay = orig_np * (1 - overlay_alpha) + rgb_mask * overlay_alpha

        # Side by side: original (left) + overlay (middle) + mask (right)
        fig, (ax_orig, ax_overlay, ax_mask) = plt.subplots(1, 3, figsize=(figsize[0] * 3, figsize[1]))

        if image_path and os.path.exists(image_path):
            orig_img_show = Image.open(image_path).convert('RGB')
            orig_img_show = orig_img_show.resize((rgb_mask.shape[1], rgb_mask.shape[0]))
            ax_orig.imshow(orig_img_show)
        ax_orig.set_title('Original')
        ax_orig.axis('off')

        ax_overlay.imshow(rgb_overlay)
        ax_overlay.set_title('Overlay')
        ax_overlay.axis('off')

        ax_mask.imshow(rgb_mask)
        ax_mask.set_title('Mask')
        ax_mask.axis('off')


        # Print stats
        unique, counts = np.unique(dominant_reg, return_counts=True)
        print(f"\nDominant register statistics (top 10):")
        sorted_idx = np.argsort(counts)[::-1]
        for i in sorted_idx[:10]:
            print(f"  R{int(unique[i])}: {int(counts[i])} patches ({100*counts[i]/len(dominant_reg):.1f}%)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print(f"\nSaved dominant map to: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser('Draw Dominant Map from ViT Last Layer')

    # Model args
    parser.add_argument('--model_type', default='dino', type=str,
                        choices=['dino', 'dino_attn_base', 'dino_attn_base_share', 'dino_attn_base_slot', 'dino_slot', 'dino_slot_ori'])
    parser.add_argument('--model_dir', default=None, type=str,
                        help='Directly specify the directory containing vision_transformer.py, overrides model_type default path lookup')
    parser.add_argument('--arch', default='vit_small', type=str)
    parser.add_argument('--patch_size', default=8, type=int)
    parser.add_argument('--pretrained_weights', default='', type=str)
    parser.add_argument('--checkpoint_key', default='teacher', type=str)
    parser.add_argument('--num_register_tokens', default=0, type=int)
    parser.add_argument('--num_reg_for_loss_per_head', default=None, type=int)
    parser.add_argument('--use_cross_attn', action='store_true')
    parser.add_argument('--slot_attn_iterations', default=3, type=int)

    # Image args
    parser.add_argument('--image_path', required=True, type=str)
    parser.add_argument('--image_size', default=[512, 512], type=int, nargs='+')

    # Output args
    parser.add_argument('--output_path', default='./dominant_map.png', type=str)
    parser.add_argument('--temperature', default=1.0, type=float,
                        help='Temperature for attention softmax (<1.0 = sharper)')

    # Dominant map args
    parser.add_argument('--dominant_mode', default='head_level', type=str,
                        choices=['all_per_head', 'head_level'],
                        help='all_per_head: dominant register; head_level: dominant head')
    parser.add_argument('--direction', default='N2L', type=str, choices=['N2L', 'L2N'])
    parser.add_argument('--metric', default='attention', type=str, choices=['attention', 'similarity'])
    parser.add_argument('--overlay', action='store_true', default=True,
                        help='Overlay dominant map on original image')
    parser.add_argument('--no_overlay', action='store_true', help='Disable overlay')
    parser.add_argument('--overlay_alpha', default=0.5, type=float)
    parser.add_argument('--top_n', default=0, type=int,
                        help='Only use top N registers by variance (0=all)')
    parser.add_argument('--select_method', default='variance', type=str,
                        choices=['variance', 'max'],
                        help='How to rank registers for top_n: variance=most discriminative, max=highest peak')
    parser.add_argument('--dominant_norm', default='none', type=str,
                        choices=['none', 'cross_softmax', 'cross_rank',
                                 'spatial_smooth', 'spatial_median', 'spatial_max',
                                 'spatial_bilateral', 'spatial_dilate', 'local_contrast'],
                        help='Cross-register normalization before argmax')
    parser.add_argument('--smooth_sigma', default=1.0, type=float,
                        help='Gaussian sigma for spatial_smooth (higher=smoother, more coherent regions)')
    parser.add_argument('--dominant_temp', default=0.1, type=float,
                        help='Temperature for cross_softmax (lower=sharper competition)')
    parser.add_argument('--sim_mode', default='raw', type=str,
                        choices=['raw', 'sharpness', 'softmax', 'topk', 'threshold'],
                        help='Per-register similarity normalization mode (only for --metric similarity)')
    parser.add_argument('--sharpness', default=1.0, type=float,
                        help='Power transform for sim_mode=sharpness (>1=sharper contrast, e.g. 2.0)')
    parser.add_argument('--sim_temperature', default=0.07, type=float,
                        help='Temperature for sim_mode=softmax (lower=sharper)')
    parser.add_argument('--clean_min_patches', default=0, type=int,
                        help='Drop connected components smaller than this absolute patch count '
                             '(reassign to neighbour mode). 0/1 = disabled.')
    parser.add_argument('--clean_min_patches_frac', default=0.0, type=float,
                        help='Drop connected components smaller than frac x total_patches '
                             '(e.g. 0.005). Effective threshold = max(abs, frac). 0 = disabled.')
    parser.add_argument('--figsize', default=[10, 10], type=float, nargs=2)
    parser.add_argument('--dpi', default=150, type=int)

    args = parser.parse_args()

    if args.no_overlay:
        args.overlay = False

    print(f"\n{'='*60}")
    print(f"Draw Dominant Map")
    print(f"Model: {args.model_type} / {args.arch} / patch{args.patch_size}")
    print(f"Image: {args.image_path}")
    print(f"Output: {args.output_path}")
    print(f"Mode: {args.dominant_mode}, Direction: {args.direction}, Metric: {args.metric}")
    print(f"{'='*60}\n")

    # Load model
    model, device = load_model(args)
    if model is None:
        return

    num_register_tokens = getattr(model, 'num_register_tokens', 0)
    num_layers = len(model.blocks)
    embed_dim = model.embed_dim
    num_heads = model.blocks[0].attn.num_heads if hasattr(model.blocks[0].attn, 'num_heads') else 6

    # Load image
    img_tensor, img_pil, (h, w) = load_image(args.image_path, args.image_size, args.patch_size)
    img_tensor = img_tensor.to(device)

    h_feat = h // args.patch_size
    w_feat = w // args.patch_size
    num_patches = h_feat * w_feat

    print(f"Image size: {h}x{w}")
    print(f"Feature map: {h_feat}x{w_feat} ({num_patches} patches)")
    print(f"Registers: {num_register_tokens}")
    print(f"Layers: {num_layers}, Heads: {num_heads}, Embed dim: {embed_dim}")

    # Extract features
    extractor = LayerFeatureExtractor(model, args.model_type, temperature=args.temperature)
    layer_data = extractor.extract(img_tensor)

    # [no_norm] Skip model's final layer norm — use raw last block output
    # last_block_name = list(layer_data.keys())[-2]  # -1 is 'final_output'
    # if last_block_name in layer_data and 'output_tokens' in layer_data[last_block_name]:
    #     with torch.no_grad():
    #         layer_data[last_block_name]['output_tokens'] = model.norm(
    #             layer_data[last_block_name]['output_tokens']
    #         )

    # Convert to CPU float16 for memory efficiency
    def to_cpu_fp16(data):
        if isinstance(data, torch.Tensor):
            if data.is_floating_point():
                return data.cpu().half()
            return data.cpu()
        elif isinstance(data, dict):
            return {k: to_cpu_fp16(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [to_cpu_fp16(item) for item in data]
        return data

    layer_data_cpu = to_cpu_fp16(layer_data)

    # Build summary (register features, patch features, attention weights per layer)
    summary = extract_register_and_patch_features(layer_data_cpu, num_register_tokens)
    summary = to_cpu_fp16(summary)

    metadata = {
        'image_path': args.image_path,
        'image_size': (h, w),
        'patch_size': args.patch_size,
        'num_register_tokens': num_register_tokens,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'embed_dim': embed_dim,
        'feature_size': (h_feat, w_feat),
        'num_patches': num_patches,
        'model_type': args.model_type,
    }

    # Draw dominant map
    draw_dominant_map(
        summary, metadata, args.image_path, args.output_path,
        dominant_mode=args.dominant_mode,
        direction=args.direction,
        metric=args.metric,
        overlay=args.overlay,
        overlay_alpha=args.overlay_alpha,
        top_n=args.top_n,
        select_method=args.select_method,
        dominant_norm=args.dominant_norm,
        smooth_sigma=args.smooth_sigma,
        dominant_temp=args.dominant_temp,
        sim_mode=args.sim_mode,
        sharpness=args.sharpness,
        sim_temperature=args.sim_temperature,
        clean_min_patches=args.clean_min_patches,
        clean_min_patches_frac=args.clean_min_patches_frac,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
    )

    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
