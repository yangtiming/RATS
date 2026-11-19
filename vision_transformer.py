# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from utils import trunc_normal_


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    Multi-head attention with optional per-head register separation.

    When use_region_attn=True and num_register_tokens > 0:
        - Per-head register separation: each head only sees its own subset of registers
        - num_register_tokens must be divisible by num_heads
        - use_cross_attn=True:  L→N→N→L cross-attention
        - use_cross_attn=False: (reg_subset + L) x (reg_subset + L) self-attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 use_region_attn=False, num_register_tokens=0, use_cross_attn=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.head_dim = head_dim

        # Region attention settings
        self.use_region_attn = use_region_attn
        self.num_register_tokens = num_register_tokens
        self.use_cross_attn = use_cross_attn
        self._debug_printed = False  # For one-time debug print

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N_total, C = x.shape
        qkv = self.qkv(x).reshape(B, N_total, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N_total, head_dim]

        if not self.use_region_attn or self.num_register_tokens == 0:
            # Standard self-attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N_total, C)
        else:
            # Per-head register separation
            N_reg = self.num_register_tokens  # N
            N_patch = N_total - N_reg         # L
            regs_per_head = N_reg // self.num_heads

            assert N_reg % self.num_heads == 0, \
                f"num_register_tokens ({N_reg}) must be divisible by num_heads ({self.num_heads})"

            # Debug: print once to confirm per-head register attention is running
            if not self._debug_printed:
                mode_str = "LxN-NxN-NxL" if self.use_cross_attn else "(reg+L)x(reg+L) self-attn"
                print(f"[DEBUG] Per-head Register Attention: N_reg={N_reg}, N_patch={N_patch}, "
                      f"heads={self.num_heads}, regs_per_head={regs_per_head}, mode={mode_str}")
                self._debug_printed = True

            # Split registers and patches
            q_reg = q[:, :, :N_reg, :]      # [B, heads, N, head_dim]
            k_reg = k[:, :, :N_reg, :]
            v_reg = v[:, :, :N_reg, :]

            q_patch = q[:, :, N_reg:, :]    # [B, heads, L, head_dim]
            k_patch = k[:, :, N_reg:, :]
            v_patch = v[:, :, N_reg:, :]

            # Reorganize registers so each head only sees its own subset
            # head_i sees reg[i*regs_per_head : (i+1)*regs_per_head]
            q_reg_reshaped = q_reg.reshape(B, self.num_heads, self.num_heads, regs_per_head, self.head_dim)
            k_reg_reshaped = k_reg.reshape(B, self.num_heads, self.num_heads, regs_per_head, self.head_dim)
            v_reg_reshaped = v_reg.reshape(B, self.num_heads, self.num_heads, regs_per_head, self.head_dim)

            # Extract diagonal: head_i gets reg_group_i
            head_indices = torch.arange(self.num_heads, device=q.device)
            q_reg_diag = q_reg_reshaped[:, head_indices, head_indices, :, :]  # [B, heads, regs_per_head, head_dim]
            k_reg_diag = k_reg_reshaped[:, head_indices, head_indices, :, :]
            v_reg_diag = v_reg_reshaped[:, head_indices, head_indices, :, :]

            if self.use_cross_attn:
                # L→N→N→L cross-attention mode (with per-head separation)
                # Step 1: Per-head registers attend to patches (LxN)
                attn_L_to_N = (q_reg_diag @ k_patch.transpose(-2, -1)) * self.scale  # [B, h, regs_per_head, N_patch]
                attn_L_to_N = attn_L_to_N.softmax(dim=-1)
                attn_L_to_N = self.attn_drop(attn_L_to_N)
                out_reg_diag = attn_L_to_N @ v_patch  # [B, h, regs_per_head, head_dim]

                # Step 2: Registers self-attend (NxN, N=regs_per_head)
                attn_N_to_N = (out_reg_diag @ out_reg_diag.transpose(-2, -1)) * self.scale  # [B, h, regs_per_head, regs_per_head]
                attn_N_to_N = attn_N_to_N.softmax(dim=-1)
                attn_N_to_N = self.attn_drop(attn_N_to_N)
               
                out_reg_diag = attn_N_to_N @ out_reg_diag  # [B, h, regs_per_head, head_dim]

                # Step 3: Patches attend to updated per-head registers (NxL)
                attn_N_to_L = (q_patch @ out_reg_diag.transpose(-2, -1)) * self.scale  # [B, h, N_patch, regs_per_head]
                attn_N_to_L = attn_N_to_L.softmax(dim=-1)
                attn_N_to_L = self.attn_drop(attn_N_to_L)
                out_patch = attn_N_to_L @ out_reg_diag  # [B, h, N_patch, head_dim]

                attn = attn_N_to_L  # For visualization
            else:
                # (reg_subset + L) x (reg_subset + L) self-attention
                # Each head sees: [reg_subset_i, patches]
                q_combined = torch.cat([q_reg_diag, q_patch], dim=2)  # [B, heads, regs_per_head+L, head_dim]
                k_combined = torch.cat([k_reg_diag, k_patch], dim=2)
                v_combined = torch.cat([v_reg_diag, v_patch], dim=2)

                # Self-attention: (regs_per_head + L) x (regs_per_head + L)
                attn = (q_combined @ k_combined.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                out_combined = attn @ v_combined  # [B, heads, regs_per_head+L, head_dim]

                # Split output back
                out_reg_diag = out_combined[:, :, :regs_per_head, :]   # [B, heads, regs_per_head, head_dim]
                out_patch = out_combined[:, :, regs_per_head:, :]      # [B, heads, L, head_dim]

            # Reconstruct full register output (vectorized)
            out_reg = torch.zeros(B, self.num_heads, N_reg, self.head_dim, device=q.device, dtype=q.dtype)
            indices = torch.arange(self.num_heads, device=q.device).unsqueeze(1) * regs_per_head + \
                      torch.arange(regs_per_head, device=q.device).unsqueeze(0)  # [heads, regs_per_head]
            indices = indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, self.head_dim)  # [B, heads, regs_per_head, head_dim]
            out_reg.scatter_(2, indices, out_reg_diag)

            # Concatenate: [registers, patches]
            x = torch.cat([out_reg, out_patch], dim=2)  # [B, heads, N+L, head_dim]
            x = x.transpose(1, 2).reshape(B, N_total, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 use_region_attn=False, num_register_tokens=0, use_cross_attn=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop,
            use_region_attn=use_region_attn, num_register_tokens=num_register_tokens,
            use_cross_attn=use_cross_attn)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class DictionaryQuantizer(nn.Module):
    """
    Dominant-map dictionary: mirrors the visualization pipeline exactly.

    Flow (matches feature_visualizer_server.py dominant map):
      1. cosine_sim(register, patch) → [B, N_reg, N_patch]
      2. Top-N registers by max activation (select_top_registers, method='max')
      3. Spatial Gaussian smooth on 14×14 grid (normalize_scores_for_dominant, spatial_smooth)
      4. Cross-register softmax → soft dominant map (each patch soft-assigned to one register)
      5. Dominant-weighted patch aggregation → region features
      6. Region features → codebook quantization → losses
    """
    def __init__(self, num_codes=1024, embed_dim=384, tau=0.1,
                 top_n=6, spatial_sigma=0.8, mask_tau=0.1,
                 commitment_weight=0.25, codebook_weight=0.25, entropy_weight=0.01,
                 ema_decay=0.99):
        super().__init__()
        self.num_codes = num_codes
        self.tau = tau                      # temperature for codebook soft assignment
        self.top_n = top_n                  # number of dominant registers to select
        self.spatial_sigma = spatial_sigma  # Gaussian smooth sigma on spatial grid
        self.mask_tau = mask_tau            # cross-register softmax temperature
        self.commitment_weight = commitment_weight
        self.codebook_weight = codebook_weight
        self.entropy_weight = entropy_weight
        self._debug_printed = False
        self._gaussian_kernel = None

        # Codebook: nn.Parameter — updated by optimizer via gradient
        self.codebook = nn.Parameter(torch.randn(num_codes, embed_dim))
        nn.init.xavier_uniform_(self.codebook)

    def _make_gaussian_kernel(self, sigma, device, dtype):
        """Create fixed 2D Gaussian kernel for spatial smoothing (differentiable via conv2d)."""
        kernel_size = max(3, int(sigma * 4) | 1)  # odd number, at least 3
        x = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
        kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()
        return kernel_2d.reshape(1, 1, kernel_size, kernel_size)

    def forward(self, register_tokens, patch_tokens, return_debug=False):
        """
        Args:
            register_tokens: [B, N_reg, D]  (96 registers)
            patch_tokens:    [B, N_patch, D] (196 or 36 patches, used only for top-K selection)
            return_debug:    if True, also return intermediate tensors for visualization
        Returns:
            dict with 'dict_loss' and logging metrics (+ 'debug' dict if return_debug)
        """
        B, N_reg, D = register_tokens.shape
        N_patch = patch_tokens.shape[1]

        # === Step 1: Cosine similarity (for top-K selection only) ===
        reg_norm = F.normalize(register_tokens, dim=-1)
        patch_norm = F.normalize(patch_tokens, dim=-1)
        sim = torch.einsum('bnd,bmd->bnm', reg_norm, patch_norm)  # [B, N_reg, N_patch]

        # === Step 2: Top-N registers by max activation ===
        top_n = min(self.top_n, N_reg)
        max_act = sim.max(dim=-1).values  # [B, N_reg]
        _, top_idx = max_act.topk(top_n, dim=-1)  # [B, top_n]

        # === Step 3: Directly gather selected register tokens (skip dominant map & patch aggregation) ===
        top_idx_d = top_idx.unsqueeze(-1).expand(-1, -1, D)
        selected_regs = torch.gather(register_tokens, 1, top_idx_d)  # [B, top_n, D]

        if not self._debug_printed:
            print(f"[DEBUG] DirectDict: N_reg={N_reg}, top_n={top_n}, "
                  f"codebook={self.num_codes}x{D}, tau={self.tau} "
                  f"(no dominant map, register tokens directly quantized)")
            self._debug_printed = True

        # === Step 4: Soft quantize selected registers against codebook ===
        rf_norm = F.normalize(selected_regs, dim=-1)
        c_norm = F.normalize(self.codebook, dim=-1)

        # Cosine similarity → soft assignment
        code_sim = torch.einsum('brd,kd->brk', rf_norm, c_norm)  # [B, top_n, K]
        assignment = F.softmax(code_sim / self.tau, dim=-1)       # [B, top_n, K]

        # Soft quantized output: weighted sum of codebook entries
        z_q = torch.einsum('brk,kd->brd', assignment, self.codebook)  # [B, top_n, D]

        # Hard assignment for monitoring (not used in loss)
        hard_idx = code_sim.argmax(dim=-1)  # [B, top_n]

        # === Losses ===
        # Commitment loss: push selected registers toward z_q
        commitment_loss = F.mse_loss(selected_regs, z_q.detach())
        # Codebook loss: push z_q toward selected registers (gradient to codebook only)
        codebook_loss = F.mse_loss(z_q, selected_regs.detach())

        # Batch diversity loss (differentiable): maximize entropy of avg assignment
        max_entropy = math.log(self.num_codes)
        avg_prob = assignment.mean(dim=(0, 1))                     # [K]
        batch_entropy = -(avg_prob * torch.log(avg_prob + 1e-8)).sum()
        diversity_loss = max_entropy - batch_entropy

        total_loss = (self.commitment_weight * commitment_loss
                      + self.codebook_weight * codebook_loss
                      + self.entropy_weight * diversity_loss)

        # Utilization: fraction of codes used in this batch (hard argmax count)
        # Aggregate across all GPUs for global utilization
        usage_counts = torch.bincount(hard_idx.reshape(-1), minlength=self.num_codes).float()
        if dist.is_initialized():
            dist.all_reduce(usage_counts, op=dist.ReduceOp.SUM)
        num_used = (usage_counts > 0).sum().float()

        # Hard entropy for monitoring
        usage_prob = usage_counts / (usage_counts.sum() + 1e-8)
        entropy = -(usage_prob * torch.log(usage_prob + 1e-5)).sum()
        entropy_loss = max_entropy - entropy

        result = {
            'dict_loss': total_loss,
            'dict_commit': commitment_loss.detach(),
            'dict_codebook': codebook_loss.detach(),
            'dict_diversity': diversity_loss.detach(),
            'dict_entropy': entropy_loss.detach(),
            'dict_util': (num_used / self.num_codes).detach(),
        }

        if return_debug:
            result['debug'] = {
                'top_idx': top_idx.detach(),            # [B, top_n]
                'max_act': max_act.detach(),             # [B, N_reg]
                'selected_regs': selected_regs.detach(), # [B, top_n, D]
                'assignment': assignment.detach(),       # [B, top_n, K]
                'hard_idx': hard_idx.detach(),           # [B, top_n]
                'top_n': top_n,
            }

        return result


class VisionTransformer(nn.Module):
    """
    Vision Transformer with optional per-head register separation.

    Args:
        use_region_attn: if True, use per-head register attention instead of full self-attention
        num_register_tokens: number of register tokens (N). Must be divisible by num_heads.
        use_cross_attn: if True, use L→N→L cross-attention; if False, use (reg+L)x(reg+L) self-attention
        num_reg_for_loss_per_head: number of registers per head that contribute to loss.
                                   The remaining registers participate in attention but not loss.
                                   Default: all registers per head (num_register_tokens // num_heads).
        dict_size: number of dictionary codes (0 to disable).
    """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm,
                 use_region_attn=False, num_register_tokens=0, use_cross_attn=True,
                 num_reg_for_loss_per_head=None,
                 dict_size=0, dict_tau=0.1, top_n=6, spatial_sigma=0.8, mask_tau=0.1,
                 commitment_weight=0.25, codebook_weight=0.25, entropy_weight=0.01,
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_region_attn = use_region_attn
        self.num_register_tokens = num_register_tokens
        self.use_cross_attn = use_cross_attn

        # Calculate registers per head
        self.regs_per_head = num_register_tokens // num_heads if num_register_tokens > 0 else 0
        # num_reg_for_loss_per_head: how many registers per head contribute to loss (default: all)
        self.num_reg_for_loss_per_head = num_reg_for_loss_per_head if num_reg_for_loss_per_head is not None else self.regs_per_head
        self._first_forward = True  # Debug flag

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # Use register tokens if num_register_tokens > 0, otherwise use CLS token
        if num_register_tokens > 0:
            self.cls_token = None
            self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))  # No pos for registers
        else:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.register_tokens = None
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_region_attn=use_region_attn, num_register_tokens=num_register_tokens,
                use_cross_attn=use_cross_attn)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Dictionary quantizer (separate branch, does not affect DINO path)
        self.dict_quantizer = None
        if dict_size > 0 and num_register_tokens > 0:
            self.dict_quantizer = DictionaryQuantizer(
                num_codes=dict_size, embed_dim=embed_dim, tau=dict_tau,
                top_n=top_n, spatial_sigma=spatial_sigma, mask_tau=mask_tau,
                commitment_weight=commitment_weight, codebook_weight=codebook_weight,
                entropy_weight=entropy_weight)

        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)
        if self.register_tokens is not None:
            trunc_normal_(self.register_tokens, std=.02)
        self.apply(self._init_weights)

        # Debug: print model configuration
        print(f"[DEBUG] VisionTransformer initialized:")
        print(f"  - embed_dim: {embed_dim}, depth: {depth}, num_heads: {num_heads}")
        print(f"  - use_region_attn: {use_region_attn}, num_register_tokens: {num_register_tokens}, use_cross_attn: {use_cross_attn}")
        if num_register_tokens > 0:
            attn_mode = "L→N→N→L cross-attention" if use_cross_attn else "(reg+L)x(reg+L) self-attention"
            print(f"  - Mode: Per-head Register Attention (N={num_register_tokens}, regs_per_head={self.regs_per_head})")
            print(f"  - Attention type: {attn_mode}")
            print(f"  - Loss registers: {self.num_reg_for_loss_per_head} per head, total={self.num_reg_for_loss_per_head * num_heads}")
            print(f"  - No-loss registers: {self.regs_per_head - self.num_reg_for_loss_per_head} per head (propagation only)")
            if self.dict_quantizer is not None:
                print(f"  - Dictionary: {dict_size} codes, tau={dict_tau}, top_n={top_n}, sigma={spatial_sigma}, mask_tau={mask_tau}")
        else:
            print(f"  - Mode: Standard Self-Attention with CLS token")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]

        # For register tokens mode: pos_embed is only for patches
        if self.num_register_tokens > 0:
            if npatch == N and w == h:
                return self.pos_embed
            dim = x.shape[-1]
            w0 = w // self.patch_embed.patch_size
            h0 = h // self.patch_embed.patch_size
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                self.pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
            )
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return patch_pos_embed
        else:
            # Original CLS token mode
            npatch = npatch - 1
            N = N - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            w0 = w // self.patch_embed.patch_size
            h0 = h // self.patch_embed.patch_size
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # [B, L, C]

        if self.num_register_tokens > 0:
            # Region attention mode: [register_tokens, patch_tokens]
            x = x + self.interpolate_pos_encoding(x, w, h)  # Add pos to patches
            register_tokens = self.register_tokens.expand(B, -1, -1)  # [B, N, C]
            x = torch.cat((register_tokens, x), dim=1)  # [B, N+L, C]
            if self._first_forward:
                print(f"[DEBUG] prepare_tokens (Region): input={B}x{nc}x{w}x{h}, "
                      f"patches={x.shape[1]-self.num_register_tokens}, registers={self.num_register_tokens}, "
                      f"total_tokens={x.shape[1]}")
                self._first_forward = False
        else:
            # Original CLS token mode: [cls_token, patch_tokens]
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.interpolate_pos_encoding(x, w, h)
            if self._first_forward:
                print(f"[DEBUG] prepare_tokens (Standard): input={B}x{nc}x{w}x{h}, "
                      f"patches={x.shape[1]-1}, cls=1, total_tokens={x.shape[1]}")
                self._first_forward = False

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.num_register_tokens > 0:
            reg = x[:, :self.num_register_tokens]
            patch = x[:, self.num_register_tokens:]

            # Dictionary branch: register mask → select patches → quantize
            dict_losses = {}
            if self.dict_quantizer is not None:
                dict_losses = self.dict_quantizer(reg, patch)

            # Original DINO path (unchanged)
            if self.num_reg_for_loss_per_head < self.regs_per_head:
                indices = []
                for h in range(self.num_heads):
                    start = h * self.regs_per_head
                    indices.extend(range(start, start + self.num_reg_for_loss_per_head))
                indices = torch.tensor(indices, device=x.device)
                reg_for_loss = reg[:, indices]
                cls_output = reg_for_loss.mean(dim=1)
            else:
                cls_output = reg.mean(dim=1)

            if dict_losses:
                return cls_output, dict_losses
            return cls_output
        else:
            return x[:, 0]

    def extract_cls(self, x):
        """Extract cls-equivalent from token sequence [B, N_total, C].
        Mirrors the same logic as forward() and forward_features().
        """
        if self.num_register_tokens > 0:
            if self.num_reg_for_loss_per_head < self.regs_per_head:
                indices = []
                for h in range(self.num_heads):
                    start = h * self.regs_per_head
                    indices.extend(range(start, start + self.num_reg_for_loss_per_head))
                indices = torch.tensor(indices, device=x.device)
                return x[:, indices].mean(1)
            else:
                return x[:, :self.num_register_tokens].mean(1)
        else:
            return x[:, 0]  # standard CLS token

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

    def forward_features(self, x):
        """Return both register/cls and patch features."""
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.num_register_tokens > 0:
            # Get registers that contribute to loss
            if self.num_reg_for_loss_per_head < self.regs_per_head:
                indices = []
                for h in range(self.num_heads):
                    start = h * self.regs_per_head
                    indices.extend(range(start, start + self.num_reg_for_loss_per_head))
                indices = torch.tensor(indices, device=x.device)
                reg_for_loss = x[:, indices]
                cls_for_loss = reg_for_loss.mean(1)
            else:
                cls_for_loss = x[:, :self.num_register_tokens].mean(1)

            return {
                'register': x[:, :self.num_register_tokens],      # [B, N, C] all registers
                'patch': x[:, self.num_register_tokens:],         # [B, L, C]
                'cls': cls_for_loss,                              # [B, C] mean of loss registers only
            }
        else:
            return {
                'cls': x[:, 0],           # [B, C]
                'patch': x[:, 1:],        # [B, L, C]
            }


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# ============ Region ViT with per-head register attention ============

def region_vit_tiny(patch_size=16, num_register_tokens=3, use_cross_attn=True, **kwargs):
    """Region ViT-Tiny with per-head register attention. num_register_tokens must be divisible by 3 (num_heads)."""
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_region_attn=True, num_register_tokens=num_register_tokens,
        use_cross_attn=use_cross_attn, **kwargs)
    return model


def region_vit_small(patch_size=16, num_register_tokens=6, use_cross_attn=True, **kwargs):
    """Region ViT-Small with per-head register attention. num_register_tokens must be divisible by 6 (num_heads)."""
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_region_attn=True, num_register_tokens=num_register_tokens,
        use_cross_attn=use_cross_attn, **kwargs)
    return model



def region_vit_small_h12(patch_size=16, num_register_tokens=6, use_cross_attn=True, **kwargs):
    """Region ViT-Small with per-head register attention. num_register_tokens must be divisible by 6 (num_heads)."""
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_region_attn=True, num_register_tokens=num_register_tokens,
        use_cross_attn=use_cross_attn, **kwargs)
    return model



def region_vit_base(patch_size=16, num_register_tokens=12, use_cross_attn=True, **kwargs):
    """Region ViT-Base with per-head register attention. num_register_tokens must be divisible by 12 (num_heads)."""
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_region_attn=True, num_register_tokens=num_register_tokens,
        use_cross_attn=use_cross_attn, **kwargs)
    return model


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
