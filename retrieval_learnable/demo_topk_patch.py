"""
Learned Dictionary evaluation and visualization (dino_attn_base_dict variant).

Pipeline:
    trained model (with DictionaryQuantizer codebook) -> take the codebook directly as the dictionary
    -> extract register features + GT annotations on PartImageNet
    -> density voting + concentration filtering
    -> match register features to codebook entries (optional K-means refinement)
    -> tag entries with GT labels + purity scoring
    -> evaluate on PartImageNet val (exclusive assignment) + visualization

Differences from eval_k-means/demo_topk.py:
    - the dictionary comes from the codebook learned during model training, not post-hoc K-means
    - model loading needs extra args such as dict_size, dict_tau
    - K defaults to 2048 (codebook size), register defaults to 96

Usage examples:

    # Step 1: build the dictionary (extract codebook from the model + match against PartImageNet GT)
    python demo_topk.py cluster \
        --partimagenet_dir ../../_REGION_DINO/eval/data/PartImageNet \
        --pretrained_weights ../dino_attn_base_dict/output/region_vit_small_ps16_dict2048_dictonly/checkpoint.pth \
        --output_dir ./output_dict2048_patch \
        --K 2048 \
        --max_images 2000 \
        --topk_registers 6 \
        --topk_method max \
        --match_mode patch \
        --vote_metric attention \
        --concentration_thresh 0.5 \
        --split train \
        --image_size 512 512 \
        --dict_size 2048

    # Step 2: evaluate the dictionary's part correspondence on PartImageNet val
    python demo_topk.py eval_parts \
        --partimagenet_dir ../../_REGION_DINO/eval/data/PartImageNet \
        --pretrained_weights ../dino_attn_base_dict/output/region_vit_small_ps16_dict2048_dictonly/checkpoint.pth \
        --feature_dir ./output_dict2048_patch \
        --K 2048 \
        --match_mode patch \
        --vote_metric attention \
        --split val \
        --dict_size 2048

    # Optional: refine the codebook with K-means (initialize from the codebook, then run K-means)
    python eval_dict/demo_topk.py cluster \
        --partimagenet_dir ../_REGION_DINO/eval/data/PartImageNet \
        --pretrained_weights ./dino_attn_base_dict/output/region_vit_small_ps16_dict2048_dictonly/checkpoint.pth \
        --output_dir ./eval_dict/output_dict2048_refined \
        --K 2048 --dict_size 2048 \
        --refine_kmeans \
        --split train
        

"""

import os
import sys
import argparse
import glob
import json
import time
from collections import OrderedDict, defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as pth_transforms
from PIL import Image

# Add dino_attn_base_dict for model imports (vision_transformer, utils)
sys.path.insert(0, os.environ['MODEL_DIR'])


# ============================================================================
# Step 1: Model Loading & Feature Extraction
# ============================================================================

def load_model(args):
    """Load pretrained RegionDINO model with optional DictionaryQuantizer."""
    import vision_transformer as vits

    # Build model kwargs — include dict params if provided
    model_kwargs = dict(
        patch_size=args.patch_size,
        num_register_tokens=args.num_register_tokens,
        use_cross_attn=True,
        num_classes=0,
    )
    # Dict-related params (for DictionaryQuantizer)
    dict_size = getattr(args, 'dict_size', 0)
    if dict_size > 0:
        model_kwargs['dict_size'] = dict_size
        model_kwargs['dict_tau'] = getattr(args, 'dict_tau', 0.1)
        model_kwargs['top_n'] = getattr(args, 'top_n', 6)
        model_kwargs['spatial_sigma'] = getattr(args, 'spatial_sigma', 0.8)
        model_kwargs['mask_tau'] = getattr(args, 'mask_tau', 0.1)
    # Optional: num_reg_for_loss_per_head
    nrlph = getattr(args, 'num_reg_for_loss_per_head', None)
    if nrlph is not None:
        model_kwargs['num_reg_for_loss_per_head'] = nrlph

    arch = getattr(args, 'arch', 'region_vit_small')
    model = getattr(vits, arch)(**model_kwargs)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.pretrained_weights and os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu", weights_only=False)
        if 'teacher' in state_dict:
            state_dict = state_dict['teacher']
        state_dict = {k.replace("module.", "").replace("backbone.", ""): v
                      for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights: {args.pretrained_weights}")
        if msg.missing_keys:
            print(f"  Missing: {msg.missing_keys[:5]}")
        if msg.unexpected_keys:
            print(f"  Unexpected: {msg.unexpected_keys[:5]}")
        # Optional: inject a separately-stored codebook (pair a plain/backbone checkpoint
        # with a small codebook-only .pth), so we don't re-upload the backbone.
        cbfile = getattr(args, 'codebook', None)
        if cbfile and os.path.isfile(cbfile) and model.dict_quantizer is not None:
            cb = torch.load(cbfile, map_location="cpu", weights_only=False)
            cb = cb['codebook'] if isinstance(cb, dict) and 'codebook' in cb else cb
            with torch.no_grad():
                model.dict_quantizer.codebook.copy_(cb.to(model.dict_quantizer.codebook.device))
            print(f"  Codebook injected: {tuple(cb.shape)} from {cbfile}")
        # Report codebook status
        if model.dict_quantizer is not None:
            codebook = model.dict_quantizer.codebook
            print(f"  Codebook loaded: {codebook.shape} (dict_size={codebook.shape[0]})")
    else:
        print(f"WARNING: No weights found at {args.pretrained_weights}")

    return model, device


def get_transform(image_size, patch_size):
    """Get image transform."""
    return pth_transforms.Compose([
        pth_transforms.Resize(image_size),
        pth_transforms.CenterCrop(image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def extract_features_single(model, img_tensor, device):
    """
    Extract register token features, L→N attention maps, and similarity maps from a single image.
    Single forward pass: capture the last layer's L→N attention (registers attend to patches) via a hook,
    while extracting register token features from the model output and computing feature similarity.

    Returns:
        reg_features: [N_reg, C] - register token features from the last layer output (used for clustering)
        attn_maps: [N_reg, h_feat, w_feat] - each register's attention over patches (L→N direction)
        similarity_maps: [N_reg, h_feat, w_feat] - register-to-patch feature similarity (recommended for filtering)
    """
    img_tensor = img_tensor.unsqueeze(0).to(device) if img_tensor.dim() == 3 else img_tensor.to(device)
    B, _, H, W = img_tensor.shape
    patch_size = model.patch_embed.patch_size
    h_feat = H // patch_size
    w_feat = W // patch_size
    num_reg = model.num_register_tokens
    num_heads = model.num_heads
    regs_per_head = num_reg // num_heads
    head_dim = model.embed_dim // num_heads

    # Hook: capture the L→N attention inside the last block's attention forward
    captured = {}

    def attn_hook(module, input, output):
        """Hook on last block's Attention module.
        input[0] is the normed x: [B, N_total, C]
        We compute L→N attention (registers attend to patches).
        """
        x_in = input[0]
        B_h, N_total, C = x_in.shape
        qkv = module.qkv(x_in)
        qkv = qkv.reshape(B_h, N_total, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]

        # Per-head register separation (same as model forward)
        q_reg = q[:, :, :num_reg, :].reshape(B_h, num_heads, num_heads, regs_per_head, head_dim)
        k_patch = k[:, :, num_reg:, :]
        head_idx = torch.arange(num_heads, device=x_in.device)
        q_reg_diag = q_reg[:, head_idx, head_idx, :, :]  # [B, heads, regs_per_head, head_dim]

        # L→N: registers attend to patches
        attn_L_to_N = (q_reg_diag @ k_patch.transpose(-2, -1)) * module.scale
        attn_L_to_N = F.softmax(attn_L_to_N, dim=-1)  # [B, heads, regs_per_head, L]
        captured['attn_L_to_N'] = attn_L_to_N.detach()

    last_block = model.blocks[-1]
    hook = last_block.attn.register_forward_hook(attn_hook)

    # Single forward pass
    with torch.no_grad():
        x = model.prepare_tokens(img_tensor)
        for blk in model.blocks:
            x = blk(x)
        x = model.norm(x)

    hook.remove()

    # 1) Register token features: the first N_reg tokens of the last layer output
    reg_features = x[0, :num_reg, :].cpu()  # [N_reg, C]

    # 2) L→N attention maps: obtained from the hook
    attn_L_to_N = captured['attn_L_to_N']  # [B, heads, regs_per_head, L]
    attn_maps = torch.zeros(num_reg, h_feat * w_feat, device=device)
    for h in range(num_heads):
        reg_start = h * regs_per_head
        reg_end = (h + 1) * regs_per_head
        attn_maps[reg_start:reg_end, :] = attn_L_to_N[0, h, :, :]

    attn_maps = attn_maps.reshape(num_reg, h_feat, w_feat).cpu()  # [N_reg, h_feat, w_feat]

    # 3) Similarity maps: cosine similarity between register features and patch features
    patch_features = x[0, num_reg:, :].cpu()  # [L, C] - patch tokens

    # L2 normalize for cosine similarity
    reg_features_norm = F.normalize(reg_features, dim=-1)  # [N_reg, C]
    patch_features_norm = F.normalize(patch_features, dim=-1)  # [L, C]

    # Compute similarity: [N_reg, C] @ [C, L] = [N_reg, L]
    similarity_maps = (reg_features_norm @ patch_features_norm.T)  # [N_reg, L]
    similarity_maps = similarity_maps.reshape(num_reg, h_feat, w_feat)  # [N_reg, h_feat, w_feat]

    return reg_features, attn_maps, similarity_maps


def select_top_registers(score_maps, n, method='variance'):
    """
    Select top N registers based on score map quality metrics.

    Adapted from the select_top_registers() function in feature_visualizer_server.py.

    Supports two kinds of score maps:
    - Attention maps: softmax-normalized attention weights
    - Similarity maps: cosine similarity (recommended for feature clustering)

    Args:
        score_maps: [N_reg, h_feat, w_feat] - attention or similarity maps for all registers
        n: number of registers to select
        method: selection criterion
            - 'variance': highest variance across spatial locations (most discriminative) - recommended
            - 'max': highest peak activation value
            - 'entropy': moderate entropy (not too uniform, not too peaked)
            - 'sparsity': highest Gini coefficient (most focused)
            - 'diversity': iteratively select most diverse registers
            - 'mean': highest mean activation

    Returns:
        indices: list of selected register indices (sorted)
    """
    score_maps_np = score_maps.numpy() if hasattr(score_maps, 'numpy') else score_maps
    num_regs, h_feat, w_feat = score_maps_np.shape

    # Flatten spatial dimensions for analysis
    all_scores = score_maps_np.reshape(num_regs, -1)  # [N_reg, L]

    if n >= num_regs:
        return list(range(num_regs))

    if method == 'variance':
        # Select by variance (highest variance = most discriminative)
        scores = all_scores.var(axis=1)
        return np.argsort(scores)[-n:][::-1].tolist()

    elif method == 'max':
        # Select by max activation (highest peak response)
        scores = all_scores.max(axis=1)
        return np.argsort(scores)[-n:][::-1].tolist()

    elif method == 'mean':
        # Select by mean activation
        scores = all_scores.mean(axis=1)
        return np.argsort(scores)[-n:][::-1].tolist()

    elif method == 'entropy':
        # Select by entropy - prefer moderate entropy (not too uniform, not too peaked)
        # Normalize each register's scores to sum to 1 (like probability)
        eps = 1e-10
        probs = all_scores - all_scores.min(axis=1, keepdims=True)
        probs = probs / (probs.sum(axis=1, keepdims=True) + eps)
        entropy = -np.sum(probs * np.log(probs + eps), axis=1)
        # Prefer moderate entropy: score = -|entropy - median_entropy|
        median_ent = np.median(entropy)
        scores = -np.abs(entropy - median_ent)
        return np.argsort(scores)[-n:][::-1].tolist()

    elif method == 'sparsity':
        # Select by sparsity using Gini coefficient (higher = more unequal/focused)
        def gini(x):
            x = np.sort(x)
            n = len(x)
            cumsum = np.cumsum(x)
            return (2 * np.sum((np.arange(1, n+1) * x)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1] + 1e-10)

        scores = np.array([gini(row) for row in all_scores])
        return np.argsort(scores)[-n:][::-1].tolist()

    elif method == 'diversity':
        # Iteratively select registers that are most different from already selected
        # Start with highest variance register
        selected = [np.argmax(all_scores.var(axis=1))]

        # Normalize for cosine similarity
        norms = np.linalg.norm(all_scores, axis=1, keepdims=True) + 1e-10
        normalized = all_scores / norms

        while len(selected) < n:
            # For each unselected, compute min similarity to any selected
            min_sims = []
            for i in range(num_regs):
                if i in selected:
                    min_sims.append(float('inf'))
                else:
                    sims = [np.dot(normalized[i], normalized[j]) for j in selected]
                    min_sims.append(max(sims))  # max similarity = least diverse

            # Select the one with lowest max similarity (most diverse)
            next_idx = np.argmin(min_sims)
            selected.append(next_idx)

        return selected

    else:
        # Default to variance
        scores = all_scores.var(axis=1)
        return np.argsort(scores)[-n:][::-1].tolist()


def extract_all_features(args):
    """
    Step 1: Extract register token features from ImageNet val set.

    Saves:
        - features.npy: [total, C] all register features
        - attn_maps.npy: [total, h_feat, w_feat] corresponding attention maps
        - metadata.json: image_id, register_idx, class_name for each feature
        - config.json: model/extraction config
    """
    print("\n" + "=" * 60)
    print("Step 1: Feature Extraction")
    print("=" * 60)

    model, device = load_model(args)
    num_reg = model.num_register_tokens
    embed_dim = model.embed_dim

    # Top-K filtering parameters
    topk_registers = getattr(args, 'topk_registers', 0)
    topk_method = getattr(args, 'topk_method', 'variance')
    score_type = getattr(args, 'score_type', 'similarity')  # 'attention' or 'similarity'

    if topk_registers > 0 and topk_registers < num_reg:
        print(f"\n[Top-K Filtering] Will select top {topk_registers}/{num_reg} registers per image")
        print(f"  Method: {topk_method}")
        print(f"  Score type: {score_type} ({'feature similarity' if score_type == 'similarity' else 'attention weights'})")
        effective_num_reg = topk_registers
    else:
        print(f"\n[No Filtering] Using all {num_reg} registers")
        topk_registers = 0
        effective_num_reg = num_reg

    # Collect image paths from ImageNet val
    image_size = tuple(args.image_size)
    transform = get_transform(image_size, args.patch_size)

    # Get sorted class dirs
    class_dirs = sorted([d for d in os.listdir(args.imagenet_dir)
                         if os.path.isdir(os.path.join(args.imagenet_dir, d))])
    print(f"Found {len(class_dirs)} classes")

    # Collect image paths
    all_image_paths = []
    all_class_names = []
    for cls_name in class_dirs:
        cls_dir = os.path.join(args.imagenet_dir, cls_name)
        imgs = sorted(glob.glob(os.path.join(cls_dir, "*.JPEG")))
        for img_path in imgs:
            all_image_paths.append(img_path)
            all_class_names.append(cls_name)

    print(f"Total images available: {len(all_image_paths)}")

    # Subsample if needed
    if args.max_images > 0 and args.max_images < len(all_image_paths):
        np.random.seed(42)
        indices = np.random.choice(len(all_image_paths), args.max_images, replace=False)
        indices.sort()
        all_image_paths = [all_image_paths[i] for i in indices]
        all_class_names = [all_class_names[i] for i in indices]
        print(f"Subsampled to {len(all_image_paths)} images")

    # Prepare output
    os.makedirs(args.output_dir, exist_ok=True)

    h_feat = image_size[0] // args.patch_size
    w_feat = image_size[1] // args.patch_size

    total_features = len(all_image_paths) * effective_num_reg
    print(f"Expected features: {len(all_image_paths)} images x {effective_num_reg} registers = {total_features}")
    print(f"Feature dim: {embed_dim}, Attn map: {h_feat}x{w_feat}")
    if topk_registers > 0:
        print(f"[Filtering] Each image contributes {topk_registers} (out of {num_reg}) registers")

    # Pre-allocate arrays (memory-mapped for large datasets)
    feat_path = os.path.join(args.output_dir, "features.npy")
    attn_path = os.path.join(args.output_dir, "attn_maps.npy")

    features_mmap = np.lib.format.open_memmap(
        feat_path, mode='w+', dtype=np.float32, shape=(total_features, embed_dim))
    attn_mmap = np.lib.format.open_memmap(
        attn_path, mode='w+', dtype=np.float16, shape=(total_features, h_feat, w_feat))

    metadata = []
    idx = 0
    t0 = time.time()

    for img_idx, (img_path, cls_name) in enumerate(zip(all_image_paths, all_class_names)):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)

            # Ensure divisible by patch_size
            _, h, w = img_tensor.shape
            h = h - h % args.patch_size
            w = w - w % args.patch_size
            img_tensor = img_tensor[:, :h, :w]

            reg_features, attn_maps, similarity_maps = extract_features_single(model, img_tensor, device)

            # Apply top-K filtering if enabled
            if topk_registers > 0:
                # Choose which score map to use for selection
                score_maps = similarity_maps if score_type == 'similarity' else attn_maps
                selected_indices = select_top_registers(score_maps, topk_registers, method=topk_method)
            else:
                selected_indices = list(range(num_reg))

            # Store only selected registers
            for r in selected_indices:
                features_mmap[idx] = reg_features[r].numpy()
                attn_mmap[idx] = attn_maps[r].numpy().astype(np.float16)
                metadata.append({
                    'image_id': img_idx,
                    'image_path': img_path,
                    'register_idx': r,
                    'class_name': cls_name,
                    'is_filtered': topk_registers > 0,
                    'filter_method': topk_method if topk_registers > 0 else None,
                    'score_type': score_type if topk_registers > 0 else None,
                })
                idx += 1

        except Exception as e:
            print(f"  Error processing {img_path}: {e}")
            # Fill with zeros for failed images (only for expected number of registers)
            for _ in range(effective_num_reg):
                features_mmap[idx] = 0
                attn_mmap[idx] = 0
                metadata.append({
                    'image_id': img_idx,
                    'image_path': img_path,
                    'register_idx': -1,  # Unknown which register
                    'class_name': cls_name,
                    'error': str(e),
                    'is_filtered': topk_registers > 0,
                    'filter_method': topk_method if topk_registers > 0 else None,
                    'score_type': score_type if topk_registers > 0 else None,
                })
                idx += 1

        if (img_idx + 1) % 50 == 0 or img_idx == 0:
            elapsed = time.time() - t0
            speed = (img_idx + 1) / elapsed
            eta = (len(all_image_paths) - img_idx - 1) / speed
            print(f"  [{img_idx+1}/{len(all_image_paths)}] "
                  f"{speed:.1f} img/s, ETA: {eta/60:.1f} min")

    # Flush memory-mapped files
    del features_mmap
    del attn_mmap

    # Truncate to actual count (in case of errors)
    actual_count = idx
    if actual_count < total_features:
        features = np.load(feat_path, mmap_mode='r')[:actual_count].copy()
        np.save(feat_path, features)
        attns = np.load(attn_path, mmap_mode='r')[:actual_count].copy()
        np.save(attn_path, attns)

    # Save metadata
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata[:actual_count], f)

    # Save config
    config = {
        'num_images': len(all_image_paths),
        'num_register_tokens': num_reg,
        'effective_registers_per_image': effective_num_reg,
        'topk_registers': topk_registers,
        'topk_method': topk_method if topk_registers > 0 else None,
        'score_type': score_type if topk_registers > 0 else None,
        'embed_dim': embed_dim,
        'patch_size': args.patch_size,
        'image_size': list(image_size),
        'h_feat': h_feat,
        'w_feat': w_feat,
        'total_features': actual_count,
        'pretrained_weights': args.pretrained_weights,
    }
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone! Extracted {actual_count} features in {elapsed/60:.1f} min")
    print(f"Saved to: {args.output_dir}")


# ============================================================================
# Step 2-3: Extract from PartImageNet + Global K-means + GT Selection
# ============================================================================

def cluster_features(args):
    """
    Step 2-3: Build dictionary from learned codebook + PartImageNet GT matching.

    Flow:
      Phase 1: Extract register features from PartImageNet train
               - Density voting + concentration → label each register with GT part
               - Keep only registers with concentration >= threshold (default 0.5)
      Phase 2: Extract dictionary from model's codebook
               - If model has DictionaryQuantizer: use codebook directly
               - If --refine_kmeans: use codebook as K-means init, refine with data
               - Fallback: standard K-means (same as original demo_topk.py)
      Phase 3: Assign register features to dictionary entries → GT labeling

    Usage:
        python eval_dict/demo_topk.py cluster \\
            --partimagenet_dir /path/to/PartImageNet \\
            --pretrained_weights ./output/.../checkpoint.pth \\
            --output_dir ./eval_dict/output_dict2048 \\
            --K 2048 --dict_size 2048 \\
            --split train
    """
    from sklearn.preprocessing import normalize
    from sklearn.cluster import MiniBatchKMeans

    K_final = args.K
    conc_thresh = getattr(args, 'concentration_thresh', 0.5)
    vote_metric = getattr(args, 'vote_metric', 'attention')
    refine_kmeans = getattr(args, 'refine_kmeans', False)
    match_mode = getattr(args, 'match_mode', 'register')

    output_dir = getattr(args, 'output_dir', None) or args.feature_dir
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"Step 2-3: Learned Dict + GT Matching (K={K_final}, mode={match_mode})")
    print("=" * 60)

    # ==================================================================
    # Phase 1: Extract features from PartImageNet train + label with GT
    # ==================================================================
    model, device = load_model(args)
    num_reg = model.num_register_tokens
    embed_dim = model.embed_dim

    image_size = tuple(args.image_size)
    transform = get_transform(image_size, args.patch_size)

    split = getattr(args, 'split', 'train')
    img_dir = os.path.join(args.partimagenet_dir, 'images', split)
    ann_dir = os.path.join(args.partimagenet_dir, 'annotations', split)

    img_files = sorted([f for f in os.listdir(img_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPEG'))])
    max_images = getattr(args, 'max_images', 0)
    if max_images > 0 and max_images < len(img_files):
        np.random.seed(42)
        indices = np.random.choice(len(img_files), max_images, replace=False)
        img_files = [img_files[i] for i in sorted(indices)]

    all_features = []    # list of [C] arrays
    all_gt_parts = []    # list of part_name strings
    total_tokens = 0
    kept_tokens = 0
    t0 = time.time()

    if match_mode == 'patch':
        # ============================================================
        # Patch mode: use image patch tokens directly, matched to GT mask positions.
        # Each patch has a fixed spatial location -> look up the GT mask directly -> no attention needed.
        # ============================================================
        print(f"\n--- Phase 1 [PATCH MODE]: Extract patch tokens + GT labels ---")
        print(f"PartImageNet {split}: {len(img_files)} images")

        for img_idx, img_file in enumerate(img_files):
            basename = os.path.splitext(img_file)[0]
            mask_path = os.path.join(ann_dir, basename + '.png')
            if not os.path.exists(mask_path):
                continue

            try:
                img = Image.open(os.path.join(img_dir, img_file)).convert('RGB')
                img_tensor = transform(img)
                _, h, w = img_tensor.shape
                h = h - h % args.patch_size
                w = w - w % args.patch_size
                img_tensor = img_tensor[:, :h, :w]

                # Forward pass — only need patch tokens
                with torch.no_grad():
                    x_in = img_tensor.unsqueeze(0).to(device)
                    x = model.prepare_tokens(x_in)
                    for blk in model.blocks:
                        x = blk(x)
                    x = model.norm(x)

                patch_features = x[0, num_reg:, :].cpu().numpy()  # [N_patch, C]

                # GT mask → patch grid
                gt_mask = np.array(Image.open(mask_path))
                h_feat = h // args.patch_size
                w_feat = w // args.patch_size
                gt_mask_resized = np.array(
                    Image.fromarray(gt_mask).resize((w_feat, h_feat), Image.NEAREST))
                gt_flat = gt_mask_resized.flatten()  # [N_patch]

                # Per-part mean pooling: average all patches belonging to the same GT part
                for label_id in np.unique(gt_flat):
                    label_id = int(label_id)
                    total_tokens += 1
                    if label_id == 40:  # background
                        continue
                    part_name = PART_NAMES.get(label_id, f'Label_{label_id}')
                    mask = (gt_flat == label_id)
                    part_feat = patch_features[mask].mean(axis=0)  # [C]
                    all_features.append(part_feat)
                    all_gt_parts.append(part_name)
                    kept_tokens += 1

            except Exception as e:
                print(f"  Error {img_file}: {e}")
                continue

            if (img_idx + 1) % 100 == 0:
                elapsed = time.time() - t0
                speed = (img_idx + 1) / elapsed
                eta = (len(img_files) - img_idx - 1) / speed
                print(f"  [{img_idx+1}/{len(img_files)}] {speed:.1f} img/s, "
                      f"ETA: {eta/60:.1f} min, kept: {kept_tokens}/{total_tokens}")

    else:
        # ============================================================
        # Register mode (original logic): register attention + density voting
        # ============================================================
        print(f"\n--- Phase 1 [REGISTER MODE]: Extract & label registers ---")
        print(f"PartImageNet {split}: {len(img_files)} images")
        print(f"Vote metric: {vote_metric}, concentration_thresh: {conc_thresh}")

        topk_registers = getattr(args, 'topk_registers', 0)
        topk_method = getattr(args, 'topk_method', 'max')
        score_type = getattr(args, 'score_type', 'similarity')
        if topk_registers > 0:
            print(f"Top-K filtering: top {topk_registers} registers per image "
                  f"(method={topk_method}, score_type={score_type})")

        for img_idx, img_file in enumerate(img_files):
            basename = os.path.splitext(img_file)[0]
            mask_path = os.path.join(ann_dir, basename + '.png')
            if not os.path.exists(mask_path):
                continue

            try:
                img = Image.open(os.path.join(img_dir, img_file)).convert('RGB')
                img_tensor = transform(img)
                _, h, w = img_tensor.shape
                h = h - h % args.patch_size
                w = w - w % args.patch_size
                img_tensor = img_tensor[:, :h, :w]

                reg_features, attn_maps, similarity_maps = extract_features_single(
                    model, img_tensor, device)

                # Top-K register filtering
                if topk_registers > 0 and topk_registers < num_reg:
                    if score_type == 'similarity':
                        sel_indices = select_top_registers(
                            similarity_maps, topk_registers, topk_method)
                    else:
                        sel_indices = select_top_registers(
                            attn_maps, topk_registers, topk_method)
                else:
                    sel_indices = list(range(num_reg))

                # Load GT mask resized to feature map resolution
                gt_mask = np.array(Image.open(mask_path))
                h_feat = h // args.patch_size
                w_feat = w // args.patch_size
                gt_mask_resized = np.array(
                    Image.fromarray(gt_mask).resize((w_feat, h_feat), Image.NEAREST))

                for r in sel_indices:
                    total_tokens += 1
                    if vote_metric == 'similarity':
                        score_map = similarity_maps[r].numpy()
                    else:
                        score_map = attn_maps[r].numpy()

                    smin, smax = score_map.min(), score_map.max()
                    if smax > smin:
                        score_map = (score_map - smin) / (smax - smin)

                    part_density = {}
                    for label_id in np.unique(gt_mask_resized):
                        mask_for_label = (gt_mask_resized == label_id).astype(np.float32)
                        area = mask_for_label.sum()
                        if area > 0:
                            part_density[int(label_id)] = float(
                                (score_map * mask_for_label).sum() / area)

                    sorted_parts = sorted(part_density.items(), key=lambda x: -x[1])
                    best_label = sorted_parts[0][0]
                    best_density = sorted_parts[0][1]

                    if best_label == 40 and len(sorted_parts) > 1:
                        fg_parts = [(k, v) for k, v in sorted_parts if k != 40]
                        if fg_parts and fg_parts[0][1] > best_density * 0.5:
                            best_label = fg_parts[0][0]
                            best_density = fg_parts[0][1]

                    if best_label == 40:
                        continue

                    total_density = sum(part_density.values())
                    concentration = best_density / total_density if total_density > 0 else 0
                    if concentration < conc_thresh:
                        continue

                    part_name = PART_NAMES.get(best_label, f'Label_{best_label}')
                    all_features.append(reg_features[r].numpy())
                    all_gt_parts.append(part_name)
                    kept_tokens += 1

            except Exception as e:
                print(f"  Error {img_file}: {e}")
                continue

            if (img_idx + 1) % 100 == 0:
                elapsed = time.time() - t0
                speed = (img_idx + 1) / elapsed
                eta = (len(img_files) - img_idx - 1) / speed
                print(f"  [{img_idx+1}/{len(img_files)}] {speed:.1f} img/s, "
                      f"ETA: {eta/60:.1f} min, kept: {kept_tokens}/{total_tokens}")

    all_features = np.array(all_features)  # [N_kept, C]
    print(f"\nPhase 1 done: {kept_tokens}/{total_tokens} "
          f"{'patches' if match_mode == 'patch' else 'registers'} kept")
    print(f"Feature matrix: {all_features.shape}")

    part_counts = defaultdict(int)
    for p in all_gt_parts:
        part_counts[p] += 1
    for pname in sorted(part_counts.keys()):
        print(f"  {pname}: {part_counts[pname]}")

    # Save config (needed by eval_parts)
    config = {
        'image_size': list(image_size),
        'patch_size': args.patch_size,
        'num_register_tokens': num_reg,
        'embed_dim': embed_dim,
    }
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # ==================================================================
    # Phase 2: Get dictionary from codebook (or K-means fallback)
    # ==================================================================
    features_norm = normalize(all_features, norm='l2')

    # Optional PCA
    pca_model = None
    pca_dim = getattr(args, 'pca_dim', 0)
    if pca_dim > 0 and pca_dim < features_norm.shape[1]:
        from sklearn.decomposition import PCA
        print(f"PCA: {features_norm.shape[1]} -> {pca_dim}")
        pca_model = PCA(n_components=pca_dim, random_state=42)
        features_norm = pca_model.fit_transform(features_norm)

    has_codebook = (model.dict_quantizer is not None)

    if has_codebook and not refine_kmeans:
        # ---- Mode A: Directly use learned codebook as dictionary ----
        print(f"\n--- Phase 2: Using learned codebook directly (K={K_final}) ---")
        codebook = model.dict_quantizer.codebook.detach().cpu().numpy()  # [K, D]
        codebook_norm = normalize(codebook, norm='l2')
        if pca_model:
            codebook_norm = pca_model.transform(codebook_norm)
        dictionary = codebook_norm
        K_actual = dictionary.shape[0]
        print(f"Codebook shape: {codebook.shape} → dictionary: {dictionary.shape}")

        # Assign each feature to nearest codebook entry
        # L2 normalized → argmax cosine = argmin L2
        # Use matmul: [N, D] @ [D, K] = [N, K] (memory-friendly)
        cos_sim = features_norm @ dictionary.T  # [N, K]
        assignments = np.argmax(cos_sim, axis=1)  # [N]
        print(f"Assigned {len(assignments)} features to {K_actual} entries")

    elif has_codebook and refine_kmeans:
        # ---- Mode B: Use codebook as K-means initialization, refine ----
        print(f"\n--- Phase 2: Codebook-initialized K-means (K={K_final}) ---")
        codebook = model.dict_quantizer.codebook.detach().cpu().numpy()
        codebook_norm = normalize(codebook, norm='l2')
        if pca_model:
            codebook_norm = pca_model.transform(codebook_norm)

        K_actual = min(K_final, codebook_norm.shape[0], len(features_norm))
        print(f"Using codebook ({codebook_norm.shape}) as init for K-means (K={K_actual})")
        t_km = time.time()
        kmeans = MiniBatchKMeans(
            n_clusters=K_actual,
            init=codebook_norm[:K_actual],
            n_init=1,  # single init since we provide codebook
            batch_size=min(10000, len(features_norm)),
            max_iter=300,
            random_state=42,
            verbose=0,
        )
        assignments = kmeans.fit_predict(features_norm.astype(np.float32))
        dictionary = kmeans.cluster_centers_
        print(f"K-means refined in {time.time() - t_km:.1f}s, shape: {dictionary.shape}")

    else:
        # ---- Mode C: Fallback to pure K-means (no codebook) ----
        K_init = getattr(args, 'K_init', 0) or K_final * 4
        K_init = min(K_init, len(features_norm))
        print(f"\n--- Phase 2: Pure K-means (K_init={K_init}) ---")
        print(f"  No codebook found, falling back to K-means")
        t_km = time.time()
        kmeans = MiniBatchKMeans(
            n_clusters=K_init,
            batch_size=min(10000, len(features_norm)),
            n_init=getattr(args, 'n_init', 3),
            max_iter=300,
            random_state=42,
            verbose=0,
        )
        assignments = kmeans.fit_predict(features_norm.astype(np.float32))
        dictionary = kmeans.cluster_centers_
        K_actual = K_init
        print(f"K-means done in {time.time() - t_km:.1f}s, shape: {dictionary.shape}")

    # ==================================================================
    # Phase 3: GT label each entry, compute purity
    # ==================================================================
    print(f"\n--- Phase 3: GT labeling (all {K_actual} entries) ---")

    # Count GT labels per entry
    entry_part_counts = defaultdict(lambda: defaultdict(int))
    for i, eid in enumerate(assignments):
        entry_part_counts[int(eid)][all_gt_parts[i]] += 1

    entry_labels = []
    entry_details = []
    active_entries = 0
    purities = []

    for eid in range(K_actual):
        counts = entry_part_counts.get(eid, {})
        total = sum(counts.values())
        if total == 0:
            entry_labels.append('Unused')
            entry_details.append({
                'entry_id': eid, 'dominant_part': 'Unused',
                'purity': 0, 'count': 0, 'score': 0,
            })
            continue
        dominant_part = max(counts, key=counts.get)
        purity = counts[dominant_part] / total
        entry_labels.append(dominant_part)
        entry_details.append({
            'entry_id': eid,
            'dominant_part': dominant_part,
            'purity': purity,
            'count': total,
            'dominant_count': counts[dominant_part],
            'score': purity * np.log1p(counts[dominant_part]),
        })
        purities.append(purity)
        active_entries += 1

    print(f"Active entries: {active_entries}/{K_actual}")
    if purities:
        print(f"Avg purity: {np.mean(purities):.2%}")

    # Part distribution
    part_to_entries = defaultdict(list)
    for e in entry_details:
        if e['dominant_part'] != 'Unused':
            part_to_entries[e['dominant_part']].append(e)
    for pname in sorted(part_to_entries.keys()):
        entries = part_to_entries[pname]
        avg_p = np.mean([e['purity'] for e in entries])
        print(f"  {pname:15s}: {len(entries):3d} entries, avg purity={avg_p:.2%}")

    print(f"\nFinal dictionary: {dictionary.shape}")

    # Save
    dict_path = os.path.join(output_dir, f"dictionary_K{K_final}.npy")
    np.save(dict_path, dictionary)

    if pca_model:
        import pickle
        pca_path = os.path.join(output_dir, f"pca_model_K{K_final}.pkl")
        with open(pca_path, 'wb') as f:
            pickle.dump(pca_model, f)

    labels_path = os.path.join(output_dir, f"entry_labels_K{K_final}.json")
    with open(labels_path, 'w') as f:
        json.dump({
            'entry_labels': entry_labels,
            'entry_details': entry_details,
            'k_allocation': dict(Counter(entry_labels)),
            'K_init': K_actual,
            'K_final': K_final,
            'dict_source': 'codebook' if has_codebook else 'kmeans',
            'refine_kmeans': refine_kmeans,
            'concentration_thresh': conc_thresh,
            'vote_metric': vote_metric,
            'total_tokens': total_tokens,
            'kept_tokens': kept_tokens,
            'match_mode': match_mode,
            'split': split,
            'n_images': len(img_files),
        }, f, indent=2)

    print(f"\nSaved: {dict_path}")
    print(f"Saved: {labels_path}")


# ============================================================================
# Step 4: Visualization
# ============================================================================

def visualize_dictionary(args):
    """
    Step 4: Visualize dictionary entries - show top-k most typical examples.

    For each entry, shows original images with attention map overlay.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("\n" + "=" * 60)
    print(f"Step 4: Visualization (K={args.K}, topk={args.topk})")
    print("=" * 60)

    # Load data
    from sklearn.preprocessing import normalize

    features = np.load(os.path.join(args.feature_dir, "features.npy"))
    features = normalize(features, norm='l2')

    # Apply PCA if model exists
    pca_path = os.path.join(args.feature_dir, f"pca_model_K{args.K}.pkl")
    if os.path.exists(pca_path):
        import pickle
        with open(pca_path, 'rb') as f:
            pca_model = pickle.load(f)
        features = pca_model.transform(features)

    attn_maps = np.load(os.path.join(args.feature_dir, "attn_maps.npy"))
    dictionary = np.load(os.path.join(args.feature_dir, f"dictionary_K{args.K}.npy"))
    assignments = np.load(os.path.join(args.feature_dir, f"assignments_K{args.K}.npy"))

    with open(os.path.join(args.feature_dir, "metadata.json")) as f:
        metadata = json.load(f)

    # Select entries to visualize
    counts = np.bincount(assignments, minlength=args.K)

    if args.entry_ids:
        entry_ids = args.entry_ids
    else:
        # Pick entries with most members (most common patterns)
        entry_ids = np.argsort(-counts)[:args.num_entries].tolist()

    vis_dir = os.path.join(args.feature_dir, f"vis_K{args.K}")
    os.makedirs(vis_dir, exist_ok=True)

    print(f"Visualizing {len(entry_ids)} dictionary entries...")

    for entry_id in entry_ids:
        indices = np.where(assignments == entry_id)[0]
        if len(indices) == 0:
            continue

        # Compute distances to centroid, pick closest (most typical)
        dists = np.linalg.norm(features[indices] - dictionary[entry_id], axis=1)
        sorted_idx = np.argsort(dists)
        top_indices = indices[sorted_idx[:args.topk]]

        # Collect unique images for these features
        ncols = min(len(top_indices), args.topk)
        if ncols == 0:
            continue

        fig, axes = plt.subplots(2, ncols, figsize=(3 * ncols, 6))
        if ncols == 1:
            axes = axes.reshape(2, 1)

        # Get class distribution for title
        classes = [metadata[i]['class_name'] for i in top_indices]
        class_counts = defaultdict(int)
        for c in [metadata[i]['class_name'] for i in indices]:
            class_counts[c] += 1
        top_classes = sorted(class_counts.items(), key=lambda x: -x[1])[:5]
        class_str = ", ".join([f"{c}({n})" for c, n in top_classes])

        for col, idx in enumerate(top_indices):
            meta = metadata[idx]
            img_path = meta['image_path']
            attn_map = attn_maps[idx]  # [h_feat, w_feat]

            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((256, 256))
                img_np = np.array(img)

                # Resize attention map to image size
                attn_resized = np.array(
                    Image.fromarray(attn_map.astype(np.float32)).resize((256, 256), Image.BILINEAR))

                # Row 0: original image
                axes[0, col].imshow(img_np)
                axes[0, col].set_title(f"r{meta['register_idx']}", fontsize=8)
                axes[0, col].axis('off')

                # Row 1: attention overlay
                axes[1, col].imshow(img_np)
                axes[1, col].imshow(attn_resized, alpha=0.6, cmap='hot')
                axes[1, col].axis('off')

            except Exception as e:
                axes[0, col].text(0.5, 0.5, 'Error', ha='center', va='center')
                axes[0, col].axis('off')
                axes[1, col].axis('off')

        plt.suptitle(
            f"Entry #{entry_id} (count={len(indices)})\n{class_str}",
            fontsize=10, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"entry_{entry_id:04d}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Saved visualizations to: {vis_dir}")

    # Also create a summary grid of all visualized entries
    _create_summary_grid(vis_dir, entry_ids, counts)


def _create_summary_grid(vis_dir, entry_ids, counts):
    """Create a single summary image showing all visualized entries."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    existing = [eid for eid in entry_ids
                if os.path.exists(os.path.join(vis_dir, f"entry_{eid:04d}.png"))]

    if len(existing) == 0:
        return

    ncols = 5
    nrows = (len(existing) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for i, eid in enumerate(existing):
        row, col = i // ncols, i % ncols
        img = Image.open(os.path.join(vis_dir, f"entry_{eid:04d}.png"))
        axes[row, col].imshow(np.array(img))
        axes[row, col].set_title(f"Entry #{eid} (n={counts[eid]})", fontsize=8)
        axes[row, col].axis('off')

    # Hide empty subplots
    for i in range(len(existing), nrows * ncols):
        row, col = i // ncols, i % ncols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "summary.png"), dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved summary grid: {os.path.join(vis_dir, 'summary.png')}")


# ============================================================================
# Step 5: Quantitative Analysis
# ============================================================================

def analyze_dictionary(args):
    """
    Step 5: Quantitative analysis of the dictionary.

    - Dictionary utilization
    - Assignment distribution
    - Cross-class consistency per entry
    """
    print("\n" + "=" * 60)
    print(f"Step 5: Quantitative Analysis (K={args.K})")
    print("=" * 60)

    assignments = np.load(os.path.join(args.feature_dir, f"assignments_K{args.K}.npy"))
    with open(os.path.join(args.feature_dir, "metadata.json")) as f:
        metadata = json.load(f)

    K = args.K
    counts = np.bincount(assignments, minlength=K)

    # 1. Dictionary utilization
    used = np.sum(counts > 0)
    print(f"\n1. Dictionary Utilization: {used}/{K} = {used/K:.2%}")

    # 2. Assignment distribution
    print(f"\n2. Assignment Distribution:")
    print(f"   Max: {counts.max()}, Min: {counts.min()}, "
          f"Median: {np.median(counts):.0f}, Mean: {counts.mean():.0f}, "
          f"Std: {counts.std():.0f}")

    # Distribution percentiles
    for p in [10, 25, 50, 75, 90]:
        print(f"   P{p}: {np.percentile(counts, p):.0f}")

    # 3. Cross-class consistency
    print(f"\n3. Cross-class Consistency:")
    entry_class_stats = []

    for entry_id in range(K):
        indices = np.where(assignments == entry_id)[0]
        if len(indices) < 5:  # skip very small clusters
            continue

        classes = [metadata[i]['class_name'] for i in indices]
        unique_classes = len(set(classes))
        total = len(classes)

        # Compute class concentration: what fraction comes from top-3 classes
        class_counts = defaultdict(int)
        for c in classes:
            class_counts[c] += 1
        top3 = sorted(class_counts.values(), reverse=True)[:3]
        top3_ratio = sum(top3) / total

        entry_class_stats.append({
            'entry_id': entry_id,
            'total': total,
            'unique_classes': unique_classes,
            'top3_ratio': top3_ratio,
        })

    if entry_class_stats:
        avg_unique = np.mean([s['unique_classes'] for s in entry_class_stats])
        avg_top3 = np.mean([s['top3_ratio'] for s in entry_class_stats])
        print(f"   Avg unique classes per entry: {avg_unique:.1f}")
        print(f"   Avg top-3 class concentration: {avg_top3:.2%}")

        # Entries with highest class concentration (likely semantic)
        by_concentration = sorted(entry_class_stats, key=lambda x: -x['top3_ratio'])
        print(f"\n   Top 10 most class-concentrated entries:")
        for s in by_concentration[:10]:
            idx = np.where(assignments == s['entry_id'])[0]
            classes = [metadata[i]['class_name'] for i in idx]
            cc = defaultdict(int)
            for c in classes:
                cc[c] += 1
            top3_cls = sorted(cc.items(), key=lambda x: -x[1])[:3]
            cls_str = ", ".join([f"{c}({n})" for c, n in top3_cls])
            print(f"     Entry #{s['entry_id']:4d}: {s['total']:5d} features, "
                  f"{s['unique_classes']:3d} classes, top3={s['top3_ratio']:.2%} [{cls_str}]")

        # Entries spread across many classes (likely universal parts)
        by_spread = sorted(entry_class_stats, key=lambda x: -x['unique_classes'])
        print(f"\n   Top 10 most class-diverse entries (potential universal parts):")
        for s in by_spread[:10]:
            print(f"     Entry #{s['entry_id']:4d}: {s['total']:5d} features, "
                  f"{s['unique_classes']:3d} classes, top3={s['top3_ratio']:.2%}")

    # Save analysis
    analysis = {
        'utilization': float(used / K),
        'count_stats': {
            'max': int(counts.max()), 'min': int(counts.min()),
            'median': float(np.median(counts)), 'mean': float(counts.mean()),
        },
        'class_consistency': {
            'avg_unique_classes': float(avg_unique) if entry_class_stats else 0,
            'avg_top3_concentration': float(avg_top3) if entry_class_stats else 0,
        },
        'entry_details': entry_class_stats[:100],  # save top entries
    }
    analysis_path = os.path.join(args.feature_dir, f"analysis_K{args.K}.json")
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved analysis: {analysis_path}")


# ============================================================================
# Step 5b: PartImageNet quantitative evaluation
# ============================================================================

# PartImageNet label → part name mapping
PART_NAMES = {
    0: 'Quad_Head', 1: 'Quad_Body', 2: 'Quad_Foot', 3: 'Quad_Tail',
    4: 'Biped_Head', 5: 'Biped_Body', 6: 'Biped_Hand', 7: 'Biped_Foot', 8: 'Biped_Tail',
    9: 'Fish_Head', 10: 'Fish_Body', 11: 'Fish_Fin', 12: 'Fish_Tail',
    13: 'Bird_Head', 14: 'Bird_Body', 15: 'Bird_Wing', 16: 'Bird_Foot', 17: 'Bird_Tail',
    18: 'Snake_Head', 19: 'Snake_Body',
    20: 'Rept_Head', 21: 'Rept_Body', 22: 'Rept_Foot', 23: 'Rept_Tail',
    24: 'Car_Body', 25: 'Car_Tier', 26: 'Car_Mirror',
    27: 'Bike_Head', 28: 'Bike_Body', 29: 'Bike_Seat', 30: 'Bike_Tier',
    31: 'Boat_Body', 32: 'Boat_Sail',
    33: 'Aero_Head', 34: 'Aero_Body', 35: 'Aero_Wing', 36: 'Aero_Engine', 37: 'Aero_Tail',
    38: 'Bottle_Body', 39: 'Bottle_Mouth',
    40: 'Background',
}

# Cross-category semantic part grouping
SEMANTIC_PARTS = {
    'Head':      [0, 4, 9, 13, 18, 20, 27, 33],
    'Body':      [1, 5, 10, 14, 19, 21, 24, 28, 31, 34, 38],
    'Foot/Tier': [2, 7, 16, 22, 25, 30],
    'Tail':      [3, 8, 12, 17, 23, 37],
    'Wing/Fin':  [11, 15, 35],
    'Other':     [6, 26, 29, 32, 36, 39],  # Hand, Mirror, Seat, Sail, Engine, Mouth
}

# Reverse mapping: label_id → semantic part name
LABEL_TO_SEMANTIC = {}
for sem_name, label_ids in SEMANTIC_PARTS.items():
    for lid in label_ids:
        LABEL_TO_SEMANTIC[lid] = sem_name
LABEL_TO_SEMANTIC[40] = 'Background'


def eval_partimagenet(args):
    """
    Quantitatively evaluate the part correspondence of register tokens using PartImageNet GT segmentation.

    Idea:
    1. Extract register attention maps for PartImageNet images
    2. Find which GT part the peak-activation region of each register's attention map falls on
    3. Assign via the existing K-means dictionary -> measure whether each entry consistently maps to the same part
    4. Report: part purity, cross-category part transferability

    Usage:
        python demo.py eval_parts \
            --partimagenet_dir /path/to/PartImageNet \
            --feature_dir ./kmeans_output \
            --K 512 \
            --pretrained_weights ../dino_attn_base/output/.../checkpoint.pth
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import normalize

    print("\n" + "=" * 60)
    print("Step 5b: PartImageNet Evaluation")
    print("=" * 60)

    # Load model
    model, device = load_model(args)

    # Load dictionary
    dict_path = os.path.join(args.feature_dir, f"dictionary_K{args.K}.npy")
    dictionary = np.load(dict_path)
    print(f"Dictionary: {dict_path} ({dictionary.shape})")

    # Load entry labels if available (from GT-guided clustering)
    entry_labels_map = None
    labels_path = os.path.join(args.feature_dir, f"entry_labels_K{args.K}.json")
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            entry_labels_info = json.load(f)
        entry_labels_map = entry_labels_info.get('entry_labels')
        if entry_labels_map:
            print(f"Entry labels loaded: {len(set(entry_labels_map))} GT parts")

    with open(os.path.join(args.feature_dir, "config.json")) as f:
        config = json.load(f)

    pca_model = None
    pca_path = os.path.join(args.feature_dir, f"pca_model_K{args.K}.pkl")
    if os.path.exists(pca_path):
        import pickle
        with open(pca_path, 'rb') as f:
            pca_model = pickle.load(f)

    # Prepare
    image_size = tuple(config['image_size'])
    transform = get_transform(image_size, args.patch_size)
    num_reg = model.num_register_tokens

    # Collect PartImageNet images & annotations
    split = args.split
    img_dir = os.path.join(args.partimagenet_dir, 'images', split)
    ann_dir = os.path.join(args.partimagenet_dir, 'annotations', split)

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.JPEG')])
    if args.max_images > 0 and args.max_images < len(img_files):
        np.random.seed(42)
        indices = np.random.choice(len(img_files), args.max_images, replace=False)
        img_files = [img_files[i] for i in sorted(indices)]

    match_mode = getattr(args, 'match_mode', 'register')
    print(f"Images: {len(img_files)} from {split} split")
    print(f"Dictionary: K={args.K}")
    print(f"Match mode: {match_mode}")
    vote_metric = getattr(args, 'vote_metric', 'attention')
    if match_mode == 'register':
        print(f"Vote metric: {vote_metric} ({'similarity-based' if vote_metric == 'similarity' else 'attention-based'} weighted voting)")

    # Pre-normalize dictionary for cosine similarity (used in patch mode)
    dictionary_norm = normalize(dictionary, norm='l2')

    # For each token: record (dict_entry, gt_part_label, semantic_part)
    all_records = []
    t0 = time.time()

    if match_mode == 'patch':
        # ==========================================================
        # Patch mode: per-part mean pool of patch tokens → cosine sim
        # ==========================================================
        print(f"\n--- PATCH MODE: per-part mean pool + codebook cosine sim ---")

        for img_idx, img_file in enumerate(img_files):
            basename = os.path.splitext(img_file)[0]
            mask_path = os.path.join(ann_dir, basename + '.png')
            if not os.path.exists(mask_path):
                continue

            try:
                img = Image.open(os.path.join(img_dir, img_file)).convert('RGB')
                img_tensor = transform(img)
                _, h, w = img_tensor.shape
                h = h - h % args.patch_size
                w = w - w % args.patch_size
                img_tensor = img_tensor[:, :h, :w]

                # Forward pass — extract patch tokens
                with torch.no_grad():
                    x_in = img_tensor.unsqueeze(0).to(device)
                    x = model.prepare_tokens(x_in)
                    for blk in model.blocks:
                        x = blk(x)
                    x = model.norm(x)

                patch_features = x[0, num_reg:, :].cpu().numpy()  # [N_patch, C]

                # GT mask → patch grid
                gt_mask = np.array(Image.open(mask_path))
                h_feat = h // args.patch_size
                w_feat = w // args.patch_size
                gt_mask_resized = np.array(
                    Image.fromarray(gt_mask).resize((w_feat, h_feat), Image.NEAREST))
                gt_flat = gt_mask_resized.flatten()  # [N_patch]

                # Per-part mean pooling → cosine sim to codebook
                for label_id in np.unique(gt_flat):
                    label_id = int(label_id)
                    if label_id == 40:  # skip background
                        continue

                    mask = (gt_flat == label_id)
                    part_feat = patch_features[mask].mean(axis=0)  # [C]
                    part_feat_norm = part_feat / (np.linalg.norm(part_feat) + 1e-8)

                    cos_sim = part_feat_norm @ dictionary_norm.T  # [K]
                    best_entry = int(np.argmax(cos_sim))
                    best_sim = float(cos_sim[best_entry])

                    part_name = PART_NAMES.get(label_id, f'Label_{label_id}')
                    semantic = LABEL_TO_SEMANTIC.get(label_id, 'Unknown')

                    all_records.append({
                        'dict_entry': best_entry,
                        'gt_label': label_id,
                        'gt_part': part_name,
                        'semantic': semantic,
                        'score_density': best_sim,
                        'concentration': 1.0,  # direct GT match
                        'vote_metric': 'patch_gt',
                        'image': img_file,
                        'register': -1,  # no register in patch mode
                    })

            except Exception as e:
                print(f"  Error {img_file}: {e}")
                continue

            if (img_idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                speed = (img_idx + 1) / elapsed
                eta = (len(img_files) - img_idx - 1) / speed
                print(f"  [{img_idx+1}/{len(img_files)}] {speed:.1f} img/s, ETA: {eta/60:.1f} min")

    else:
        # ==========================================================
        # Register mode (original): attention density voting
        # ==========================================================
        print(f"\n--- REGISTER MODE: attention density voting ---")

        for img_idx, img_file in enumerate(img_files):
            basename = os.path.splitext(img_file)[0]
            mask_path = os.path.join(ann_dir, basename + '.png')
            if not os.path.exists(mask_path):
                continue

            try:
                # Load image
                img = Image.open(os.path.join(img_dir, img_file)).convert('RGB')
                img_tensor = transform(img)
                _, h, w = img_tensor.shape
                h = h - h % args.patch_size
                w = w - w % args.patch_size
                img_tensor = img_tensor[:, :h, :w]

                # Extract register features + attention maps + similarity maps
                reg_features, attn_maps, similarity_maps = extract_features_single(model, img_tensor, device)

                # Assign to dictionary
                reg_features_np = normalize(reg_features.numpy(), norm='l2')
                if pca_model:
                    reg_features_np = pca_model.transform(reg_features_np)
                dists = np.linalg.norm(
                    reg_features_np[:, None, :] - dictionary[None, :, :], axis=2)
                assignments = np.argmin(dists, axis=1)  # [N_reg]

                # Load GT mask and resize to attention map resolution
                gt_mask = np.array(Image.open(mask_path))  # [H_orig, W_orig]
                h_feat = h // args.patch_size
                w_feat = w // args.patch_size
                # Resize GT mask to feature map size using nearest neighbor
                gt_mask_resized = np.array(
                    Image.fromarray(gt_mask).resize((w_feat, h_feat), Image.NEAREST))

                for r in range(num_reg):
                    # Choose metric for weighted voting
                    if vote_metric == 'similarity':
                        score_map = similarity_maps[r].numpy()  # [h_feat, w_feat]
                    else:  # 'attention'
                        score_map = attn_maps[r].numpy()  # [h_feat, w_feat]

                    # Min-max normalization
                    smin, smax = score_map.min(), score_map.max()
                    if smax > smin:
                        score_map = (score_map - smin) / (smax - smin)

                    # Density-based voting: average score per pixel in each region
                    part_density = {}
                    for label_id in np.unique(gt_mask_resized):
                        mask_for_label = (gt_mask_resized == label_id).astype(np.float32)
                        area = mask_for_label.sum()
                        if area > 0:
                            part_density[int(label_id)] = float(
                                (score_map * mask_for_label).sum() / area)

                    # Pick the part with highest density
                    sorted_parts = sorted(part_density.items(), key=lambda x: -x[1])
                    best_label = sorted_parts[0][0]
                    best_density = sorted_parts[0][1]

                    # If best is background, prefer foreground if its density is reasonable
                    if best_label == 40 and len(sorted_parts) > 1:
                        fg_parts = [(k, v) for k, v in sorted_parts if k != 40]
                        if fg_parts and fg_parts[0][1] > best_density * 0.5:
                            best_label = fg_parts[0][0]
                            best_density = fg_parts[0][1]

                    # Concentration: best density vs. overall mean density
                    total_density = sum(part_density.values())
                    concentration = (best_density / total_density
                                     if total_density > 0 else 0)

                    semantic = LABEL_TO_SEMANTIC.get(best_label, 'Unknown')
                    part_name = PART_NAMES.get(best_label, f'Label_{best_label}')

                    all_records.append({
                        'dict_entry': int(assignments[r]),
                        'gt_label': int(best_label),
                        'gt_part': part_name,
                        'semantic': semantic,
                        'score_density': float(best_density),
                        'concentration': float(concentration),
                        'vote_metric': vote_metric,
                        'image': img_file,
                        'register': r,
                    })

            except Exception as e:
                print(f"  Error {img_file}: {e}")
                continue

            if (img_idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                speed = (img_idx + 1) / elapsed
                eta = (len(img_files) - img_idx - 1) / speed
                print(f"  [{img_idx+1}/{len(img_files)}] {speed:.1f} img/s, ETA: {eta/60:.1f} min")

    print(f"\nProcessed {len(all_records)} {'parts' if match_mode == 'patch' else 'register tokens'}")

    # =========================================================
    # Analysis 1: Per-entry part purity (using fine-grained GT labels)
    # =========================================================
    print(f"\n{'='*60}")
    print("Analysis 1: Dictionary Entry Part Purity (Fine-grained Labels)")
    print(f"{'='*60}")

    # Group by fine-grained gt_part instead of semantic
    entry_gtpart_counts = defaultdict(lambda: defaultdict(int))
    for rec in all_records:
        entry_gtpart_counts[rec['dict_entry']][rec['gt_part']] += 1

    purities = []
    entry_results = []
    for entry_id in sorted(entry_gtpart_counts.keys()):
        counts = entry_gtpart_counts[entry_id]
        total = sum(counts.values())
        if total < 3:
            continue
        dominant_part = max(counts, key=counts.get)  # Now this is fine-grained label
        purity = counts[dominant_part] / total
        purities.append(purity)
        entry_results.append({
            'entry_id': entry_id,
            'dominant_part': dominant_part,  # Fine-grained label like "Quadrupedal_Head"
            'purity': purity,
            'total': total,
            'distribution': dict(counts),
        })

    if purities:
        avg_purity = np.mean(purities)
        print(f"\nAverage part purity: {avg_purity:.2%}")
        print(f"  (1.0 = every entry maps to exactly one semantic part)")
        print(f"  Entries evaluated: {len(purities)}")

        # If we have designed labels from cluster, check consistency
        if entry_labels_map and len(entry_labels_map) == args.K:
            n_match = 0
            n_total = 0
            for e in entry_results:
                eid = e['entry_id']
                if eid < len(entry_labels_map):
                    designed = entry_labels_map[eid]
                    observed = e['dominant_part']
                    n_total += 1
                    if designed == observed:
                        n_match += 1
            if n_total > 0:
                print(f"\n  Designed vs. observed label consistency: "
                      f"{n_match}/{n_total} ({n_match/n_total:.1%})")
                print(f"  (How often the eval-dominant label matches "
                      f"the cluster-designed label)")

        # Show top pure entries (excluding Background-dominated)
        fg_entries = [e for e in entry_results if e['dominant_part'] != 'Background']
        fg_entries.sort(key=lambda x: -x['purity'])
        print(f"\nTop 20 purest foreground entries:")
        for e in fg_entries[:20]:
            dist_str = ", ".join(f"{k}:{v}" for k, v in
                                sorted(e['distribution'].items(), key=lambda x: -x[1])[:4])
            print(f"  Entry #{e['entry_id']:4d}: {e['dominant_part']:12s} "
                  f"purity={e['purity']:.2%} (n={e['total']}) [{dist_str}]")

    # =========================================================
    # Analysis 2: Cross-category part transfer
    # =========================================================
    print(f"\n{'='*60}")
    print("Analysis 2: Cross-category Part Transfer")
    print(f"{'='*60}")
    print("(Do different animal types' 'Head' land in the same entry?)")

    # For each semantic part, which dict entries capture it?
    semantic_to_entries = defaultdict(lambda: defaultdict(int))
    for rec in all_records:
        if rec['semantic'] != 'Background':
            semantic_to_entries[rec['semantic']][rec['dict_entry']] += 1

    for sem_part in ['Head', 'Body', 'Foot/Tier', 'Tail', 'Wing/Fin']:
        entry_counts = semantic_to_entries.get(sem_part, {})
        if not entry_counts:
            continue
        total = sum(entry_counts.values())
        top_entries = sorted(entry_counts.items(), key=lambda x: -x[1])[:5]
        top5_mass = sum(v for _, v in top_entries) / total

        # For top entries: which specific part labels (cross-category) contribute?
        print(f"\n  {sem_part} ({total} tokens):")
        print(f"    Top-5 entries capture {top5_mass:.1%} of all {sem_part} tokens")
        for eid, cnt in top_entries:
            # Which categories contribute to this entry for this part?
            cats = defaultdict(int)
            for rec in all_records:
                if rec['dict_entry'] == eid and rec['semantic'] == sem_part:
                    cats[rec['gt_part']] += 1
            cat_str = ", ".join(f"{k}:{v}" for k, v in
                                sorted(cats.items(), key=lambda x: -x[1])[:4])
            print(f"      Entry #{eid:4d}: {cnt:4d} tokens [{cat_str}]")

    # =========================================================
    # Analysis 3: Per-image part decomposition
    # =========================================================
    print(f"\n{'='*60}")
    print("Analysis 3: Single-image Part Decomposition")
    print(f"{'='*60}")
    print("(Do different registers of the same image go to different parts?)")

    # Group by image
    img_records = defaultdict(list)
    for rec in all_records:
        img_records[rec['image']].append(rec)

    decomposition_scores = []
    for img_file, recs in img_records.items():
        parts_found = set(r['semantic'] for r in recs if r['semantic'] != 'Background')
        fg_count = sum(1 for r in recs if r['semantic'] != 'Background')
        if fg_count > 0:
            decomposition_scores.append(len(parts_found))

    if decomposition_scores:
        avg_parts = np.mean(decomposition_scores)
        print(f"  Avg semantic parts discovered per image: {avg_parts:.1f}")
        print(f"  Distribution: {dict(sorted(defaultdict(int, zip(*np.unique(decomposition_scores, return_counts=True))).items()))}")

        # Show a few example images
        print(f"\n  Example decompositions:")
        examples = list(img_records.items())[:5]
        for img_file, recs in examples:
            part_counts = defaultdict(int)
            for r in recs:
                part_counts[r['semantic']] += 1
            parts_str = ", ".join(f"{k}:{v}" for k, v in
                                  sorted(part_counts.items(), key=lambda x: -x[1]))
            print(f"    {img_file}: {parts_str}")

    # =========================================================
    # Save results
    # =========================================================
    results = {
        'avg_purity': float(avg_purity) if purities else 0,
        'num_entries_evaluated': len(purities),
        'entry_results': entry_results[:200],
        'semantic_distribution': {
            sem: dict(entries) for sem, entries in semantic_to_entries.items()
        },
        'avg_parts_per_image': float(avg_parts) if decomposition_scores else 0,
        'vote_metric': vote_metric,
        'match_mode': match_mode,
    }
    # Add suffixes to output paths
    metric_suffix = f"_{vote_metric}" if vote_metric != 'attention' else ""
    if match_mode == 'patch':
        metric_suffix += "_patch"
    out_path = os.path.join(args.feature_dir, f"eval_partimagenet_K{args.K}{metric_suffix}.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


    # =========================================================
    # Visualization: top entry per GT label
    # =========================================================
    print(f"\n{'='*60}")
    print("Generating per-image part-decomposition visualizations...")
    print(f"{'='*60}")

    vis_dir = os.path.join(args.feature_dir, f"vis_partimagenet_K{args.K}{metric_suffix}")
    os.makedirs(vis_dir, exist_ok=True)

    # Cache for maps to avoid re-extracting same image
    cache = {}

    def get_maps(img_file, register_idx, entry_id=None):
        """Get score maps and GT mask.

        Register mode: returns per-register attention/similarity maps.
        Patch mode: returns per-patch cosine similarity to the given codebook entry.
        """
        if match_mode == 'patch':
            # Patch mode: compute per-patch similarity to given entry
            if img_file not in cache:
                img_path = os.path.join(img_dir, img_file)
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                _, h, w = img_tensor.shape
                h = h - h % args.patch_size
                w = w - w % args.patch_size
                img_tensor = img_tensor[:, :h, :w]

                with torch.no_grad():
                    x_in = img_tensor.unsqueeze(0).to(device)
                    x = model.prepare_tokens(x_in)
                    for blk in model.blocks:
                        x = blk(x)
                    x = model.norm(x)

                patch_features = x[0, num_reg:, :].cpu().numpy()  # [N_patch, C]
                h_feat = h // args.patch_size
                w_feat = w // args.patch_size
                # L2 normalize patch features
                pf_norm = patch_features / (
                    np.linalg.norm(patch_features, axis=1, keepdims=True) + 1e-8)

                mask_file = os.path.splitext(img_file)[0] + '.png'
                mask_path_file = os.path.join(ann_dir, mask_file)
                gt_mask = np.array(Image.open(mask_path_file))

                cache[img_file] = {
                    'pf_norm': pf_norm,  # [N_patch, C]
                    'h_feat': h_feat,
                    'w_feat': w_feat,
                    'gt_mask': gt_mask,
                }

            data = cache[img_file]
            gt_mask = data['gt_mask']
            if entry_id is not None:
                # Cosine similarity of each patch to this codebook entry
                sim_map = data['pf_norm'] @ dictionary_norm[entry_id]  # [N_patch]
                sim_map = sim_map.reshape(data['h_feat'], data['w_feat'])
            else:
                sim_map = np.zeros((data['h_feat'], data['w_feat']), dtype=np.float32)
            return sim_map, sim_map, gt_mask
        else:
            # Register mode: per-register attention/similarity maps
            if img_file not in cache:
                img_path = os.path.join(img_dir, img_file)
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                _, h, w = img_tensor.shape
                h = h - h % args.patch_size
                w = w - w % args.patch_size
                img_tensor = img_tensor[:, :h, :w]

                _, attn_maps, similarity_maps = extract_features_single(
                    model, img_tensor, device)

                mask_file = os.path.splitext(img_file)[0] + '.png'
                mask_path_file = os.path.join(ann_dir, mask_file)
                gt_mask = np.array(Image.open(mask_path_file))

                cache[img_file] = {
                    'attn_maps': attn_maps,
                    'similarity_maps': similarity_maps,
                    'gt_mask': gt_mask,
                }

            data = cache[img_file]
            return (data['attn_maps'][register_idx].numpy(),
                    data['similarity_maps'][register_idx].numpy(),
                    data['gt_mask'])

    # Group foreground images by object category (Car, Bird, Quad, ...), richest first,
    # then keep only quadrupeds (cats / dogs / …), birds, and a few cars; skip the rest.
    def img_category(recs):
        cats = defaultdict(int)
        for r in recs:
            if r['semantic'] != 'Background':
                cats[r['gt_part'].split('_')[0]] += 1
        return max(cats, key=cats.get) if cats else None

    by_cat = defaultdict(list)
    for f, recs in img_records.items():
        cat = img_category(recs)
        if cat:
            n_parts = len(set(r['semantic'] for r in recs if r['semantic'] != 'Background'))
            by_cat[cat].append((f, recs, n_parts))
    for cat in by_cat:
        by_cat[cat].sort(key=lambda x: -x[2])

    # Quadrupeds (cats / dogs / …) and birds are the main show; add a few cars. Others skipped.
    N_VIS = int(getattr(args, 'num_vis_images', 40))
    N_CAR = int(getattr(args, 'num_vis_cars', 6))
    cars = by_cat.get('Car', [])[:N_CAR]
    animal_cats = ['Quad', 'Bird']
    animals, depth, budget = [], 0, max(0, N_VIS - len(cars))
    while len(animals) < budget and any(depth < len(by_cat.get(c, [])) for c in animal_cats):
        for c in animal_cats:
            lst = by_cat.get(c, [])
            if depth < len(lst) and len(animals) < budget:
                animals.append(lst[depth])
        depth += 1
    selected = [(f, recs) for f, recs, _ in animals + cars]
    print(f"Rendering {len(animals)} quad/bird + {len(cars)} car images -> {vis_dir}")

    def threshold_normalize(m, percentile=75):
        """Threshold at percentile, zero below, re-normalize to [0,1] (suppress background)."""
        thresh = np.percentile(m, percentile)
        m = np.where(m >= thresh, m, 0.0)
        mmax = m.max()
        if mmax > 0:
            m = m / mmax
        return m

    part_order = ['Head', 'Body', 'Foot/Tier', 'Tail', 'Wing/Fin']
    saved = 0
    for img_file, recs in selected:
        try:
            img = Image.open(os.path.join(img_dir, img_file)).convert('RGB').resize((256, 256))
            img_np = np.array(img)
            mask_file = os.path.splitext(img_file)[0] + '.png'
            gt_mask = np.array(Image.open(os.path.join(ann_dir, mask_file)))
            gt_mask_resized = np.array(Image.fromarray(gt_mask).resize((256, 256), Image.NEAREST))
            gt_parts = sorted({PART_NAMES.get(int(l), '') for l in np.unique(gt_mask)} - {'', 'Background'})

            best_per_part = {}
            for rec in recs:
                sem = rec['semantic']
                if sem == 'Background':
                    continue
                if sem not in best_per_part or rec['concentration'] > best_per_part[sem]['concentration']:
                    best_per_part[sem] = rec
            present = [p for p in part_order if p in best_per_part]
            if not present:
                continue

            ncol = 2 + len(present)
            fig, axes = plt.subplots(1, ncol, figsize=(ncol * 3, 3.4))
            axes[0].imshow(img_np); axes[0].set_title(os.path.basename(img_file), fontsize=7); axes[0].axis('off')
            axes[1].imshow(gt_mask_resized, cmap='tab20'); axes[1].set_title(f"GT: {', '.join(gt_parts)}", fontsize=6); axes[1].axis('off')
            for k, part in enumerate(present):
                rec = best_per_part[part]
                attn_map, sim_map, _ = get_maps(rec['image'], rec['register'], rec.get('dict_entry'))
                score_map = sim_map if vote_metric == 'similarity' else attn_map
                score_resized = np.array(Image.fromarray(score_map.astype(np.float32)).resize((256, 256), Image.BILINEAR))
                score_resized = threshold_normalize(score_resized)
                ax = axes[2 + k]
                ax.imshow(img_np)
                ax.imshow(score_resized, alpha=0.6, cmap='hot', vmin=0, vmax=1)
                ax.set_title(f"{part}", fontsize=8); ax.axis('off')
            plt.suptitle("Original  |  GT part mask  |  per-part register heatmap (thresholded)", fontsize=9)
            plt.tight_layout(rect=[0, 0, 1, 0.92])
            safe = os.path.splitext(os.path.basename(img_file))[0]
            plt.savefig(os.path.join(vis_dir, f"{safe}.png"), dpi=140, bbox_inches='tight')
            plt.close()
            saved += 1
        except Exception:
            continue

    cache.clear()
    print(f"Saved {saved} per-image visualizations: {vis_dir}")


# ============================================================================
# Step 6: Nearest Neighbor Retrieval
# ============================================================================

def retrieve_new_image(args):
    """
    Step 6: For a new image, extract register features and find nearest
    dictionary entries. Visualize the assignments.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import normalize

    print("\n" + "=" * 60)
    print(f"Step 6: Nearest Neighbor Retrieval")
    print("=" * 60)

    # Load model
    model, device = load_model(args)

    # Load dictionary
    dictionary = np.load(os.path.join(args.feature_dir, f"dictionary_K{args.K}.npy"))
    with open(os.path.join(args.feature_dir, "config.json")) as f:
        config = json.load(f)

    # Load PCA if exists
    pca_model = None
    pca_path = os.path.join(args.feature_dir, f"pca_model_K{args.K}.pkl")
    if os.path.exists(pca_path):
        import pickle
        with open(pca_path, 'rb') as f:
            pca_model = pickle.load(f)

    # Process new image
    image_size = tuple(config['image_size'])
    transform = get_transform(image_size, args.patch_size)

    img = Image.open(args.image_path).convert('RGB')
    img_tensor = transform(img)
    _, h, w = img_tensor.shape
    h = h - h % args.patch_size
    w = w - w % args.patch_size
    img_tensor = img_tensor[:, :h, :w]

    reg_features, attn_maps, similarity_maps = extract_features_single(model, img_tensor, device)

    # Preprocess same as training
    reg_features_np = reg_features.numpy()
    reg_features_np = normalize(reg_features_np, norm='l2')
    if pca_model:
        reg_features_np = pca_model.transform(reg_features_np)

    # Find nearest dictionary entries
    # [N_reg] assignments
    dists = np.linalg.norm(
        reg_features_np[:, None, :] - dictionary[None, :, :], axis=2)  # [N_reg, K]
    new_assignments = np.argmin(dists, axis=1)  # [N_reg]
    min_dists = dists[np.arange(len(new_assignments)), new_assignments]

    print(f"\nImage: {args.image_path}")
    print(f"Register assignments:")
    unique_entries = np.unique(new_assignments)
    entry_counts = defaultdict(list)
    for r, (entry, dist) in enumerate(zip(new_assignments, min_dists)):
        entry_counts[entry].append((r, dist))

    for entry in sorted(entry_counts.keys()):
        regs = entry_counts[entry]
        reg_str = ", ".join([f"r{r}" for r, _ in regs])
        avg_dist = np.mean([d for _, d in regs])
        print(f"  Entry #{entry:4d}: {len(regs)} registers [{reg_str}], avg_dist={avg_dist:.4f}")

    print(f"\nUnique entries used: {len(unique_entries)}/{model.num_register_tokens}")

    # Visualize: Group by GT label from PartImageNet
    # Only show registers assigned to labeled entries
    num_reg = model.num_register_tokens

    img_display = img.resize((256, 256))
    img_np = np.array(img_display)

    # Load entry → GT label mapping from eval_partimagenet
    entry_to_gtlabel = {}
    eval_result_path = os.path.join(args.feature_dir, f"eval_partimagenet_K{args.K}.json")

    if os.path.exists(eval_result_path):
        try:
            with open(eval_result_path, 'r') as f:
                eval_results = json.load(f)
            # Build entry → label mapping (only for non-background)
            for entry_data in eval_results.get('entry_results', []):
                entry_id = entry_data['entry_id']
                dominant_part = entry_data['dominant_part']
                purity = entry_data['purity']

                # Only include non-background labels
                if dominant_part != 'Background':
                    entry_to_gtlabel[entry_id] = (dominant_part, purity)

            print(f"Loaded {len(entry_to_gtlabel)} labeled entries from PartImageNet")
        except Exception as e:
            print(f"Warning: Could not load entry labels: {e}")

    # Build GT label → entries mapping
    gtlabel_to_entries = defaultdict(list)
    for entry_id, (gtlabel, purity) in entry_to_gtlabel.items():
        gtlabel_to_entries[gtlabel].append({
            'entry_id': entry_id,
            'purity': purity,
        })

    print(f"\nGT labels and their entries:")
    for gtlabel, entries in sorted(gtlabel_to_entries.items()):
        entry_ids = [e['entry_id'] for e in entries]
        print(f"  {gtlabel}: {len(entries)} entries {entry_ids[:5]}{'...' if len(entry_ids) > 5 else ''}")

    # Re-assign registers to GT labels (not individual entries)
    # For each register, compute average distance to all entries of each GT label
    reg_norms = np.linalg.norm(reg_features_np, axis=1, keepdims=True)
    reg_features_norm = reg_features_np / (reg_norms + 1e-8)  # Add epsilon to avoid division by zero

    dict_norms = np.linalg.norm(dictionary, axis=1, keepdims=True)
    dict_norm = dictionary / (dict_norms + 1e-8)

    gtlabel_assignments = []
    gtlabel_distances = []
    gtlabel_best_entries = []  # Track which entry was closest

    for r in range(num_reg):
        reg_feat = reg_features_norm[r:r+1]

        # Compute distances to all entries
        dists_to_all = np.linalg.norm(dict_norm - reg_feat, axis=1)

        # For each GT label, find the minimum distance to any of its entries
        best_gtlabel = None
        best_min_dist = float('inf')
        best_entry_id = None

        for gtlabel, entries in gtlabel_to_entries.items():
            entry_ids = [e['entry_id'] for e in entries]
            dists_to_label = dists_to_all[entry_ids]
            min_dist = np.min(dists_to_label)  # Minimum distance to any entry of this label
            min_idx = np.argmin(dists_to_label)

            if min_dist < best_min_dist:
                best_min_dist = min_dist
                best_gtlabel = gtlabel
                best_entry_id = entry_ids[min_idx]  # Track closest entry

        gtlabel_assignments.append(best_gtlabel)
        gtlabel_distances.append(best_min_dist)
        gtlabel_best_entries.append(best_entry_id)

    # Filter by distance threshold
    dist_threshold = np.percentile(gtlabel_distances, 70)
    print(f"\nDistance threshold (70th percentile): {dist_threshold:.3f}")

    # Group registers by GT label
    gtlabel_groups = defaultdict(list)
    filtered_by_distance = 0

    for r, (gtlabel, dist, entry_id) in enumerate(zip(gtlabel_assignments, gtlabel_distances, gtlabel_best_entries)):
        if dist > dist_threshold:
            filtered_by_distance += 1
            continue

        if gtlabel:
            purity = entry_to_gtlabel[entry_id][1]
            gtlabel_groups[gtlabel].append({
                'reg_idx': r,
                'dist': dist,
                'best_entry': entry_id,  # Closest entry for this register
                'purity': purity,
            })

    print(f"Filtered out {filtered_by_distance} registers with distance > {dist_threshold:.3f}")

    # Sort GT labels by number of registers
    sorted_gtlabels = sorted(gtlabel_groups.items(), key=lambda x: -len(x[1]))
    best_labels = [lbl for lbl, regs in sorted_gtlabels if len(regs) >= 3][:8]

    if not best_labels:
        best_labels = [lbl for lbl, _ in sorted_gtlabels[:8]]

    print(f"\nGT labels to visualize:")
    for gtlabel in best_labels:
        regs = gtlabel_groups[gtlabel]
        entries_used = set(r['best_entry'] for r in regs)
        print(f"  {gtlabel}: {len(regs)} registers (using {len(entries_used)} entries)")

    # Visualization grid
    num_rows = len(best_labels)
    num_cols = 4

    if num_rows == 0:
        print("\nNo labeled entries found for this image's registers.")
        return

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 3.5 * num_rows))
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    for row_idx, gtlabel in enumerate(best_labels):
        regs = gtlabel_groups[gtlabel]
        regs_sorted = sorted(regs, key=lambda x: x['dist'])[:3]

        # Column 0: GT label info with original image
        ax = axes[row_idx, 0]
        ax.imshow(img_np)

        # Get entry statistics
        entries_used = set(r['best_entry'] for r in regs)
        avg_purity = np.mean([r['purity'] for r in regs])

        ax.set_title(f"{gtlabel}\n{len(regs)} regs, {len(entries_used)} entries\navg_pur={avg_purity:.1%}",
                    fontsize=9, fontweight='bold')
        ax.axis('off')

        # Columns 1-3: Show top 3 closest registers' attention maps
        for col_idx, reg_data in enumerate(regs_sorted, start=1):
            reg_idx = reg_data['reg_idx']
            dist = reg_data['dist']
            best_entry = reg_data['best_entry']

            attn_map = attn_maps[reg_idx].numpy()
            attn_resized = np.array(
                Image.fromarray(attn_map.astype(np.float32)).resize((256, 256), Image.BILINEAR))

            ax = axes[row_idx, col_idx]
            ax.imshow(img_np)
            ax.imshow(attn_resized, alpha=0.6, cmap='hot')
            ax.set_title(f"reg{reg_idx}\nE#{best_entry}, d={dist:.3f}",
                        fontsize=7)
            ax.axis('off')

    plt.suptitle(f"Registers Grouped by PartImageNet GT Label\n"
                f"({os.path.basename(args.image_path)})",
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    vis_dir = os.path.join(args.feature_dir, f"vis_K{args.K}")
    os.makedirs(vis_dir, exist_ok=True)
    out_path = os.path.join(vis_dir, f"retrieve_{os.path.splitext(os.path.basename(args.image_path))[0]}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Learned Dictionary Evaluation (dino_attn_base_dict)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Common args
    def add_common_args(p):
        p.add_argument('--arch', type=str, default='region_vit_small',
                        help='Model arch: region_vit_small / region_vit_base / ...')
        p.add_argument('--patch_size', type=int, default=16)
        p.add_argument('--num_register_tokens', type=int, default=96)
        # Dict model params
        p.add_argument('--dict_size', type=int, default=2048,
                        help='Dictionary/codebook size in the model (0 = no dict)')
        p.add_argument('--dict_tau', type=float, default=0.1)
        p.add_argument('--top_n', type=int, default=6)
        p.add_argument('--spatial_sigma', type=float, default=0.8)
        p.add_argument('--mask_tau', type=float, default=0.1)
        p.add_argument('--num_reg_for_loss_per_head', type=int, default=16)

    def add_model_args(p):
        add_common_args(p)
        p.add_argument('--pretrained_weights', type=str, required=True)
        p.add_argument('--codebook', type=str, default=None,
                        help='Optional codebook-only .pth to inject (pairs a backbone with a dict codebook)')
        p.add_argument('--image_size', type=int, nargs='+', default=[224, 224])

    def add_cluster_args(p):
        p.add_argument('--feature_dir', type=str, required=True,
                        help='Directory containing extracted features')
        p.add_argument('--K', type=int, default=2048, help='Number of dictionary entries')

    # === extract ===
    p_extract = subparsers.add_parser('extract', help='Step 1: Extract features with optional top-K filtering')
    add_model_args(p_extract)
    p_extract.add_argument('--imagenet_dir', type=str, required=True,
                            help='Path to ImageNet val directory')
    p_extract.add_argument('--output_dir', type=str, default='./kmeans_output_topk')
    p_extract.add_argument('--max_images', type=int, default=1000,
                            help='Max images to process (0=all, default=1000)')
    p_extract.add_argument('--topk_registers', type=int, default=0,
                            help='Number of top registers to select per image (0=all, e.g., 32 out of 96)')
    p_extract.add_argument('--topk_method', type=str, default='variance',
                            choices=['variance', 'max', 'mean', 'entropy', 'sparsity', 'diversity'],
                            help='Method to select top registers (variance=most discriminative, max=highest peak)')
    p_extract.add_argument('--score_type', type=str, default='similarity',
                            choices=['attention', 'similarity'],
                            help='Score type for selection: similarity (feature-based, recommended for clustering) or attention (attention weights)')

    # === cluster ===
    p_cluster = subparsers.add_parser('cluster',
                                       help='Step 2-3: Extract codebook + PartImageNet GT matching')
    add_model_args(p_cluster)
    p_cluster.add_argument('--output_dir', type=str, default='./eval_dict/output_dict2048',
                            help='Output directory for dictionary and metadata')
    p_cluster.add_argument('--K', type=int, default=2048, help='Number of dictionary entries (should match dict_size)')
    p_cluster.add_argument('--K_init', type=int, default=0,
                            help='Initial K for over-clustering (only used in K-means fallback)')
    p_cluster.add_argument('--pca_dim', type=int, default=0,
                            help='PCA dimension (0=no PCA)')
    p_cluster.add_argument('--n_init', type=int, default=3,
                            help='Number of K-means initializations (fallback only)')
    p_cluster.add_argument('--refine_kmeans', action='store_true',
                            help='If set, use codebook as K-means init and refine with data')
    p_cluster.add_argument('--match_mode', type=str, default='register',
                            choices=['register', 'patch'],
                            help='Feature matching mode: register (attention+density voting) or patch (direct spatial GT lookup)')
    p_cluster.add_argument('--partimagenet_dir', type=str, required=True,
                            help='Path to PartImageNet root')
    p_cluster.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    p_cluster.add_argument('--max_images', type=int, default=0,
                            help='Max images to process (0=all)')
    p_cluster.add_argument('--vote_metric', type=str, default='attention',
                            choices=['attention', 'similarity'],
                            help='Metric for weighted voting')
    p_cluster.add_argument('--concentration_thresh', type=float, default=0.5,
                            help='Min concentration to keep a register (0-1, default 0.5)')
    p_cluster.add_argument('--topk_registers', type=int, default=0,
                            help='Top-K registers per image (0=all)')
    p_cluster.add_argument('--topk_method', type=str, default='max',
                            choices=['variance', 'max', 'mean', 'entropy', 'sparsity', 'diversity'],
                            help='Method for Top-K register selection')
    p_cluster.add_argument('--score_type', type=str, default='similarity',
                            choices=['attention', 'similarity'],
                            help='Score type for Top-K selection')

    # === visualize ===
    p_vis = subparsers.add_parser('visualize', help='Step 4: Visualize dictionary')
    add_cluster_args(p_vis)
    p_vis.add_argument('--topk', type=int, default=10,
                        help='Top-k most typical examples per entry')
    p_vis.add_argument('--num_entries', type=int, default=50,
                        help='Number of entries to visualize')
    p_vis.add_argument('--entry_ids', type=int, nargs='*', default=None,
                        help='Specific entry IDs to visualize')

    # === analyze ===
    p_analyze = subparsers.add_parser('analyze', help='Step 5: Quantitative analysis')
    add_cluster_args(p_analyze)

    # === eval_parts ===
    p_eval = subparsers.add_parser('eval_parts',
                                    help='Step 5b: PartImageNet part-level evaluation')
    add_model_args(p_eval)
    add_cluster_args(p_eval)
    p_eval.add_argument('--partimagenet_dir', type=str, required=True,
                         help='Path to PartImageNet root (containing images/ and annotations/)')
    p_eval.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    p_eval.add_argument('--max_images', type=int, default=0,
                         help='Max images to process (0=all)')
    p_eval.add_argument('--vote_metric', type=str, default='attention',
                         choices=['attention', 'similarity'],
                         help='Metric for weighted voting: attention (L->N) or similarity (cosine)')
    p_eval.add_argument('--match_mode', type=str, default='register',
                         choices=['register', 'patch'],
                         help='Feature matching mode: register (attention+density voting) or patch (per-part mean pool + cosine sim)')
    # === retrieve ===
    p_retrieve = subparsers.add_parser('retrieve', help='Step 6: Retrieve for new image')
    add_model_args(p_retrieve)
    add_cluster_args(p_retrieve)
    p_retrieve.add_argument('--image_path', type=str, required=True)

    # === all ===
    p_all = subparsers.add_parser('all', help='Run full pipeline with top-K filtering')
    add_model_args(p_all)
    p_all.add_argument('--imagenet_dir', type=str, required=True)
    p_all.add_argument('--output_dir', type=str, default='./eval_dict/output_dict2048')
    p_all.add_argument('--max_images', type=int, default=1000)
    p_all.add_argument('--K', type=int, default=2048)
    p_all.add_argument('--pca_dim', type=int, default=0)
    p_all.add_argument('--n_init', type=int, default=3)
    p_all.add_argument('--refine_kmeans', action='store_true',
                        help='If set, use codebook as K-means init and refine with data')
    p_all.add_argument('--topk', type=int, default=10)
    p_all.add_argument('--num_entries', type=int, default=50)
    p_all.add_argument('--entry_ids', type=int, nargs='*', default=None)
    p_all.add_argument('--retrieve_image', type=str, default=None,
                        help='Optional: image for retrieval test')
    p_all.add_argument('--topk_registers', type=int, default=0,
                        help='Number of top registers to select per image (0=all)')
    p_all.add_argument('--topk_method', type=str, default='variance',
                        choices=['variance', 'max', 'mean', 'entropy', 'sparsity', 'diversity'],
                        help='Method to select top registers')
    p_all.add_argument('--score_type', type=str, default='similarity',
                        choices=['attention', 'similarity'],
                        help='Score type for selection')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == 'extract':
        extract_all_features(args)

    elif args.command == 'cluster':
        cluster_features(args)

    elif args.command == 'visualize':
        visualize_dictionary(args)

    elif args.command == 'analyze':
        analyze_dictionary(args)

    elif args.command == 'eval_parts':
        eval_partimagenet(args)

    elif args.command == 'retrieve':
        retrieve_new_image(args)

    elif args.command == 'all':
        # Run full pipeline
        args.feature_dir = args.output_dir
        print("=" * 60)
        print("Running Full Pipeline")
        print("=" * 60)

        # Step 1: Extract
        extract_all_features(args)

        # Step 2-3: Cluster
        args.feature_dir = args.output_dir
        cluster_features(args)

        # Step 4: Visualize
        visualize_dictionary(args)

        # Step 5: Analyze
        analyze_dictionary(args)

        # Step 6: Retrieve (optional)
        if args.retrieve_image:
            args.image_path = args.retrieve_image
            retrieve_new_image(args)

        print("\n" + "=" * 60)
        print("Full pipeline complete!")
        print(f"Results: {args.output_dir}")
        print("=" * 60)


if __name__ == '__main__':
    main()
