"""
RATS part-whole probe.
Reuses draw_dominant_map.py's model loading + feature extraction, then:
  1) shows what each of the 12 HEADS captures (head-level heatmaps + argmax map)  -> "whole" level
  2) for a chosen head, shows its 16 REGISTERS' heatmaps + argmax map             -> "part" level
register r belongs to head r // (num_reg//num_heads).
"""
import os, sys, argparse, types
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import draw_dominant_map as ddm

NUM_REG = 192
NUM_HEADS = 12
REGS_PER_HEAD = NUM_REG // NUM_HEADS

def build_args():
    a = types.SimpleNamespace()
    a.model_type = 'dino_attn_base'; a.model_dir = '.'; a.arch = 'region_vit_base'
    a.patch_size = 16; a.num_register_tokens = NUM_REG; a.use_cross_attn = True
    a.num_reg_for_loss_per_head = 16; a.slot_attn_iterations = 3
    a.pretrained_weights = 'checkpoint_teacher.pth'; a.checkpoint_key = 'teacher'
    return a

def get_summary(img_path, size, need_attn=True):
    args = build_args()
    model, device = ddm.load_model(args)
    img_tensor, pil, (h, w) = ddm.load_image(img_path, [size, size], args.patch_size)
    hf, wf = h // args.patch_size, w // args.patch_size

    if not need_attn:
        # lightweight: only grab last block's output tokens (no NxN attn matrix -> OK at hi-res)
        cap = {}
        def hook(m, i, o):
            cap['t'] = (o[0] if isinstance(o, tuple) else o).detach()
        hh = model.blocks[-1].register_forward_hook(hook)
        with torch.no_grad():
            model(img_tensor.to(device))
        hh.remove()
        tok = cap['t']  # [B, N_total, C]
        reg = tok[:, :NUM_REG].cpu()
        patch = tok[:, NUM_REG:].cpu()
        summary = {'register_features': reg.unsqueeze(0), 'patch_features': patch.unsqueeze(0)}
        return summary, pil, (hf, wf)

    extractor = ddm.LayerFeatureExtractor(model, model_type='dino_attn_base')
    layer_data = extractor.extract(img_tensor.to(device))
    summary = ddm.extract_register_and_patch_features(layer_data, num_register_tokens=NUM_REG)
    for k in ('register_features', 'patch_features', 'attention_weights'):
        if isinstance(summary.get(k), torch.Tensor):
            summary[k] = summary[k].cpu()
    return summary, pil, (hf, wf)

def to_grid(scores_1d, hf, wf):
    return np.asarray(scores_1d, dtype=np.float32).reshape(hf, wf)

def overlay_argmax(pil, label_grid, ncolors, title, out, valid_mask=None):
    base = np.asarray(pil.resize((label_grid.shape[1]*16, label_grid.shape[0]*16))).astype(np.float32)/255.
    colors = np.array(ddm.get_distinct_colors(ncolors))
    lab_up = np.kron(label_grid, np.ones((16,16), dtype=int))
    cmap = colors[lab_up % ncolors]
    if valid_mask is not None:
        vm = np.kron(valid_mask, np.ones((16,16))).astype(bool)
        cmap[~vm] = base[~vm]
    blend = 0.55*cmap + 0.45*base
    fig, ax = plt.subplots(1, 2, figsize=(12,6))
    ax[0].imshow(base); ax[0].set_title('original'); ax[0].axis('off')
    ax[1].imshow(blend); ax[1].set_title(title); ax[1].axis('off')
    plt.tight_layout(); plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()

def heat_grid(pil, score_rows, hf, wf, titles, out, cols=4):
    n = len(score_rows); rows = (n + cols - 1)//cols
    base = np.asarray(pil.resize((wf*16, hf*16))).astype(np.float32)/255.
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = np.array(axes).reshape(-1)
    for i in range(len(axes)):
        axes[i].axis('off')
        if i < n:
            g = to_grid(score_rows[i], hf, wf)
            g = (g - g.min())/(g.max()-g.min()+1e-8)
            gup = np.kron(g, np.ones((16,16)))
            axes[i].imshow(base); axes[i].imshow(gup, cmap='jet', alpha=0.55)
            axes[i].set_title(titles[i], fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches='tight'); plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--out_prefix', required=True)
    p.add_argument('--head', type=int, default=-1, help='which head to expand into parts; -1 = auto')
    args = p.parse_args()

    summary, pil, (hf, wf) = get_summary(args.image, args.size)
    L = hf*wf

    head_scores = ddm.get_head_level_scores(summary, -1, NUM_REG, NUM_HEADS, metric='similarity')  # [12, L]
    all_scores  = ddm.get_all_scores_per_head(summary, -1, NUM_REG, NUM_HEADS, metric='similarity') # [192, L]

    # remove per-patch common-mode so each head/register shows its RELATIVE selectivity
    head_rel = head_scores - head_scores.mean(axis=0, keepdims=True)
    all_rel  = all_scores  - all_scores.mean(axis=0, keepdims=True)

    # ---- WHOLE level: relative argmax over heads ----
    whole = head_rel.argmax(axis=0).reshape(hf, wf)
    overlay_argmax(pil, whole, NUM_HEADS, 'WHOLE: dominant HEAD (relative, 12)', f'{args.out_prefix}_whole_heads.png')
    heat_grid(pil, list(head_rel), hf, wf, [f'head {h}' for h in range(NUM_HEADS)],
              f'{args.out_prefix}_head_heatmaps.png', cols=4)

    # ---- PART level: relative argmax over ALL 192 registers (each patch -> its most selective register) ----
    part_all = all_rel.argmax(axis=0).reshape(hf, wf)
    overlay_argmax(pil, part_all, NUM_REG, 'PARTS: dominant REGISTER over all 192 (relative)',
                   f'{args.out_prefix}_parts_all192.png')

    # ---- choose object head = head with strongest relative peak ----
    if args.head >= 0:
        obj_head = args.head
    else:
        obj_head = int(head_rel.max(axis=1).argmax())
    print(f'object head = {obj_head}')

    # ---- PART level within obj_head's 16 registers (relative competition) ----
    r0, r1 = obj_head*REGS_PER_HEAD, (obj_head+1)*REGS_PER_HEAD
    head_regs = all_scores[r0:r1]                       # [16, L]
    head_regs_rel = head_regs - head_regs.mean(axis=0, keepdims=True)
    part_lab = head_regs_rel.argmax(axis=0).reshape(hf, wf)
    fg = (head_rel.argmax(axis=0).reshape(hf, wf) == obj_head)
    overlay_argmax(pil, part_lab, REGS_PER_HEAD,
                   f'PARTS: 16 registers of head {obj_head} (masked)',
                   f'{args.out_prefix}_parts_head{obj_head}.png', valid_mask=fg)
    overlay_argmax(pil, part_lab, REGS_PER_HEAD,
                   f'PARTS: 16 registers of head {obj_head} (full)',
                   f'{args.out_prefix}_parts_head{obj_head}_full.png')
    heat_grid(pil, list(head_regs_rel), hf, wf, [f'h{obj_head} reg{r0+i}' for i in range(REGS_PER_HEAD)],
              f'{args.out_prefix}_parts_head{obj_head}_heatmaps.png', cols=4)
    print('saved', args.out_prefix)

if __name__ == '__main__':
    main()
