# Part Retrieval with the Learned Dictionary (PartImageNet)

RATS trains a small **register dictionary** (a codebook, via `run_train_dict.sh`).
Each codebook entry is a proto-semantic **part prototype**. This module matches those
entries against **PartImageNet** part annotations and shows that a single entry captures
the same part (e.g. *Head*) **across very different object categories** — birds, dogs,
snakes, cars… — with high part purity.

Pipeline: `cluster` (build the entry↔part dictionary on PartImageNet *train*) →
`eval_parts` (measure part purity on *val*) → visualizations.

## 1. Weights (auto-downloaded)

`run_base_dict4096.sh` pulls both pieces from the Hub and stitches them at load time —
you don't need to download anything manually:

| piece | Hub path (`TimingYang/RATS`) | size |
|-------|------------------------------|------|
| backbone | `base/checkpoint_teacher_ep0388.pth` | ~436 MB |
| dict codebook | `dict/codebook_base_dict4096.pth` | ~13 MB |

To use your own local dict checkpoint instead, set `CKPT=/path/to/checkpoint.pth`
and drop `--codebook` from the script.

## 2. Data — PartImageNet

Download the **PartImageNet segmentation** set (per-image part masks) — hosted on the
Hub at [`turkeyju/PartImageNet`](https://huggingface.co/datasets/turkeyju/PartImageNet)
(official mirror of <https://github.com/TACJu/PartImageNet>):

```bash
# from the repo root — puts it exactly where the scripts look
mkdir -p data && cd data
wget https://huggingface.co/datasets/turkeyju/PartImageNet/resolve/main/PartImageNet_Seg.zip
unzip -q PartImageNet_Seg.zip -x "__MACOSX/*"   # extracts to ./PartImageNet/
cd ..
```

Expected layout (`PARTIMAGENET_ROOT`, default `./data/PartImageNet`):

```
data/PartImageNet/
├── images/
│   ├── train/        # *.JPEG
│   └── val/
└── annotations/
    ├── train/        # per-image part-segmentation PNG masks
    └── val/
```

To keep it elsewhere, point the script at it: `PARTIMAGENET_ROOT=/path/to/PartImageNet bash retrieval_learnable/run_base_dict4096.sh`.

## 3. Run

From the repo root:

```bash
bash retrieval_learnable/run_base_dict4096.sh                 # full run (uses all images)
MAX_IMAGES=1000 bash retrieval_learnable/run_base_dict4096.sh # smaller / faster
```

Environment overrides:

| var | default | meaning |
|-----|---------|---------|
| `PARTIMAGENET_ROOT` | `./data/PartImageNet` | dataset location |
| `MAX_IMAGES` | `20000` | cap on images used for cluster / eval |
| `CKPT` | *(auto: ep0388 from Hub)* | backbone checkpoint |
| `CODEBOOK` | *(auto: codebook from Hub)* | dict codebook to stitch on |

Patch-matching variant: edit the script's `SCRIPT=demo_topk.py` → `demo_topk_patch.py`.

## 4. Outputs

Everything lands in `output/retrieval_base_dict4096/`:

```
dictionary_K4096.npy              # the codebook used as the dictionary
entry_labels_K4096.json           # each entry → its PartImageNet part label + purity
eval_partimagenet_K4096.json      # part purity, #parts/image, semantic distribution
vis_partimagenet_K4096/           # one PNG per image: original | GT mask | per-part heatmaps
```

Each `vis_partimagenet_K4096/<image>.png` decomposes one image into its parts:
`Original | GT part mask | Head | Body | Foot/Tier | Tail | Wing/Fin` — one clean
(thresholded) register heatmap per semantic part present, not a blended composite.
Images shown are quadrupeds (cats / dogs / …) and birds, plus a few cars
(`N_VIS`=40, `N_CAR`=6 in `demo_topk.py`); other categories are skipped.

## Files

- `demo_topk.py` — the pipeline (`cluster` / `eval_parts` / `visualize` / `retrieve`); register-vote matching.
- `demo_topk_patch.py` — same, but patch-level matching.
- `run_base_dict4096.sh` — end-to-end driver (region_vit_base, dict K=4096).
