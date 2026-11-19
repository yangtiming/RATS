import torch
import sys

ckpt_path = "/path/to/checkpoint.pth"
out_path = "/path/to/checkpoint_teacher.pth"

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
teacher = ckpt["teacher"]
torch.save({"teacher": teacher}, out_path)
print(f"Saved teacher-only checkpoint to {out_path}")
print(f"Original keys: {list(ckpt.keys())}")
