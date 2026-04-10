import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # добавляем корень в путь

from src.dataset import BDD100KDataset, get_normalize_transform

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ds = BDD100KDataset(
    images_dir=os.path.join(BASE_DIR, "data/mock/images/train"),
    labels_dir=os.path.join(BASE_DIR, "data/mock/labels/train"),
    target_height=64,
    target_width=64,
    transform=None,
    normalize=get_normalize_transform()
)

print(f"Dataset size: {len(ds)}")
img, mask = ds[0]
print(f"Image shape: {img.shape}, dtype: {img.dtype}")
print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
print(f"Unique mask values: {torch.unique(mask)}")