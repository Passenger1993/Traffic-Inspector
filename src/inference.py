import argparse
import json
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config, get_device
from src.dataset import VOC2012Dataset, get_normalize_transform
from src.model import UNet
from src.utils import iou_score, get_class_target, VOC_COLORS

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    device = get_device(config)
    print(f"Using device: {device}")

    # Создаём выходную папку
    out_dir = Path(config.inference.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем валидационный датасет
    normalize = get_normalize_transform()
    val_dataset = VOC2012Dataset(
        root=config.data.root,
        split='val',                       # или config.data.split_val
        transform=None,
        normalize=normalize,
        target_size=(config.data.img_height, config.data.img_width)
    )
    val_loader = DataLoader(val_dataset, batch_size=config.data.batch_size, shuffle=False,
                            num_workers=config.data.num_workers, pin_memory=True)

    # Загружаем модель
    checkpoint = torch.load(config.inference.model_checkpoint, map_location=device, weights_only=False)
    model = UNet(n_classes=config.model.num_classes, bilinear=config.model.bilinear).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from {config.inference.model_checkpoint}")

    total_iou = 0.0
    total_acc = 0.0
    total_samples = 0
    visual_samples = []

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc='Evaluating'):
            images = images.to(device)
            masks = masks.to(device)
            cls_targets = get_class_target(masks)

            seg_pred, cls_pred = model(images)
            seg_labels = seg_pred.argmax(dim=1)

            iou = iou_score(seg_pred, masks, config.model.num_classes)
            cls_acc = (cls_pred.argmax(dim=1) == cls_targets).float().mean().item()

            total_iou += iou * images.size(0)
            total_acc += cls_acc * images.size(0)
            total_samples += images.size(0)

            # Собираем визуализации
            if len(visual_samples) < config.inference.num_samples:
                for i in range(images.size(0)):
                    if len(visual_samples) >= config.inference.num_samples:
                        break
                    visual_samples.append({
                        'image': images[i].cpu(),
                        'mask_gt': masks[i].cpu(),
                        'mask_pred': seg_labels[i].cpu(),
                        'cls_gt': cls_targets[i].item(),
                        'cls_pred': cls_pred.argmax(dim=1)[i].item()
                    })

    avg_iou = total_iou / total_samples
    avg_acc = total_acc / total_samples
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Classification Accuracy: {avg_acc:.4f}")

    # Сохраняем метрики
    metrics = {
        'avg_iou': avg_iou,
        'cls_accuracy': avg_acc,
        'num_samples': total_samples
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Визуализация
    for idx, sample in enumerate(visual_samples):
        img = sample['image'].numpy().transpose(1,2,0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        mask_gt = sample['mask_gt'].numpy()
        mask_pred = sample['mask_pred'].numpy()

        gt_rgb = np.zeros((*mask_gt.shape, 3), dtype=np.uint8)
        pred_rgb = np.zeros((*mask_pred.shape, 3), dtype=np.uint8)
        for cls, col in enumerate(VOC_COLORS):
            gt_rgb[mask_gt == cls] = col
            pred_rgb[mask_pred == cls] = col

        overlay = img * 0.5 + pred_rgb / 255.0 * 0.5

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img)
        axes[0].set_title('Image')
        axes[0].axis('off')
        axes[1].imshow(gt_rgb)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        axes[2].imshow(overlay)
        axes[2].set_title(f'Prediction (cls: {sample["cls_pred"]}, gt: {sample["cls_gt"]})')
        axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(out_dir / f'result_{idx}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization {idx+1}/{len(visual_samples)}")

    print("Evaluation finished.")


if __name__ == '__main__':
    main()