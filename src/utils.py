import torch
import torch.nn as nn
import numpy as np

class CombinedLoss(nn.Module):
    def __init__(self, seg_weight=1.0, cls_weight=0.5):
        super().__init__()
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.cls_loss = nn.CrossEntropyLoss()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight

    def forward(self, seg_pred, seg_target, cls_pred, cls_target):
        seg_l = self.seg_loss(seg_pred, seg_target)
        cls_l = self.cls_loss(cls_pred, cls_target)
        return self.seg_weight * seg_l + self.cls_weight * cls_l, seg_l, cls_l


def iou_score(pred, target, num_classes, ignore_index=0):
    pred = pred.argmax(dim=1)
    ious = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())
    return np.nanmean(ious)


def get_class_target(mask):
    """Возвращает доминирующий класс (не фон) для каждого изображения в батче."""
    batch_size = mask.size(0)
    targets = torch.zeros(batch_size, dtype=torch.long, device=mask.device)
    for i in range(batch_size):
        unique, counts = torch.unique(mask[i], return_counts=True)
        # Исключаем фон (0)
        if len(unique) == 1 and unique[0] == 0:
            targets[i] = 0
        else:
            max_count = 0
            best_class = 0
            for cls, cnt in zip(unique, counts):
                if cls != 0 and cnt > max_count:
                    max_count = cnt
                    best_class = cls
            targets[i] = best_class
    return targets


VOC_COLORS = [
    (0, 0, 0),       # background
    (128, 0, 0),     # aeroplane
    (0, 128, 0),     # bicycle
    (128, 128, 0),   # bird
    (0, 0, 128),     # boat
    (128, 0, 128),   # bottle
    (0, 128, 128),   # bus
    (128, 128, 128), # car
    (64, 0, 0),      # cat
    (192, 0, 0),     # chair
    (64, 128, 0),    # cow
    (192, 128, 0),   # diningtable
    (64, 0, 128),    # dog
    (192, 0, 128),   # horse
    (64, 128, 128),  # motorbike
    (192, 128, 128), # person
    (0, 64, 0),      # pottedplant
    (128, 64, 0),    # sheep
    (0, 192, 0),     # sofa
    (128, 192, 0),   # train
    (0, 64, 128)     # tvmonitor
]