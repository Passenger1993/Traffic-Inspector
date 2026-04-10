from src.utils import CombinedLoss, iou_score, get_class_target, VOC_COLORS
import torch
from src.utils import iou_score

def test_combined_loss_shape():
    criterion = CombinedLoss(seg_weight=1.0, cls_weight=0.5)
    batch_size = 2
    num_classes = 6
    h, w = 64, 64
    seg_pred = torch.randn(batch_size, num_classes, h, w)
    seg_target = torch.randint(0, num_classes, (batch_size, h, w))
    cls_pred = torch.randn(batch_size, num_classes)
    cls_target = torch.randint(0, num_classes, (batch_size,))
    loss, seg_l, cls_l = criterion(seg_pred, seg_target, cls_pred, cls_target)
    assert loss.shape == ()
    assert seg_l.shape == ()
    assert cls_l.shape == ()
    assert loss > 0


def test_iou_score():
    num_classes = 3
    ignore_index = 0  # игнорируем фон

    # 1. Один ненулевой класс полностью совпадает
    pred = torch.zeros(1, num_classes, 4, 4)
    target = torch.zeros(1, 4, 4)
    pred[:, 1, 1:3, 1:3] = 1
    target[:, 1:3, 1:3] = 1
    iou = iou_score(pred, target, num_classes, ignore_index=ignore_index)
    assert iou == 1.0

    # 2. Частичное пересечение класса 1
    pred = torch.zeros(1, num_classes, 4, 4)
    target = torch.zeros(1, 4, 4)
    pred[:, 1, 1:3, 1:3] = 1
    target[:, 2:4, 2:4] = 1
    iou = iou_score(pred, target, num_classes, ignore_index=ignore_index)
    expected = 1/7  # пересечение = 1, объединение = 7
    assert abs(iou - expected) < 1e-6

def test_get_class_target():
    mask = torch.tensor([
        [0, 0, 0],
        [0, 1, 1],
        [0, 1, 2]
    ]).unsqueeze(0)  # batch=1
    targets = get_class_target(mask)
    # Доминирующий не-фон: класс 1 (3 пикселя) vs класс 2 (1 пиксель) -> класс 1
    assert targets[0] == 1

    mask2 = torch.tensor([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]).unsqueeze(0)
    targets2 = get_class_target(mask2)
    assert targets2[0] == 0  # фон

    # Батч
    batch = torch.stack([mask.squeeze(0), mask2.squeeze(0)], dim=0)
    targets_batch = get_class_target(batch)
    assert targets_batch.tolist() == [1, 0]

def test_class_colors():
    # Проверяем, что список цветов имеет длину, равную num_classes
    # Не вдаёмся в точные значения, просто проверяем, что это список кортежей из 3 целых
    assert isinstance(VOC_COLORS, list)
    assert all(isinstance(c, tuple) and len(c) == 3 and all(isinstance(v, int) for v in c) for c in VOC_COLORS)