# Пишем
from src.dataset import VOC2012Dataset
import argparse
import os
import json
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config, merge_cli_args, get_device
from src.dataset import VOC2012Dataset, get_train_transform, get_normalize_transform
from src.model import UNet
from src.utils import CombinedLoss, iou_score, get_class_target

def parse_args():
    parser = argparse.ArgumentParser(description='Обучение модели сегментации')
    parser.add_argument('--config', type=str, required=True, help='Путь к YAML конфигурации')
    # Опциональные аргументы для переопределения
    parser.add_argument('--epochs', type=int, help='Количество эпох')
    parser.add_argument('--batch_size', type=int, help='Размер батча')
    parser.add_argument('--lr', type=float, help='Начальная скорость обучения')
    parser.add_argument('--device', type=str, help='Устройство (cuda/cpu/auto)')
    parser.add_argument('--resume', type=str, help='Путь к чекпоинту для возобновления')
    # Можно добавить и другие параметры по необходимости
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    config = merge_cli_args(config, args)

    config.train.lr = float(config.train.lr)
    config.train.weight_decay = float(config.train.weight_decay)
    config.train.seg_weight = float(config.train.seg_weight)
    config.train.cls_weight = float(config.train.cls_weight)
    config.train.epochs = int(config.train.epochs)

    config.data.img_height = int(config.data.img_height)
    config.data.img_width = int(config.data.img_width)
    config.data.batch_size = int(config.data.batch_size)
    config.data.num_workers = int(config.data.num_workers)

    config.model.num_classes = int(config.model.num_classes)


    device = get_device(config)
    print(f"Using device: {device}")

    # Создаём директории
    Path(config.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.train.log_dir).mkdir(parents=True, exist_ok=True)

    # Подготовка датасетов
    train_transform = get_train_transform(config.data.img_height, config.data.img_width)
    normalize = get_normalize_transform()

    train_dataset = VOC2012Dataset(
    root=config.data.root,
    split='train',                     # или config.data.split
    transform=get_train_transform(config.data.img_height, config.data.img_width),
    normalize=normalize,
    target_size=(config.data.img_height, config.data.img_width)
    )

    val_dataset = VOC2012Dataset(
        root=config.data.root,
        split='val',
        transform=None,
        normalize=normalize,
        target_size=(config.data.img_height, config.data.img_width)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.data.batch_size,
                              shuffle=True, num_workers=config.data.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.data.batch_size,
                            shuffle=False, num_workers=config.data.num_workers, pin_memory=True)

    # Инициализация модели
    model = UNet(n_classes=config.model.num_classes, bilinear=config.model.bilinear).to(device)

    # Функция потерь
    criterion = CombinedLoss(seg_weight=config.train.seg_weight, cls_weight=config.train.cls_weight)

    # Оптимизатор
    optimizer = optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Возобновление обучения
    start_epoch = 0
    best_val_iou = 0.0
    if config.train.resume:
        checkpoint = torch.load(config.train.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_iou = checkpoint.get('best_val_iou', 0.0)
        print(f"Resumed from {config.train.resume}, epoch {start_epoch}")

    metrics_log = []

    # Цикл обучения
    for epoch in range(start_epoch, config.train.epochs):
        print(f"\nEpoch {epoch+1}/{config.train.epochs}")

        # Обучение
        model.train()
        train_loss = 0.0
        train_seg_loss = 0.0
        train_cls_loss = 0.0
        train_bar = tqdm(train_loader, desc='Train')
        for images, masks in train_bar:
            images = images.to(device)
            masks = masks.to(device)
            cls_targets = get_class_target(masks)

            optimizer.zero_grad()
            seg_pred, cls_pred = model(images)
            loss, seg_l, cls_l = criterion(seg_pred, masks, cls_pred, cls_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_seg_loss += seg_l.item()
            train_cls_loss += cls_l.item()
            train_bar.set_postfix(loss=loss.item(), seg=seg_l.item(), cls=cls_l.item())

        avg_train_loss = train_loss / len(train_loader)
        avg_train_seg = train_seg_loss / len(train_loader)
        avg_train_cls = train_cls_loss / len(train_loader)

        # Валидация
        model.eval()
        val_loss = 0.0
        val_seg_loss = 0.0
        val_cls_loss = 0.0
        total_iou = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc='Val')
            for images, masks in val_bar:
                images = images.to(device)
                masks = masks.to(device)
                cls_targets = get_class_target(masks)

                seg_pred, cls_pred = model(images)
                loss, seg_l, cls_l = criterion(seg_pred, masks, cls_pred, cls_targets)

                val_loss += loss.item()
                val_seg_loss += seg_l.item()
                val_cls_loss += cls_l.item()

                iou = iou_score(seg_pred, masks, config.model.num_classes)
                total_iou += iou
                val_bar.set_postfix(loss=loss.item(), iou=iou)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_seg = val_seg_loss / len(val_loader)
        avg_val_cls = val_cls_loss / len(val_loader)
        avg_val_iou = total_iou / len(val_loader)

        print(f"Train Loss: {avg_train_loss:.4f} (seg: {avg_train_seg:.4f}, cls: {avg_train_cls:.4f})")
        print(f"Val Loss: {avg_val_loss:.4f} (seg: {avg_val_seg:.4f}, cls: {avg_val_cls:.4f}) | Val IoU: {avg_val_iou:.4f}")

        # Сохранение метрик
        metrics_log.append({
            'epoch': epoch+1,
            'train_loss': avg_train_loss,
            'train_seg_loss': avg_train_seg,
            'train_cls_loss': avg_train_cls,
            'val_loss': avg_val_loss,
            'val_seg_loss': avg_val_seg,
            'val_cls_loss': avg_val_cls,
            'val_iou': avg_val_iou
        })
        with open(os.path.join(config.train.log_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_log, f, indent=2)

        # Сохранение лучшей модели
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_iou,
                'num_classes': config.model.num_classes
            }, os.path.join(config.train.checkpoint_dir, 'best_model.pth'))
            print(f"New best model saved with IoU {best_val_iou:.4f}")

        # Сохранение промежуточного чекпоинта
        if (epoch+1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_iou,
                'num_classes': config.model.num_classes
            }, os.path.join(config.train.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        scheduler.step(avg_val_loss)

    print("Training finished.")


if __name__ == '__main__':
    main()