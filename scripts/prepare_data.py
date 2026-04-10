#!/usr/bin/env python3
"""
scripts/prepare_data.py

Скрипт для подготовки данных BDD100K (семантическая сегментация) к обучению.
Выполняет проверку структуры, создаёт списки пар изображение-маска и выводит статистику.
"""

import os
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Подготовка данных BDD100K для обучения')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Корневая папка, содержащая bdd100k_seg (например, data/bdd100k_seg)')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Директория для сохранения списков файлов (по умолчанию data)')
    parser.add_argument('--check_classes', action='store_true',
                        help='Проверить распределение классов в масках (может быть медленно)')
    return parser.parse_args()

def main():
    args = parse_args()

    # Определяем пути
    data_root = Path(args.data_root)
    seg_dir = data_root / 'bdd100k_seg'
    if not seg_dir.exists():
        # Возможно, структура уже такая, что bdd100k_seg находится прямо в data_root
        # Проверим
        if (data_root / 'images').exists() and (data_root / 'labels').exists():
            seg_dir = data_root
        else:
            raise FileNotFoundError(f"Папка bdd100k_seg не найдена в {data_root}")

    splits = ['train', 'val']
    for split in splits:
        img_dir = seg_dir / 'images' / split
        lbl_dir = seg_dir / 'labels' / split
        if not img_dir.exists() or not lbl_dir.exists():
            print(f"Предупреждение: не найдены папки для split {split} (img: {img_dir}, lbl: {lbl_dir})")
            continue

        # Поиск всех изображений
        img_paths = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.jpeg'))
        pairs = []
        missing = []
        for img_path in tqdm(img_paths, desc=f'Проверка {split}'):
            mask_path = lbl_dir / (img_path.stem + '.png')
            if mask_path.exists():
                pairs.append((str(img_path), str(mask_path)))
            else:
                missing.append(img_path.name)

        print(f"\n{split.upper()}: найдено {len(pairs)} пар изображение-маска")
        if missing:
            print(f"  Отсутствуют маски для {len(missing)} изображений, например: {missing[:5]}")
        else:
            print("  Все маски на месте.")

        # Сохраняем список пар в файл
        out_file = Path(args.output_dir) / f"{split}_pairs.txt"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, 'w') as f:
            for img_path, mask_path in pairs:
                f.write(f"{img_path}\t{mask_path}\n")
        print(f"  Список сохранён в {out_file}")

        # Опционально: статистика по классам
        if args.check_classes and pairs:
            print(f"  Анализ распределения классов в масках (может занять время)...")
            class_counts = defaultdict(int)
            for _, mask_path in tqdm(pairs[:100], desc='Анализ масок (первые 100)'):  # ограничим для скорости
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                unique, counts = np.unique(mask, return_counts=True)
                for cls, cnt in zip(unique, counts):
                    class_counts[int(cls)] += cnt
            print("  Распределение классов в масках (первые 100):")
            for cls, cnt in sorted(class_counts.items()):
                print(f"    класс {cls}: {cnt} пикселей")

    print("\nГотово. Теперь можно запускать обучение, указав в конфиге пути к данным.")


if __name__ == '__main__':
    main()