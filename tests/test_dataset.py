import pytest
import numpy as np
import torch
import cv2
import tempfile
from pathlib import Path

def test_voc_dataset_mock():
    """Проверяет загрузку VOC2012Dataset на синтетических данных."""
    from src.dataset import VOC2012Dataset   # импортируем новый класс

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / 'VOC2012'
        # Создаём минимальную структуру папок VOC2012
        (root / 'JPEGImages').mkdir(parents=True)
        (root / 'SegmentationClass').mkdir()
        (root / 'ImageSets' / 'Segmentation').mkdir(parents=True)

        # Генерируем одно тестовое изображение
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Маска: фон (0) и прямоугольник класса 7 (car в VOC)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 20:40] = 7

        cv2.imwrite(str(root / 'JPEGImages' / 'test.jpg'), img)
        # Для упрощения сохраняем маску как одноканальную (в реальности нужна палитра)
        cv2.imwrite(str(root / 'SegmentationClass' / 'test.png'), mask)

        # Создаём train.txt
        with open(root / 'ImageSets' / 'Segmentation' / 'train.txt', 'w') as f:
            f.write('test\n')

        # Создаём датасет с целевым размером 64x64
        dataset = VOC2012Dataset(
            root=str(root),
            split='train',
            target_size=(64, 64)
        )
        assert len(dataset) == 1
        img_t, mask_t = dataset[0]

        # Проверяем размеры и типы
        assert img_t.shape == (3, 64, 64)
        assert mask_t.shape == (64, 64)
        assert mask_t.dtype == torch.long

        # Проверяем, что класс 7 присутствует в маске
        assert 7 in torch.unique(mask_t)