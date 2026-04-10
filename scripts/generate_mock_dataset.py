import os
import numpy as np
import cv2
from pathlib import Path

def generate_mock_dataset(root_dir=None, num_train=10, num_val=5, img_size=(100,100)):
    """
    Создаёт поддельные изображения и маски для тестирования.
    Изображения: случайный шум.
    Маски: случайный прямоугольник с классом 13 (car) в центре.
    """
    if root_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        root_dir = project_root / "data" / "mock"
    else:
        root_dir = Path(root_dir)

    root = Path(root_dir)
    for split in ['train', 'val']:
        num = num_train if split == 'train' else num_val
        img_dir = root / 'images' / split
        mask_dir = root / 'labels' / split
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num):
            # Изображение (случайный шум)
            img = np.random.randint(0, 255, (*img_size, 3), dtype=np.uint8)
            # Маска (фон 0, прямоугольник с классом 13)
            mask = np.zeros(img_size, dtype=np.uint8)
            # Рисуем прямоугольник 30x30 в центре
            h, w = img_size
            y0, y1 = h//2 - 15, h//2 + 15
            x0, x1 = w//2 - 15, w//2 + 15
            mask[y0:y1, x0:x1] = 13  # класс car

            # Сохраняем
            fname = f"{i+1:04d}.jpg"
            cv2.imwrite(str(img_dir / fname), img)
            cv2.imwrite(str(mask_dir / fname.replace('.jpg', '.png')), mask)

    print(f"Mock dataset created at {root}")

if __name__ == "__main__":
    generate_mock_dataset(num_train=10, num_val=5, img_size=(100,100))