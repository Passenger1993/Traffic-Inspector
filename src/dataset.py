import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import albumentations as A
from pathlib import Path
from typing import Dict, Optional, Tuple

# -----------------------------------------------------------------------------
class VOC2012Dataset(Dataset):
    """Датасет VOC2012 для семантической сегментации.
    Ожидается структура:
        root/
          JPEGImages/       # все изображения
          SegmentationClass/# маски в формате палитры
          ImageSets/Segmentation/
              train.txt     # список файлов для обучения
              val.txt       # список файлов для валидации
    """
    def __init__(self, root: str, split: str = 'train',
                 transform: Optional[A.Compose] = None,
                 normalize: Optional[callable] = None,
                 target_size: Tuple[int, int] = (512, 512)):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.normalize = normalize
        self.target_size = target_size

        # Читаем список имён файлов (без расширения)
        split_file = self.root / 'ImageSets' / 'Segmentation' / f'{split}.txt'
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]

        self.image_paths = [self.root / 'JPEGImages' / f'{img_id}.jpg' for img_id in self.image_ids]
        self.mask_paths = [self.root / 'SegmentationClass' / f'{img_id}.png' for img_id in self.image_ids]

        # Количество классов VOC2012: 20 объектов + фон = 21
        self.num_classes = 21
        print(f"[VOC2012Dataset] Загружено {len(self.image_paths)} пар для split='{split}'")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Загрузка изображения
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Загрузка маски (формат палитры)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)  # (H, W, 3) или (H, W)
        # Преобразование палитры в индексы классов (0..20)
        mask = self._palette_to_indices(mask)

        # Resize до целевого размера (если нужно)
        if self.target_size:
            image = cv2.resize(image, (self.target_size[1], self.target_size[0]),
                               interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]),
                              interpolation=cv2.INTER_NEAREST)

        # Аугментации (если есть)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Преобразование в тензоры
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()

        if self.normalize:
            image = self.normalize(image)

        return image, mask

    @staticmethod
    def _palette_to_indices(mask: np.ndarray) -> np.ndarray:
        """Преобразует маску в формате палитры (RGB) в маску индексов (0..20)."""
        # VOC2012 использует стандартную палитру: для каждого индекса свой RGB
        # Здесь можно использовать предопределённый словарь или просто взять первый канал
        # (В официальных масках все три канала одинаковы, поэтому можно взять первый)
        if mask.ndim == 3:
            # Если маска трёхканальная, то индексы закодированы в первом канале? Нет.
            # В VOC2012 палитра: разные RGB для каждого класса. Нужно сопоставить RGB -> индекс.
            # Упрощённый способ: использовать готовую функцию из библиотеки torchvision или реализовать свою.
            # Для простоты приведём пример с использованием предустановленной палитры:
            voc_palette = [
                0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128,
                0, 128, 128, 128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0,
                64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0,
                0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128
            ]
            # Создаём массив индексов
            indices = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
            for i in range(21):
                r, g, b = voc_palette[i*3:(i+1)*3]
                indices[(mask[:,:,0] == r) & (mask[:,:,1] == g) & (mask[:,:,2] == b)] = i
            return indices
        else:
            # Если маска уже одноканальная (иногда бывает)
            return mask

# -----------------------------------------------------------------------------
def get_train_transform(img_height: int, img_width: int) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(
            size=(img_height, img_width),   # всегда приводит к целевому размеру
            scale=(0.5, 1.0),
            p=1.0,                          # всегда применяется
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST
        ),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5,
                 interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_NEAREST),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
    ])

def get_val_transform(img_height: int, img_width: int) -> A.Compose:
    return A.Compose([
        A.Resize(size=(img_height, img_width),
                 interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
    ])
def get_normalize_transform():
    from torchvision.transforms import Normalize
    return Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])