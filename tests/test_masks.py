import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Используем бэкенд без GUI, чтобы избежать ошибок Tkinter
import matplotlib.pyplot as plt
from PIL import Image

# ============================================
# 1. Укажите путь к распакованному датасету
# ============================================
VOC_ROOT = 'data/VOCdevkit/VOC2012'  # измените при необходимости

# ============================================
# 2. Проверка структуры папок
# ============================================
required_dirs = ['JPEGImages', 'SegmentationClass', 'ImageSets/Segmentation']
for d in required_dirs:
    path = os.path.join(VOC_ROOT, d)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Папка не найдена: {path}")
    print(f"✓ Найдена папка: {path}")

# ============================================
# 3. Загрузка списка ID из trainval.txt
# ============================================
trainval_file = os.path.join(VOC_ROOT, 'ImageSets', 'Segmentation', 'trainval.txt')
if not os.path.exists(trainval_file):
    raise FileNotFoundError(f"Не найден файл {trainval_file}")

with open(trainval_file) as f:
    image_ids = [line.strip() for line in f.readlines()]
print(f"Количество ID в trainval.txt: {len(image_ids)}")

img_dir = os.path.join(VOC_ROOT, 'JPEGImages')
mask_dir = os.path.join(VOC_ROOT, 'SegmentationClass')

# Проверяем, что для всех ID есть изображения и маски
missing_img = []
missing_mask = []
for img_id in image_ids:
    img_path = os.path.join(img_dir, img_id + '.jpg')
    mask_path = os.path.join(mask_dir, img_id + '.png')
    if not os.path.exists(img_path):
        missing_img.append(img_id)
    if not os.path.exists(mask_path):
        missing_mask.append(img_id)

print(f"Из них отсутствуют изображения: {len(missing_img)}")
print(f"Из них отсутствуют маски: {len(missing_mask)}")

if missing_img:
    print(f"Примеры пропущенных изображений: {missing_img[:5]}")
if missing_mask:
    print(f"Примеры пропущенных масок: {missing_mask[:5]}")

# Оставляем только те ID, для которых есть и изображение, и маска
valid_ids = [img_id for img_id in image_ids
             if img_id not in missing_img and img_id not in missing_mask]
print(f"Доступно для использования: {len(valid_ids)} пар")

if not valid_ids:
    raise ValueError("Нет ни одной корректной пары изображение-маска")

# ============================================
# 4. Функция для поиска образца с автомобильными классами
# ============================================
def find_image_with_classes(class_ids, max_attempts=100):
    """Ищет случайный ID, в маске которого есть хотя бы один из указанных индексов."""
    attempts = 0
    while attempts < max_attempts:
        img_id = random.choice(valid_ids)
        mask_path = os.path.join(mask_dir, img_id + '.png')
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        unique = np.unique(mask)
        if any(cls in unique for cls in class_ids):
            return img_id, mask
        attempts += 1
    raise ValueError(f"Не найден образец с классами {class_ids} за {max_attempts} попыток")

# Автомобильные классы в VOC: 7=car, 6=bus, 2=bicycle, 14=motorbike
car_classes = {7, 6, 2, 14}
img_id, mask = find_image_with_classes(car_classes)
print(f"\nВыбран образец с автомобильным классом: {img_id}")

# ============================================
# 5. Загрузка изображения и маски
# ============================================
img_path = os.path.join(img_dir, img_id + '.jpg')
mask_path = os.path.join(mask_dir, img_id + '.png')

image_pil = Image.open(img_path)
mask_pil = Image.open(mask_path)

mask_np = np.array(mask_pil)
unique_values = np.unique(mask_np)
print(f"Уникальные значения в маске (индексы классов): {unique_values}")

# ============================================
# 6. Визуализация
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(image_pil)
axes[0].set_title(f"Изображение: {img_id}")
axes[0].axis('off')

# Для отображения маски в цветах VOC преобразуем в RGB
mask_display = np.array(mask_pil.convert('RGB'))
axes[1].imshow(mask_display)
axes[1].set_title("Маска сегментации (VOC цвета)")
axes[1].axis('off')

plt.tight_layout()
plt.savefig('sample_voc.png')  # сохраняем в файл
print("Визуализация сохранена как sample_voc.png")
plt.show()  # может открыть окно, но из-за бэкенда Agg не должно

# ============================================
# 7. Классы VOC (для справки)
# ============================================
voc_classes = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}

print("\nНайденные классы в выбранной маске:")
for val in unique_values:
    if val in voc_classes:
        print(f"  {val}: {voc_classes[val]}")
    else:
        print(f"  {val}: неизвестный класс")