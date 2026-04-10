"""
config.py — управление конфигурациями проекта.
Поддерживает загрузку из YAML, переопределение через CLI и валидацию.
"""

import yaml
import argparse
from pathlib import Path
from typing import Any, Dict, Optional
import torch


class Config:
    """
    Класс для хранения и доступа к конфигурационным параметрам.
    Поддерживает вложенные параметры через точечную нотацию.
    """

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = {}
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # Рекурсивно превращаем в Config
                self._config[key] = Config(value)
            else:
                self._config[key] = value

    def __getattr__(self, name: str) -> Any:
        """Позволяет обращаться к параметрам как config.param_name."""
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"Config has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Обработка установки атрибутов (чтобы не конфликтовать с _config)."""
        if name == '_config':
            super().__setattr__(name, value)
        else:
            self._config[name] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Безопасное получение значения."""
        return self._config.get(key, default)

    def update(self, updates: Dict[str, Any]) -> None:
        """Обновление конфигурации (например, из CLI)."""
        for key, value in updates.items():
            self._config[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Возвращает словарь (рекурсивно)."""
        result = {}
        for key, value in self._config.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_config(config_path: str) -> Config:
    """
    Загружает YAML-файл и возвращает объект Config.
    :param config_path: путь к YAML-файлу.
    :return: Config объект.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def merge_cli_args(config: Config, args: argparse.Namespace) -> Config:
    """
    Объединяет параметры из командной строки с конфигом.
    Параметры CLI имеют приоритет.
    :param config: исходный объект Config.
    :param args: argparse.Namespace с аргументами CLI.
    :return: обновлённый объект Config.
    """
    updates = {}
    for key, value in vars(args).items():
        if value is not None:
            updates[key] = value
    config.update(updates)

    return config


def create_default_config() -> Config:
    """
    Создаёт конфигурацию по умолчанию (для быстрого старта).
    """
    default_dict = {
        'data': {
            'train_images': '/path/to/bdd100k_seg/images/train',
            'train_labels': '/path/to/bdd100k_seg/labels/train',
            'val_images': '/path/to/bdd100k_seg/images/val',
            'val_labels': '/path/to/bdd100k_seg/labels/val',
            'img_height': 512,
            'img_width': 512,
            'batch_size': 8,
            'num_workers': 4,
        },
        'model': {
            'num_classes': 6,  # фон + 5 классов транспорта
            'bilinear': False,
        },
        'train': {
            'epochs': 20,
            'lr': 1e-4,
            'weight_decay': 1e-5,
            'seg_weight': 1.0,
            'cls_weight': 0.5,
            'checkpoint_dir': './checkpoints',
            'log_dir': './logs',
            'resume': None,
        },
        'evaluate': {
            'model_checkpoint': './checkpoints/best_model.pth',
            'output_dir': './evaluation_results',
            'num_samples': 10,
        },
        'device': 'auto',
    }
    return Config(default_dict)


def get_device(config: Config) -> torch.device:
    device_str = config.get('device', 'auto')
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    return device


if __name__ == '__main__':
    # Тест
    cfg = create_default_config()
    print(cfg.train.checkpoint_dir)  # должно вывести ./checkpoints
    print(cfg.to_dict())