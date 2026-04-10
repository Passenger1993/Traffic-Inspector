import pytest
import tempfile
import yaml
from argparse import Namespace
from src.config import Config, load_config, merge_cli_args, create_default_config, get_device
import torch

def test_config_creation():
    data = {'a': 1, 'b': {'c': 2}}
    cfg = Config(data)
    assert cfg.a == 1
    assert cfg.b.c == 2
    assert isinstance(cfg.b, Config)

def test_config_update():
    cfg = Config({'x': 10})
    cfg.update({'x': 20, 'y': 30})
    assert cfg.x == 20
    assert cfg.y == 30

def test_config_to_dict():
    data = {'a': 1, 'b': {'c': 2}}
    cfg = Config(data)
    d = cfg.to_dict()
    assert d == data

def test_load_config(tmp_path):
    config_content = """
data:
  batch_size: 8
model:
  num_classes: 6
"""
    config_file = tmp_path / "test.yaml"
    config_file.write_text(config_content)
    cfg = load_config(str(config_file))
    assert cfg.data.batch_size == 8
    assert cfg.model.num_classes == 6

def test_merge_cli_args():
    cfg = Config({'batch_size': 8})
    args = Namespace(batch_size=16, epochs=None)
    cfg = merge_cli_args(cfg, args)
    assert cfg.batch_size == 16
    # Проверяем, что None не перезаписывает
    assert not hasattr(cfg, 'epochs')  # или assert cfg.epochs == None? Но в нашей реализации epochs не добавляется
    # В текущей реализации update добавляет только значения не None. Если epoch не был в конфиге, его не будет.
    # Проверим, что epoch не появился
    with pytest.raises(AttributeError):
        _ = cfg.epochs

def test_create_default_config():
    cfg = create_default_config()
    assert cfg.train.epochs == 20
    assert cfg.model.num_classes == 6
    assert cfg.device == 'auto'

def test_get_device():
    cfg = Config({'device': 'auto'})
    dev = get_device(cfg)
    # Не проверяем конкретное устройство, просто что возвращается torch.device
    assert isinstance(dev, torch.device)

    cfg = Config({'device': 'cpu'})
    dev = get_device(cfg)
    assert dev.type == 'cpu'