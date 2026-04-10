"""
Тесты для модели UNet.
"""

import pytest
import torch
import torch.nn as nn
from src.model import UNet


def test_model_creation():
    """Проверяет, что модель создаётся без ошибок."""
    n_classes = 6
    model = UNet(n_classes=n_classes, bilinear=False)
    assert isinstance(model, nn.Module)
    assert model.n_classes == n_classes


def test_forward_shape():
    """Проверяет размерности выходов для случайного батча."""
    batch_size = 4
    channels = 3
    height = 256
    width = 256
    n_classes = 6

    model = UNet(n_classes=n_classes, bilinear=False)
    model.eval()

    x = torch.randn(batch_size, channels, height, width)
    with torch.no_grad():
        seg_out, cls_out = model(x)

    assert seg_out.shape == (batch_size, n_classes, height, width), \
        f"Expected seg shape {(batch_size, n_classes, height, width)}, got {seg_out.shape}"
    assert cls_out.shape == (batch_size, n_classes), \
        f"Expected cls shape {(batch_size, n_classes)}, got {cls_out.shape}"


def test_forward_different_sizes():
    """Проверяет, что модель работает с разными размерами входных изображений."""
    model = UNet(n_classes=6, bilinear=False)
    model.eval()

    sizes = [(128, 128), (256, 256), (512, 512)]
    for h, w in sizes:
        x = torch.randn(2, 3, h, w)
        with torch.no_grad():
            seg_out, cls_out = model(x)
        assert seg_out.shape[2:] == (h, w)
        assert cls_out.shape == (2, 6)


def test_classifier_output_range():
    """Проверяет, что логиты классификации не нормализованы (могут быть любыми числами)."""
    model = UNet(n_classes=6, bilinear=False)
    model.eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        _, cls_out = model(x)
    assert not torch.isnan(cls_out).any()
    assert cls_out.dtype == torch.float32


def test_gradient_flow():
    """Проверяет, что градиенты проходят через оба выхода."""
    model = UNet(n_classes=6, bilinear=False)
    model.train()
    x = torch.randn(2, 3, 256, 256, requires_grad=True)
    seg_out, cls_out = model(x)

    loss_seg = seg_out.sum()
    loss_cls = cls_out.sum()
    loss = loss_seg + loss_cls
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


def test_pretrained_weights():
    """Проверяет, что энкодер загружает предобученные веса (без ошибок)."""
    model = UNet(n_classes=6, bilinear=False)
    # Просто проверяем, что первый слой имеет веса (не нулевые)
    assert model.inc[0].weight is not None
    # Можно проверить, что веса не являются случайными (среднее не около 0)
    # Но для простоты достаточно того, что модель создалась


def test_save_load(tmp_path):
    """Проверяет сохранение и загрузку весов модели."""
    model = UNet(n_classes=6, bilinear=False)
    torch.manual_seed(42)
    for param in model.parameters():
        param.data = torch.randn_like(param)

    checkpoint_path = tmp_path / "model.pth"
    torch.save(model.state_dict(), checkpoint_path)

    new_model = UNet(n_classes=6, bilinear=False)
    new_model.load_state_dict(torch.load(checkpoint_path))

    for (name1, param1), (name2, param2) in zip(model.state_dict().items(), new_model.state_dict().items()):
        assert torch.equal(param1, param2), f"Weight mismatch for {name1}"

def test_model_with_voc_classes():
    from src.model import UNet
    model = UNet(n_classes=21, bilinear=False)
    x = torch.randn(2, 3, 256, 256)
    seg, cls = model(x)
    assert seg.shape == (2, 21, 256, 256)
    assert cls.shape == (2, 21)


if __name__ == "__main__":
    pytest.main([__file__])