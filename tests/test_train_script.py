import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

def test_train_help():
    train_script = ROOT_DIR / "src" / "train.py"   # изменено
    result = subprocess.run([sys.executable, str(train_script), '-h'],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert '--config' in result.stdout

def test_inference_help():
    inference_script = ROOT_DIR / "src" / "inference.py"   # изменено
    result = subprocess.run([sys.executable, str(inference_script), '-h'],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert '--config' in result.stdout