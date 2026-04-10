from setuptools import setup, find_packages

setup(
    name="vehicle_segmentation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
        "opencv-python",
        "albumentations",
        "pyyaml",
        "tqdm",
    ],
)