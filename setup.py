from pathlib import Path
from setuptools import setup, find_packages


BASE_DIR = Path(__file__).parent
README = (BASE_DIR / "README.md").read_text(encoding="utf-8")

setup(
    name="lpc",
    version="1.1.0",
    description="Training framework to reproduce latent point collapse experiments in deep classifiers",
    long_description=README,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.5",
        "torchvision>=0.20",
        "numpy>=1.26",
        "scikit-learn>=1.6",
        "scipy>=1.15",
        "faiss-gpu>=1.7",
        "PyYAML>=6.0",
    ],
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="latent point collapse, neural collapse, deep learning, distributed training",
)
