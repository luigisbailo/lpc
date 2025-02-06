from setuptools import setup, find_packages

setup(
    name='lpc',
    version='1.0.0',
    description='Train deep neural network classifiers to induce latent point collapse',
    author='Luigi Sbailò',
    author_email='luigi.sbailo@gmail.com',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'torch>=2.5',
        'numpy>=1.26',
        'scikit-learn>=1.6',
        'scipy>=1.15',
        'faiss-gpu>=1.7',
    ],
)
