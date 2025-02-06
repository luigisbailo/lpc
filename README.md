# Latent Point Collapse  on a Low Dimensional Embedding in Deep Neural Network Classifiers

### Abstract
The configuration of latent representations plays a critical role in determining the performance of deep neural network classifiers. In particular, the emergence of well-separated class embeddings in the latent space has been shown to improve both generalization and robustness. In this paper, we propose a method to induce the collapse of latent representations belonging to the same class into a single point, which enhances class separability in the latent space while enforcing Lipschitz continuity in the network.
We demonstrate that this phenomenon, which we call \textit{latent point collapse}, is achieved by adding a strong $L_2$ penalty on the penultimate-layer representations and is the result of a push-pull tension developed with the cross-entropy loss function.
In addition, we show the practical utility of applying this compressing loss term to the latent representations of a low-dimensional linear penultimate layer.
The proposed approach is straightforward to implement and yields substantial improvements in discriminative feature embeddings, along with remarkable gains in robustness to input perturbations.

### Dependencies
- Python 3.10  
- PyTorch 2.5  
- NumPy 1.26  
- scikit-learn 1.6  
- SciPy 1.15  
- faiss-gpu 1.7  
- CUDA 11.8  

### Installation
Run the following command to install the package:

```bash
pip install .
```

### Reproducing Results
You can reproduce the results on a GPU cluster using the scripts provided in the `scripts` directory:

- `scripts/train_cifar10.sh`
- `scripts/train_cifar100.sh`

These scripts assume the existence of a conda environment named **lpc**, where all the dependencies listed above (including this package) are already installed. We used NVIDIA A100 GPUs, as indicated by the `_gres_` argument.

Running these scripts will:
1. Create a `dataset` directory where the CIFAR-10 or CIFAR-100 datasets are automatically downloaded.
2. Produce a `results` directory where all experiment outputs are stored.

To aggregate and select the best results (based on different initial learning rates), run:

- `scripts/best_results_cifar10.sh`
- `scripts/best_results_cifar100.sh`

These scripts gather the best-performing results across all experiments and models into a single `best_results.pkl` file in the `results` directory.