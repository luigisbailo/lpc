# Latent Point Collapse (LPC)

Latent Point Collapse (LPC) is the official implementation accompanying the paper *Latent Point Collapse on a Low Dimensional Embedding in Deep Neural Network Classifiers*. The package provides the exact training code and configuration files needed to reproduce experiments reported in the paper across CIFAR-10/100, ImageNet, and related settings.

## Abstract
The topological properties of latent representations play a critical role in determining the performance of deep neural network classifiers. In particular, the emergence of well-separated class embeddings in the latent space has been shown to improve both generalization and robustness. In this paper, we propose a method to induce the collapse of latent representations belonging to the same class into a single point, which enhances class separability in the latent space while making the network Lipschitz continuous.
We demonstrate that this phenomenon, which we call *latent point collapse* (LPC), is achieved by adding a strong L2 penalty on the penultimate-layer representations and is the result of a push-pull tension developed with the cross-entropy loss function.
In addition, we show the practical utility of applying this compressing loss term to the latent representations of a low-dimensional linear penultimate layer.
LPC can be viewed as a stronger manifestation of *neural collapse* (NC): while NC entails that within-class representations converge around their class means, LPC causes these representations to collapse in absolute value to a single point. As a result, the network improvements typically associated with NC—namely better generalization and robustness—are even more pronounced when LPC develops.

## Repository Structure

**Core package (`lpc/`)**
- `networks.py` – Neural network architectures (ResNet, WideResNet, MLP) with configurable penultimate layers for LPC training.
- `trainer.py` – Training orchestrator handling the loop, LR/gamma scheduling, evaluation, and checkpointing.
- `metrics.py` – Evaluation utilities for collapse diagnostics, entropy estimation, distance margins, DeepFool robustness, and Lipschitz analysis.
- `losses.py` – Auxiliary loss implementations (ArcFace, CosFace, supervised contrastive learning).
- `utils.py` – Helper functions for checkpoint I/O and data handling.

**Scripts and configuration**
- `scripts/main.py` – Primary entry point for training that parses configs, prepares datasets, builds models, and coordinates distributed execution.
- `configs/*.yml` – Experiment definitions (dataset, architecture, training hyperparameters) used to reproduce all paper results.
- `scripts/train_*.sh` – Convenience bash wrappers for launching the CIFAR-10, CIFAR-100, and ImageNet sweeps reported in the paper.
- `scripts/*_slurm_jobs.py` – SLURM job generators for running the experiment grid on cluster hardware.

**Execution flow**
1. **Configure** – Pick an experiment configuration from `configs/` (e.g., `cifar10.yml`).
2. **Launch** – Start training with `torchrun scripts/main.py` and any CLI overrides you require.
3. **Initialize** – The main script loads data, instantiates the selected backbone, and wraps it for distributed training.
4. **Train** – `Trainer` runs the loop with LPC gamma scheduling, logging metrics and saving checkpoints to `results_dir`.
5. **Evaluate** – Collapse metrics and accuracy statistics are written to pickle files for downstream analysis/visualisation.

## Configuration Files
Every configuration under `configs/` follows the same three-part structure:

```yaml
dataset:
    name: CIFAR10

architecture:
    backbone: ResNet
    backbone_model: 18
    hypers:
        penultimate_nodes: 64
        activation: SiLU

training:
    hypers:
        batch_size: 128
        lr: 0.001
        total_epochs: 1000
        logging: 50
        gamma: 0.0001
        gamma_scheduler_type: exponential
        ...
```

- The `dataset` section selects the torchvision dataset class. For CIFAR-style datasets the loader downloads automatically; for ImageNet the directory must contain the usual `train/` and `val/` folders.
- The `architecture` section chooses the backbone class from `lpc/networks.py` (`ResNet`, `WideResNet`, `MLPvanilla`, …) and its hyper-parameters. The `architecture_type` flag passed on the CLI must be compatible with the backbone (e.g. use `lin_pen` for linear penultimate layers).
- The `training.hypers` section contains optimiser, scheduler, and LPC gamma schedules. All values can be overridden from the CLI without editing the YAML file.

## Running Training
Training is launched through `torchrun` so the same command works for single- or multi-GPU runs. The mandatory arguments are the config file, the dataset directory, the results directory, and the architecture type.

Single GPU example:

```bash
torchrun --nproc_per_node=1 scripts/main.py \
    --config configs/cifar10.yml \
    --dataset-dir /path/to/data \
    --results-dir runs/cifar10_lin_pen \
    --architecture-type lin_pen \
    --l2-loss true
```

Multi-GPU example (2 GPUs on a single node):

```bash
torchrun --nproc_per_node=4 scripts/main.py \
    --config configs/cifar10.yml \
    --dataset-dir /path/to/data \
    --results-dir runs/cifar10_multi \
    --architecture-type lin_pen \
    --l2-loss true \
    --scl false \
    --arcface false
```

For convenience, the repository also ships bash helpers that submit the exact sweeps used in the paper:

```bash
bash scripts/train_cifar10.sh
bash scripts/train_cifar100.sh
bash scripts/train_imagenet.sh
```

Export environment variables such as `DATASET_DIR`, `RESULTS_DIR`, `HOURS`, or `GPUS` before running these helpers to override the defaults.

> **Note:** the generated SLURM scripts assume the software stack and scheduler configuration of the cluster used for the LPC paper (e.g., `module load python`, `conda activate lpc`, A100 GPUs, and NCCL settings). If your environment differs, adapt the templates in `scripts/generate_submit_slurm_jobs*.py` and `scripts/imagenet_generate_submit_slurm_jobs.py` accordingly before submitting jobs.

CLI arguments accepted by `scripts/main.py`:

| Flag | Purpose |
| --- | --- |
| `--config` | Path to the YAML configuration file. |
| `--dataset-dir` | Root directory where datasets are stored/downloaded. CIFAR datasets download automatically; ImageNet must already be present. |
| `--results-dir` | Output directory for checkpoints, result pickles, and flag files. Created if missing. |
| `--architecture-type` | One of `lin_pen`, `nonlin_pen`, or `no_pen`, matching the backbone. |
| `--l2-loss`, `--arcface`, `--scl` | Enable auxiliary loss terms. Pass the string `true`/`false` (case insensitive). |
| `--lr`, `--weight-decay` | Override the learning rate or weight decay defined in the YAML. |
| `--store-penultimate` | Store penultimate representations at the final epoch (`true`/`false`). |
| `--penultimate-nodes` | Select alternative penultimate width declared in the config (`wide` or `narrow`). |
| `--sample` | Optional string appended to the run name (useful for bookkeeping). |

Behind the scenes `scripts/main.py`:
1. Loads the dataset with the correct augmentations and normalisation (mean/std are computed automatically for non-ImageNet datasets).
2. Builds the requested backbone with the appropriate penultimate head and wraps it in `DistributedDataParallel`.
3. Instantiates `Trainer`, which runs the training loop, periodically evaluates on the train/test splits, and computes the collapse metrics.
4. Saves checkpoints and merges fresh metrics into a pickle file placed in `results_dir`.

## Monitoring and Outputs
`results_dir` will contain the following artefacts during training:

- `checkpoint_<run-name>.pth.tar` — Model weights, optimiser state, scheduler state, last epoch, and gamma value. The trainer automatically resumes from this checkpoint if it already exists.
- `<run-name>.pkl` — Pickle file containing all logged metrics. New epochs are merged on top of existing metrics thanks to `merge_with_existing_results`.
- `training_<run-name>.flag` — Job guard file created during execution and deleted when training finishes cleanly.
- `training_completed_<run-name>.flag` — Written when the run reaches `total_epochs`, making it easy to detect fully converged experiments.
- Optional penultimate representations (NumPy arrays) when `--store-penultimate true`.

All metrics reported to stdout are also available in the pickle file, including:

- Train/test accuracies and top-5 accuracy.
- Collapse diagnostics (within-class variation, equiangularity, equinorm, etc.).
- Coefficient of variation, distance margins, entropy estimates.
- DeepFool robustness scores and Lipschitz gradient norms.

## Resuming and Continuing Runs
Launching the same command again with an existing `results_dir` automatically resumes:

- Checkpoints are reloaded (`Trainer._load_checkpoint`) and the optimiser/scheduler states are restored.
- Metrics from newly completed epochs are merged into the existing pickle.
- If the previous run already reached `total_epochs`, training exits with a message indicating completion.


## Citation
If you use LPC in academic work, please cite the accompanying paper once it is publicly available.
