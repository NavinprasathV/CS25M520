# VGG6 on CIFAR-10: effect of activation, optimizer, batch size, epochs and learning rate

CS6886W Assignment 1. Code and trained model for the experiments in the report PDF.

## Environment

- Windows laptop, NVIDIA GeForce RTX 5060 Laptop GPU (8 GB), Intel i7-14650HX
- Python 3.12.10
- torch 2.11.0+cu128, torchvision 0.26.0+cu128, wandb 0.31.0, numpy 2.5.4, PyYAML 6.0.3

## Setup

```
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install wandb==0.31.0 numpy==2.5.4 PyYAML==6.0.3
wandb login
```

CIFAR-10 is downloaded automatically into `./data` on the first run. Use `--no_wandb` to run without a W&B account.

## Files

| File | Purpose |
|---|---|
| `model.py` | VGG6 (conv 64-64, pool, conv 128-128, pool, global average pool, linear) with a selectable activation |
| `data.py` | CIFAR-10 loading, normalization, augmentation, train/validation/test split |
| `evaluate.py` | Evaluation loop (loss and accuracy) |
| `train.py` | Training loop, optimizer selection, W&B logging, test accuracy of the best-validation epoch |
| `sweep_*.yaml` | W&B sweep definitions (activation, optimizer x learning rate, batch size x epochs) |
| `checkpoints/best.pt` | Trained model of the best configuration (state_dict) |

## Data and transforms

- Training images: RandomCrop(32, padding=4), RandomHorizontalFlip, ToTensor, Normalize(mean 0.4914, 0.4822, 0.4465; std 0.2470, 0.2435, 0.2616), Cutout (one 16x16 patch).
- Validation and test images: ToTensor and Normalize only.
- Split: 45,000 train / 5,000 validation, taken from the 50,000 training images with a fixed permutation (seed 0, identical for every run). The 10,000-image official test set is used only once per run, with the weights of the best-validation epoch.

## Seeds

`--seed 42` seeds Python, NumPy and PyTorch (CPU and CUDA), with cuDNN deterministic mode on and benchmark mode off. The train/validation split uses its own fixed seed (0). The default `--num_workers 2` is part of the configuration: changing it changes the random augmentation stream and therefore the results. On the same machine, repeated runs gave identical accuracies. Other GPUs or library versions may differ slightly.

## Reproduce the best configuration

ReLU, Nadam, learning rate 0.01, batch size 128, 100 epochs, seed 42:

```
python train.py --activation relu --optimizer nadam --lr 0.01 --batch_size 128 --epochs 100 --seed 42 --num_workers 2 --save_path checkpoints/best.pt
```

Result obtained: **best validation accuracy 88.04%, test accuracy 86.79%**. This takes about 12 minutes on the machine above.

## Reproduce the sweeps

The `.yaml` files contain the absolute path of my virtual-environment Python (`C:\Projects\vgg6-cifar10\.venv\Scripts\python.exe`). Edit that line to your own path first.

```
wandb sweep --project vgg6-cifar10 sweep_activation.yaml
wandb sweep --project vgg6-cifar10 sweep_optimizer.yaml
wandb sweep --project vgg6-cifar10 sweep_batch_epochs.yaml
wandb agent <entity>/vgg6-cifar10/<sweep_id>
```

## Results and discussion

All sweep runs use seed 42 and report the best validation accuracy over the epochs.

### Activation (Adam, lr 0.001, batch size 128, 60 epochs)

| Activation | Best val acc | Test acc |
|---|---|---|
| ReLU | 85.34 | 84.76 |
| Sigmoid | 63.06 | 62.49 |
| Tanh | 77.00 | 76.87 |
| SiLU | 85.34 | 84.22 |
| GELU | 85.46 | 84.21 |

ReLU, SiLU and GELU are within about 0.1 points of each other, which is within run-to-run noise (one seed, 5,000 validation images). Tanh and sigmoid are clearly worse. Sigmoid and tanh saturate, so gradients shrink through the layers, sigmoid outputs are not zero-centred, and the weight initialisation used (He-style) is designed for ReLU-type activations.

### Optimizer x learning rate (ReLU, batch size 128, 60 epochs): best validation accuracy

| Optimizer | 0.1 | 0.01 | 0.001 | 0.0001 |
|---|---|---|---|---|
| SGD (momentum 0.9) | 86.38 | 82.86 | 72.52 | 53.82 |
| Nesterov SGD (momentum 0.9) | 86.70 | 83.06 | 73.20 | 54.10 |
| Adam | 73.44 | 86.98 | 85.34 | 75.98 |
| Adagrad | 83.60 | 79.28 | 62.98 | 43.40 |
| RMSprop | 56.74 | 85.56 | 83.70 | 73.70 |
| Nadam | 83.40 | 87.20 | 84.82 | 76.30 |

Each optimizer has a different best learning rate: the SGD family and Adagrad peak at the largest rate tried (0.1), while Adam, RMSprop and Nadam peak at 0.01. Adaptive methods tolerate small learning rates much better (about 75% at 0.0001) than plain SGD (about 54%), which converges slowly there. Nesterov is slightly ahead of plain momentum SGD at every learning rate, by a small margin. Adagrad's accumulated squared gradients keep shrinking its step size, so it needs a large initial rate. RMSprop is the most sensitive to a too-large rate. SGD, Nesterov and Adagrad peaked at the largest learning rate tried, so higher rates were not explored.

### Batch size x epochs (ReLU, Nadam, lr 0.01): best validation accuracy

| Batch size | 20 epochs | 40 epochs | 60 epochs |
|---|---|---|---|
| 64 | 82.54 | 86.30 | 87.22 |
| 128 | 81.54 | 85.58 | 87.20 |
| 256 | 80.94 | 85.86 | 87.10 |

More epochs always help, with diminishing returns. Smaller batches help when training is short because they make more weight updates per epoch; at 60 epochs the three batch sizes are within noise of each other. The learning-rate effect for the same Nadam/batch-128 setting is in the optimizer table: 76.30 (0.0001), 84.82 (0.001), 87.20 (0.01), 83.40 (0.1), so too small converges slowly and too large is unstable.

### Final model (100 epochs)

ReLU, Nadam, lr 0.01, batch size 128: best validation accuracy 88.04%, test accuracy 86.79%.
