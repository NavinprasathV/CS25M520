import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class Cutout:
    """Zero out one random square patch of the image."""
    def __init__(self, length=16):
        self.length = length

    def __call__(self, img):
        _, h, w = img.shape
        y = torch.randint(h, (1,)).item()
        x = torch.randint(w, (1,)).item()
        y1, y2 = max(y - self.length // 2, 0), min(y + self.length // 2, h)
        x1, x2 = max(x - self.length // 2, 0), min(x + self.length // 2, w)
        img = img.clone()
        img[:, y1:y2, x1:x2] = 0
        return img


def get_data(batch_size, num_workers=4):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        Cutout(16),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # Same 50k training images loaded twice: augmented (train) and clean (validation)
    train_full = datasets.CIFAR10("./data", train=True, transform=train_tf, download=True)
    val_full = datasets.CIFAR10("./data", train=True, transform=test_tf, download=True)
    test_set = datasets.CIFAR10("./data", train=False, transform=test_tf, download=True)

    # Fixed split: 45,000 train / 5,000 validation (identical for every run)
    idx = torch.randperm(50000, generator=torch.Generator().manual_seed(0)).tolist()
    train_set = Subset(train_full, idx[5000:])
    val_set = Subset(val_full, idx[:5000])

    kw = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **kw)
    val_loader = DataLoader(val_set, batch_size=500, shuffle=False, **kw)
    test_loader = DataLoader(test_set, batch_size=500, shuffle=False, **kw)
    return train_loader, val_loader, test_loader