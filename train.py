import argparse
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from data import get_data
from evaluate import evaluate
from model import VGG6


def get_optimizer(name, params, lr, momentum, weight_decay):
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "nesterov":
        return optim.SGD(params, lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    if name == "nadam":
        return optim.NAdam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError("unknown optimizer: " + name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--activation", default="relu")
    p.add_argument("--optimizer", default="adam")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_path", default="")
    p.add_argument("--no_wandb", action="store_true")
    args = p.parse_args()

    # Fix all random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = f"act_{args.activation}_bs_{args.batch_size}_opt_{args.optimizer}_lr_{args.lr}_ep_{args.epochs}"
    wandb.init(project="vgg6-cifar10", name=name, config=vars(args),
               mode="disabled" if args.no_wandb else "online")

    train_loader, val_loader, test_loader = get_data(args.batch_size, args.num_workers)
    model = VGG6(args.activation).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr,
                              args.momentum, args.weight_decay)

    best_val, best_state = -1.0, None
    for epoch in range(args.epochs):
        # ---- training ----
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        train_loss, train_acc = total_loss / total, 100.0 * correct / total

        # ---- validation ----
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        wandb.log({"train_loss": train_loss, "train_acc": train_acc,
                   "val_loss": val_loss, "val_acc": val_acc}, step=epoch)
        print(f"Epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.2f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.2f}")

        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())

    # ---- final test accuracy using the best-validation weights ----
    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Best val acc: {best_val:.2f} | Test acc: {test_acc:.2f}")
    wandb.summary["test_acc"] = test_acc
    wandb.summary["val_acc"] = best_val      # best validation accuracy of this run
    if args.save_path:
        torch.save(best_state, args.save_path)
    wandb.finish()


if __name__ == "__main__":   # required on Windows because of DataLoader workers
    main()