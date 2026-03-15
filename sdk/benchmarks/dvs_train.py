import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from dvs_loader import DVSGestureDataset, collate_fn, N_CHANNELS, N_CLASSES
from shd_train import SubtractiveLIF, surrogate_spike

class DVSSNN(nn.Module):

    def __init__(self, n_input=N_CHANNELS, n_hidden=512, n_output=N_CLASSES,
                 threshold=1.0, leak=0.003):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.fc1 = nn.Linear(n_input, n_hidden, bias=False)
        self.fc2 = nn.Linear(n_hidden, n_output, bias=False)
        self.fc_rec = nn.Linear(n_hidden, n_hidden, bias=False)

        self.lif1 = SubtractiveLIF(n_hidden, threshold=threshold, leak=leak)
        self.output_leak = leak * 0.5

        nn.init.xavier_uniform_(self.fc1.weight, gain=0.1)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.3)
        nn.init.orthogonal_(self.fc_rec.weight, gain=0.1)

    def forward(self, x):
        batch, T, _ = x.shape
        device = x.device

        v1 = torch.zeros(batch, self.n_hidden, device=device)
        v2 = torch.zeros(batch, self.n_output, device=device)
        spk1 = torch.zeros(batch, self.n_hidden, device=device)
        out_sum = torch.zeros(batch, self.n_output, device=device)

        for t in range(T):
            I1 = self.fc1(x[:, t]) + self.fc_rec(spk1)
            v1, spk1 = self.lif1(I1, v1)

            I2 = self.fc2(spk1)
            v2 = v2 + I2 - self.output_leak
            v2 = torch.clamp(v2, min=0.0)
            out_sum = out_sum + v2

        return out_sum / T

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += (output.argmax(1) == labels).sum().item()
        total += inputs.size(0)

    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        loss = F.cross_entropy(output, labels)
        total_loss += loss.item() * inputs.size(0)
        correct += (output.argmax(1) == labels).sum().item()
        total += inputs.size(0)

    return total_loss / total, correct / total

def main():
    parser = argparse.ArgumentParser(description="Train SNN on DVS Gesture")
    parser.add_argument("--data-dir", default="data/dvs_gesture")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--leak", type=float, default=0.003)
    parser.add_argument("--dt", type=float, default=10e-3,
                        help="Time bin width (10ms -> 150 bins for 1.5s)")
    parser.add_argument("--duration", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default="dvs_model.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading DVS Gesture dataset (first load downloads ~1.5GB)...")
    train_ds = DVSGestureDataset(args.data_dir, train=True,
                                  dt=args.dt, duration=args.duration)
    test_ds = DVSGestureDataset(args.data_dir, train=False,
                                 dt=args.dt, duration=args.duration)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)

    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}, "
          f"Time bins: {train_ds.n_bins} (dt={args.dt*1000:.1f}ms)")

    model = DVSSNN(
        n_hidden=args.hidden,
        threshold=args.threshold,
        leak=args.leak,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {N_CHANNELS}->{args.hidden}->{N_CLASSES}, {n_params:,} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
                'args': vars(args),
            }, args.save)

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train: {train_loss:.4f} / {train_acc*100:.1f}% | "
              f"Test: {test_loss:.4f} / {test_acc*100:.1f}% | "
              f"LR={lr:.2e} | Best={best_acc*100:.1f}%")

    print(f"\nDone. Best test accuracy: {best_acc*100:.1f}%")
    print(f"Model saved to {args.save}")

if __name__ == "__main__":
    main()
