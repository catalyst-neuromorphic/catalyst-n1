import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from shd_loader import SHDDataset, collate_fn, N_CHANNELS, N_CLASSES

class SurrogateSpikeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        scale = 25.0
        grad = grad_output / (scale * torch.abs(x) + 1.0) ** 2
        return grad

surrogate_spike = SurrogateSpikeFunction.apply

class LIFNeuron(nn.Module):

    def __init__(self, size, beta_init=0.95, threshold=1.0, learn_beta=True):
        super().__init__()
        self.size = size
        self.threshold = threshold
        if learn_beta:
            init_val = np.log(beta_init / (1.0 - beta_init))
            self.beta_raw = nn.Parameter(torch.full((size,), init_val))
        else:
            self.register_buffer('beta_raw',
                                 torch.full((size,), np.log(beta_init / (1.0 - beta_init))))

    @property
    def beta(self):
        return torch.sigmoid(self.beta_raw)

    def forward(self, input_current, v_prev):
        beta = self.beta
        v = beta * v_prev + (1.0 - beta) * input_current
        spikes = surrogate_spike(v - self.threshold)
        v = v * (1.0 - spikes)
        return v, spikes

class AdaptiveLIFNeuron(nn.Module):

    def __init__(self, size, alpha_init=0.90, rho_init=0.85, beta_a_init=1.8,
                 threshold=1.0):
        super().__init__()
        self.size = size
        self.threshold_base = nn.Parameter(torch.full((size,), threshold))

        init_alpha = np.log(alpha_init / (1.0 - alpha_init))
        self.alpha_raw = nn.Parameter(torch.full((size,), init_alpha))

        init_rho = np.log(rho_init / (1.0 - rho_init))
        self.rho_raw = nn.Parameter(torch.full((size,), init_rho))

        init_beta_a = np.log(np.exp(beta_a_init) - 1.0)
        self.beta_a_raw = nn.Parameter(torch.full((size,), init_beta_a))

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_raw)

    def forward(self, input_current, v_prev, a_prev, spike_prev):
        alpha = torch.sigmoid(self.alpha_raw)
        rho = torch.sigmoid(self.rho_raw)
        beta_a = F.softplus(self.beta_a_raw)

        a_new = rho * a_prev + spike_prev
        theta = self.threshold_base + beta_a * a_new

        v = alpha * v_prev + (1.0 - alpha) * input_current
        spikes = surrogate_spike(v - theta)
        v = v * (1.0 - spikes)

        return v, spikes, a_new

def event_drop_augment(spikes_batch, drop_time_prob=0.1, drop_neuron_prob=0.05):
    if random.random() < 0.5:
        B, T, C = spikes_batch.shape
        mask = (torch.rand(1, T, 1, device=spikes_batch.device)
                > drop_time_prob).float()
        return spikes_batch * mask
    else:
        B, T, C = spikes_batch.shape
        mask = (torch.rand(1, 1, C, device=spikes_batch.device)
                > drop_neuron_prob).float()
        return spikes_batch * mask

class SHDSNN(nn.Module):

    def __init__(self, n_input=N_CHANNELS, n_hidden=256, n_output=N_CLASSES,
                 beta_hidden=0.95, beta_out=0.9, threshold=1.0, dropout=0.3,
                 neuron_type='lif', alpha_init=0.90, rho_init=0.85,
                 beta_a_init=1.8):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.dropout_p = dropout
        self.neuron_type = neuron_type

        self.fc1 = nn.Linear(n_input, n_hidden, bias=False)
        self.fc2 = nn.Linear(n_hidden, n_output, bias=False)

        self.fc_rec = nn.Linear(n_hidden, n_hidden, bias=False)

        if neuron_type == 'adlif':
            self.lif1 = AdaptiveLIFNeuron(
                n_hidden, alpha_init=alpha_init, rho_init=rho_init,
                beta_a_init=beta_a_init, threshold=threshold)
        else:
            self.lif1 = LIFNeuron(n_hidden, beta_init=beta_hidden,
                                   threshold=threshold, learn_beta=True)

        self.lif2 = LIFNeuron(n_output, beta_init=beta_out,
                               threshold=threshold, learn_beta=True)

        self.dropout = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.orthogonal_(self.fc_rec.weight, gain=0.2)

    def forward(self, x):
        batch, T, _ = x.shape
        device = x.device

        v1 = torch.zeros(batch, self.n_hidden, device=device)
        v2 = torch.zeros(batch, self.n_output, device=device)
        spk1 = torch.zeros(batch, self.n_hidden, device=device)

        out_sum = torch.zeros(batch, self.n_output, device=device)

        if self.neuron_type == 'adlif':
            a1 = torch.zeros(batch, self.n_hidden, device=device)

        for t in range(T):
            I1 = self.fc1(x[:, t]) + self.fc_rec(spk1)

            if self.neuron_type == 'adlif':
                v1, spk1, a1 = self.lif1(I1, v1, a1, spk1)
            else:
                v1, spk1 = self.lif1(I1, v1)

            spk1_drop = self.dropout(spk1) if self.training else spk1

            I2 = self.fc2(spk1_drop)
            beta_out = self.lif2.beta
            v2 = beta_out * v2 + (1.0 - beta_out) * I2

            out_sum = out_sum + v2

        return out_sum / T

def train_epoch(model, loader, optimizer, device, use_event_drop=False,
                label_smoothing=0.0):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        if use_event_drop:
            inputs = event_drop_augment(inputs)

        optimizer.zero_grad()
        output = model(inputs)
        loss = F.cross_entropy(output, labels, label_smoothing=label_smoothing)
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
    parser = argparse.ArgumentParser(description="Train SNN on SHD benchmark")
    parser.add_argument("--data-dir", default="data/shd")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--beta-hidden", type=float, default=0.95,
                        help="Initial membrane decay factor for hidden layer")
    parser.add_argument("--beta-out", type=float, default=0.9,
                        help="Initial membrane decay factor for output layer")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--dt", type=float, default=4e-3,
                        help="Time bin width in seconds (4ms -> 250 bins)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default="shd_model.pt")
    parser.add_argument("--no-recurrent", action="store_true",
                        help="Disable recurrent hidden connection")
    parser.add_argument("--neuron-type", choices=["lif", "adlif"], default="lif",
                        help="Neuron model: lif (standard) or adlif (adaptive, SE)")
    parser.add_argument("--alpha-init", type=float, default=0.90,
                        help="Initial membrane decay for adLIF (default: 0.90)")
    parser.add_argument("--rho-init", type=float, default=0.85,
                        help="Initial adaptation decay for adLIF (default: 0.85)")
    parser.add_argument("--beta-a-init", type=float, default=1.8,
                        help="Initial adaptation strength for adLIF (default: 1.8)")
    parser.add_argument("--event-drop", action="store_true", default=None,
                        help="Enable event-drop augmentation (auto-enabled for adlif)")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing factor (0.0=off, 0.1=recommended)")
    args = parser.parse_args()

    if args.event_drop is None:
        args.event_drop = (args.neuron_type == 'adlif')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading SHD dataset...")
    train_ds = SHDDataset(args.data_dir, "train", dt=args.dt)
    test_ds = SHDDataset(args.data_dir, "test", dt=args.dt)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)

    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}, "
          f"Time bins: {train_ds.n_bins} (dt={args.dt*1000:.1f}ms)")

    model = SHDSNN(
        n_hidden=args.hidden,
        threshold=args.threshold,
        beta_hidden=args.beta_hidden,
        beta_out=args.beta_out,
        dropout=args.dropout,
        neuron_type=args.neuron_type,
        alpha_init=args.alpha_init,
        rho_init=args.rho_init,
        beta_a_init=args.beta_a_init,
    ).to(device)

    if args.no_recurrent:
        model.fc_rec.weight.data.zero_()
        model.fc_rec.weight.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    neuron_info = args.neuron_type.upper()
    if args.neuron_type == 'adlif':
        neuron_info += f" (alpha={args.alpha_init}, rho={args.rho_init}, beta_a={args.beta_a_init})"
    print(f"Model: {N_CHANNELS}->{args.hidden}->{N_CLASSES}, "
          f"{n_params:,} params ({neuron_info}, "
          f"recurrent={'off' if args.no_recurrent else 'on'}, "
          f"dropout={args.dropout}, event_drop={args.event_drop})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs,
                                                            eta_min=1e-5)

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device,
                                               use_event_drop=args.event_drop,
                                               label_smoothing=args.label_smoothing)
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
