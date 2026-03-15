import os
import sys
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

_SDK_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _SDK_DIR not in sys.path:
    sys.path.insert(0, _SDK_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from shd_loader import SHDDataset, collate_fn, N_CHANNELS, N_CLASSES
from shd_train import SHDSNN

from neurocore import Network
from neurocore.constants import WEIGHT_MIN, WEIGHT_MAX

def quantize_weights(w_float, threshold_float, threshold_hw=1000):
    scale = threshold_hw / threshold_float
    w_scaled = w_float * scale
    w_int = np.clip(np.round(w_scaled), WEIGHT_MIN, WEIGHT_MAX).astype(np.int32)
    return w_int.T

def detect_neuron_type(checkpoint):
    state = checkpoint['model_state_dict']
    if 'lif1.alpha_raw' in state:
        return 'adlif'
    return 'lif'

def compute_hardware_params(checkpoint, threshold_hw=1000, neuron_type=None):
    state = checkpoint['model_state_dict']
    if neuron_type is None:
        neuron_type = detect_neuron_type(checkpoint)

    params = {'neuron_type': neuron_type}

    if neuron_type == 'adlif':
        alpha_raw = state.get('lif1.alpha_raw', None)
        if alpha_raw is not None:
            alpha = torch.sigmoid(alpha_raw).cpu().numpy()
            params['hidden_alpha_mean'] = float(alpha.mean())
            params['hidden_alpha_std'] = float(alpha.std())
            params['hidden_decay_v'] = int(round(alpha.mean() * 4096))
            params['hidden_beta_mean'] = float(alpha.mean())

        rho_raw = state.get('lif1.rho_raw', None)
        if rho_raw is not None:
            rho = torch.sigmoid(rho_raw).cpu().numpy()
            params['hidden_rho_mean'] = float(rho.mean())
            params['hidden_rho_note'] = 'training-only (not deployed)'

        beta_a_raw = state.get('lif1.beta_a_raw', None)
        if beta_a_raw is not None:
            import torch.nn.functional as F_
            beta_a = F_.softplus(beta_a_raw).cpu().numpy()
            params['hidden_beta_a_mean'] = float(beta_a.mean())
            params['hidden_beta_a_note'] = 'training-only (not deployed)'
    else:
        beta_hid_raw = state.get('lif1.beta_raw', None)
        if beta_hid_raw is not None:
            beta_hid = torch.sigmoid(beta_hid_raw).cpu().numpy()
            params['hidden_beta_mean'] = float(beta_hid.mean())
            params['hidden_beta_std'] = float(beta_hid.std())
            params['hidden_decay_v'] = int(round(beta_hid.mean() * 4096))

    beta_out_raw = state.get('lif2.beta_raw', None)
    if beta_out_raw is not None:
        beta_out = torch.sigmoid(beta_out_raw).cpu().numpy()
        params['output_beta_mean'] = float(beta_out.mean())
        params['output_beta_std'] = float(beta_out.std())
        params['output_decay_v'] = int(round(beta_out.mean() * 4096))

    params['threshold_hw'] = threshold_hw
    return params

def build_sdk_network(checkpoint, threshold_hw=1000):
    args = checkpoint['args']
    threshold_float = args['threshold']
    n_hidden = args['hidden']

    state = checkpoint['model_state_dict']
    w_fc1 = state['fc1.weight'].cpu().numpy()
    w_fc2 = state['fc2.weight'].cpu().numpy()
    w_rec = state['fc_rec.weight'].cpu().numpy()

    wm_fc1 = quantize_weights(w_fc1, threshold_float, threshold_hw)
    wm_fc2 = quantize_weights(w_fc2, threshold_float, threshold_hw)
    wm_rec = quantize_weights(w_rec, threshold_float, threshold_hw)

    hw = compute_hardware_params(checkpoint, threshold_hw)
    leak_hid = max(1, int(round((1 - hw.get('hidden_beta_mean', 0.95)) * threshold_hw)))
    leak_out = max(1, int(round((1 - hw.get('output_beta_mean', 0.9)) * threshold_hw)))

    net = Network()
    inp = net.population(N_CHANNELS,
                         params={'threshold': 65535, 'leak': 0, 'refrac': 0},
                         label="input")
    hid = net.population(n_hidden,
                         params={'threshold': threshold_hw, 'leak': leak_hid, 'refrac': 0},
                         label="hidden")
    out = net.population(N_CLASSES,
                         params={'threshold': threshold_hw, 'leak': leak_out, 'refrac': 0},
                         label="output")

    net.connect(inp, hid, weight_matrix=wm_fc1)
    net.connect(hid, out, weight_matrix=wm_fc2)
    net.connect(hid, hid, weight_matrix=wm_rec)

    nonzero_fc1 = np.count_nonzero(wm_fc1)
    nonzero_fc2 = np.count_nonzero(wm_fc2)
    nonzero_rec = np.count_nonzero(wm_rec)
    total_conn = nonzero_fc1 + nonzero_fc2 + nonzero_rec
    print(f"Quantized weights (threshold_hw={threshold_hw}):")
    print(f"  fc1: {wm_fc1.shape}, {nonzero_fc1:,} nonzero, "
          f"range [{wm_fc1.min()}, {wm_fc1.max()}]")
    print(f"  fc2: {wm_fc2.shape}, {nonzero_fc2:,} nonzero, "
          f"range [{wm_fc2.min()}, {wm_fc2.max()}]")
    print(f"  rec: {wm_rec.shape}, {nonzero_rec:,} nonzero, "
          f"range [{wm_rec.min()}, {wm_rec.max()}]")
    print(f"  Total connections: {total_conn:,}")
    if 'hidden_decay_v' in hw:
        print(f"  Hardware decay_v (hidden): {hw['hidden_decay_v']} "
              f"(beta={hw['hidden_beta_mean']:.4f})")
    if 'output_decay_v' in hw:
        print(f"  Hardware decay_v (output): {hw['output_decay_v']} "
              f"(beta={hw['output_beta_mean']:.4f})")

    return net, n_hidden

def run_pytorch_quantized_inference(checkpoint, test_ds, device='cpu',
                                     neuron_type=None):
    args = checkpoint['args']
    threshold_float = args['threshold']
    threshold_hw = 1000
    if neuron_type is None:
        neuron_type = args.get('neuron_type', detect_neuron_type(checkpoint))

    model = SHDSNN(
        n_hidden=args['hidden'],
        threshold=args['threshold'],
        beta_hidden=args.get('beta_hidden', 0.95),
        beta_out=args.get('beta_out', 0.9),
        dropout=0.0,
        neuron_type=neuron_type,
        alpha_init=args.get('alpha_init', 0.90),
        rho_init=args.get('rho_init', 0.85),
        beta_a_init=args.get('beta_a_init', 1.8),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    scale = threshold_hw / threshold_float
    skip_keys = ('beta', 'alpha', 'rho', 'threshold_base')
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and not any(k in name for k in skip_keys):
                q = torch.round(param * scale).clamp(WEIGHT_MIN, WEIGHT_MAX) / scale
                param.copy_(q)

    model.eval()
    loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            correct += (output.argmax(1) == labels).sum().item()
            total += inputs.size(0)

    acc = correct / total
    print(f"  PyTorch quantized accuracy: {correct}/{total} = {acc*100:.1f}%")
    return acc

def main():
    parser = argparse.ArgumentParser(description="Deploy trained SHD model")
    parser.add_argument("--checkpoint", default="shd_model.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--data-dir", default="data/shd")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Limit test samples (default: all)")
    parser.add_argument("--threshold-hw", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=4e-3)
    parser.add_argument("--neuron-type", choices=["lif", "adlif"], default=None,
                        help="Neuron model (auto-detected from checkpoint if omitted)")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    train_args = ckpt['args']

    neuron_type = args.neuron_type or train_args.get('neuron_type', detect_neuron_type(ckpt))
    print(f"  Training accuracy: {ckpt['test_acc']*100:.1f}%")
    print(f"  Architecture: {N_CHANNELS}->{train_args['hidden']}->{N_CLASSES} ({neuron_type.upper()})")

    print("\nLoading test dataset...")
    test_ds = SHDDataset(args.data_dir, "test", dt=args.dt)
    print(f"  {len(test_ds)} samples, {test_ds.n_bins} time bins")

    print("\nHardware parameter mapping:")
    hw_params = compute_hardware_params(ckpt, args.threshold_hw, neuron_type)
    for k, v in sorted(hw_params.items()):
        print(f"  {k}: {v}")

    print("\nPyTorch quantized inference:")
    pytorch_acc = run_pytorch_quantized_inference(ckpt, test_ds,
                                                   neuron_type=neuron_type)

    print("\nSDK network summary:")
    net, n_hidden = build_sdk_network(ckpt, threshold_hw=args.threshold_hw)

    print("\nResults:")
    print(f"  PyTorch float accuracy:     {ckpt['test_acc']*100:.1f}%")
    print(f"  PyTorch quantized accuracy: {pytorch_acc*100:.1f}%")
    gap = abs(ckpt['test_acc'] - pytorch_acc) * 100
    print(f"  Quantization loss:          {gap:.1f}%")
    print(f"\n  Hardware deployment: CUBA mode (decay_v={hw_params.get('hidden_decay_v', 'N/A')})")
    print(f"  Total synapses: {sum(1 for c in net.connections for _ in range(1)):,}")

if __name__ == "__main__":
    main()
