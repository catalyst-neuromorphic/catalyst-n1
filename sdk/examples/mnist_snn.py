import sys
import os
import time
import argparse
import functools
import builtins
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import neurocore as nc
from neurocore.constants import NEURONS_PER_CORE, POOL_DEPTH

try:
    import torch
    import torchvision
    import torchvision.transforms as transforms
except ImportError:
    print("Requires: pip install torch torchvision")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def load_mnist(data_dir="data"):
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform)
    return train_set, test_set

def rate_encode(image_tensor, timesteps, rng):
    flat = image_tensor.view(-1).numpy()
    rand = rng.random((timesteps, 784)).astype(np.float32)
    return rand < flat[np.newaxis, :]

def build_mnist_network(n_exc=39, n_input=784, exc_threshold=5000,
                        inh_threshold=3000, inh_weight=-300,
                        exc_inh_weight=5000):
    max_exc = (NEURONS_PER_CORE - n_input) // 2
    if n_exc > max_exc:
        n_exc = max_exc

    net = nc.Network()
    input_pop = net.population(n_input, params={
        "threshold": 100, "leak": 0, "refrac": 0}, label="input")
    exc_pop = net.population(n_exc, params={
        "threshold": exc_threshold, "leak": 1, "refrac": 5}, label="excitatory")
    inh_pop = net.population(n_exc, params={
        "threshold": inh_threshold, "leak": 1, "refrac": 2}, label="inhibitory")

    pool_for_others = n_exc + n_exc * n_exc
    max_fan_out = (POOL_DEPTH - pool_for_others) // n_input

    if n_exc <= max_fan_out:
        net.connect(input_pop, exc_pop, topology="all_to_all", weight=80)
        fan_out_used = n_exc
        print(f"  Input->Exc: all_to_all ({n_input * n_exc} synapses)")
    else:
        fan_out_used = max_fan_out
        net.connect(input_pop, exc_pop, topology="fixed_fan_out",
                    fan_out=fan_out_used, weight=80, seed=42)
        print(f"  Input->Exc: fixed_fan_out={fan_out_used}")

    net.connect(exc_pop, inh_pop, topology="one_to_one", weight=exc_inh_weight)
    net.connect(inh_pop, exc_pop, topology="all_to_all", weight=inh_weight)

    total_pool = n_input * fan_out_used + n_exc + n_exc * n_exc
    print(f"  Pool: {total_pool}/{POOL_DEPTH} ({100 * total_pool / POOL_DEPTH:.0f}%)")
    return net, input_pop, exc_pop, inh_pop

def compute_gid_arrays(sim, input_pop, exc_pop, n_input=784):
    placement = sim._compiled.placement
    dev = sim.device
    n_exc = exc_pop.size

    exc_gids = [placement.neuron_map[(exc_pop.id, i)] for i in range(n_exc)]
    exc_gid_np = np.array([c * NEURONS_PER_CORE + n for c, n in exc_gids], dtype=np.int64)
    exc_gid_t = torch.from_numpy(exc_gid_np).to(dev)

    pixel_gids = [placement.neuron_map[(input_pop.id, px)] for px in range(n_input)]
    pixel_gid_np = np.array([c * NEURONS_PER_CORE + n for c, n in pixel_gids], dtype=np.int64)
    pixel_gid_t = torch.from_numpy(pixel_gid_np).to(dev)

    return exc_gid_np, exc_gid_t, pixel_gid_np, pixel_gid_t

def prototype_initialize(sim, train_set, n_exc, exc_gid_t, pixel_gid_t,
                         weight_norm_target):
    dev = sim.device
    stride = max(1, len(train_set) // n_exc)
    labels_used = []

    for i in range(n_exc):
        proto_idx = i * stride
        img, label = train_set[proto_idx]
        labels_used.append(label)
        pixel_intensity = img.view(-1).to(dev)

        winner_gid_t = exc_gid_t[i:i + 1]
        sim.competitive_update(
            winner_gid_t, pixel_intensity, pixel_gid_t,
            eta_ltp=1.0, eta_ltd=0.0)
        sim.normalize_learnable_weights(weight_norm_target,
                                        target_gids=winner_gid_t)

    from collections import Counter
    dist = Counter(labels_used)
    dist_str = " ".join(f"{d}:{c}" for d, c in sorted(dist.items()))
    print(f"  Prototype class distribution: {dist_str}")

def dot_product_batch(sim, images_flat, pixel_gid_t, exc_gid_t):
    dev = sim.device
    input_vec = torch.zeros(sim._n, dtype=torch.float32, device=dev)
    input_vec[pixel_gid_t] = images_flat
    acc = torch.sparse.mm(sim._W_soma, input_vec.unsqueeze(1)).squeeze(1)
    return acc[exc_gid_t].cpu().numpy()

def train_epoch(sim, train_set, n_exc,
                exc_gid_t, pixel_gid_t,
                max_images=None, epoch=0,
                weight_norm_target=10000,
                eta_ltp=0.05, eta_ltd=0.01, k_winners=3,
                ior=None, ior_frac=0.3, ior_decay=0.95):
    n_images = len(train_set) if max_images is None else min(max_images, len(train_set))
    dev = sim.device

    if ior is None:
        ior = np.zeros(n_exc)

    winner_class_counts = np.zeros((n_exc, 10))
    winner_tracker = []

    t_start = time.perf_counter()

    for img_idx in range(n_images):
        image, label = train_set[img_idx]
        pixel_intensity = image.view(-1).to(dev)

        exc_input = dot_product_batch(sim, pixel_intensity, pixel_gid_t, exc_gid_t)

        ior *= ior_decay

        adjusted = exc_input - ior
        sorted_idx = np.argsort(adjusted)[::-1]
        winners = sorted_idx[:k_winners]
        winners = winners[adjusted[winners] > 0]

        if winners:
            for w in winners:
                winner_class_counts[w, label] += 1
            winner_idx_t = torch.from_numpy(winners.astype(np.int64)).to(dev)
            winner_gids_t = exc_gid_t[winner_idx_t]

            sim.competitive_update(
                winner_gids_t, pixel_intensity, pixel_gid_t,
                eta_ltp=eta_ltp, eta_ltd=eta_ltd)

            mean_input = max(1.0, np.mean(exc_input))
            for idx in winners:
                ior[idx] += mean_input * ior_frac

            winner_tracker.append(int(winners[0]))

        sim.normalize_learnable_weights(weight_norm_target, target_gids=exc_gid_t)

        if (img_idx + 1) % 1000 == 0:
            elapsed = time.perf_counter() - t_start
            rate = (img_idx + 1) / elapsed
            recent = winner_tracker[-1000:]
            n_unique = len(set(recent))
            print(f"  [{img_idx + 1}/{n_images}] {rate:.0f} img/s, "
                  f"unique winners: {n_unique}/{n_exc}")

    elapsed = time.perf_counter() - t_start
    print(f"  Epoch: {n_images} images in {elapsed:.1f}s ({n_images / elapsed:.0f} img/s)")

    sim._sync_weights_to_adjacency()
    return winner_class_counts, ior

def assign_neurons(winner_class_counts, n_exc, n_classes=10):
    assignments = np.argmax(winner_class_counts, axis=1)
    never_won = winner_class_counts.sum(axis=1) == 0
    n_active = n_exc - np.sum(never_won)
    for c in range(n_classes):
        count = np.sum((assignments == c) & ~never_won)
        if count > 0:
            print(f"    Digit {c}: {count} neurons")
    if np.sum(never_won) > 0:
        print(f"    Unassigned (never won): {np.sum(never_won)} neurons")
    print(f"    Active neurons: {n_active}/{n_exc}")
    return assignments

def assign_neurons_dot(sim, train_set, n_exc, exc_gid_t, pixel_gid_t,
                       n_images=5000):
    dev = sim.device
    class_responses = np.zeros((n_exc, 10))
    class_counts = np.zeros(10)

    for img_idx in range(min(n_images, len(train_set))):
        image, label = train_set[img_idx]
        exc_input = dot_product_batch(sim, image.view(-1).to(dev),
                                       pixel_gid_t, exc_gid_t)
        class_responses[:, label] += exc_input
        class_counts[label] += 1

    avg = class_responses / np.maximum(class_counts[np.newaxis, :], 1)
    assignments = np.argmax(avg, axis=1)

    for c in range(10):
        count = np.sum(assignments == c)
        if count > 0:
            print(f"    Digit {c}: {count} neurons")

    sorted_avg = np.sort(avg, axis=1)[:, ::-1]
    selectivity = sorted_avg[:, 0] / np.maximum(sorted_avg[:, 1], 1)
    print(f"    Selectivity (best/2nd): min={selectivity.min():.2f}, "
          f"median={np.median(selectivity):.2f}, max={selectivity.max():.2f}")

    return assignments

def classify_snn(sim, test_set, n_exc, assignments,
                 exc_gid_np, pixel_gid_np,
                 presentation_time=50, max_images=None, rng=None,
                 stim_current=200):
    if rng is None:
        rng = np.random.RandomState(999)
    n_images = len(test_set) if max_images is None else min(max_images, len(test_set))
    n_total = sim._n
    dev = sim.device
    sim.set_learning(learn=False)

    predictions, labels = [], []
    t_start = time.perf_counter()

    for img_idx in range(n_images):
        image, label = test_set[img_idx]
        spikes_pattern = rate_encode(image, presentation_time, rng)
        schedule_np = np.zeros((presentation_time, n_total), dtype=np.int32)
        for t in range(presentation_time):
            sp = np.where(spikes_pattern[t])[0]
            if sp:
                schedule_np[t, pixel_gid_np[sp]] = stim_current
        schedule = torch.from_numpy(schedule_np).to(dev)
        sim.reset_state()
        spike_counts, _ = sim.run_with_schedule(schedule, rest_steps=0)
        exc_counts = spike_counts[exc_gid_np]

        class_votes = np.zeros(10)
        for ni, count in enumerate(exc_counts):
            class_votes[assignments[ni]] += count
        predictions.append(int(np.argmax(class_votes)))
        labels.append(label)

        if (img_idx + 1) % 200 == 0:
            correct = sum(p == l for p, l in zip(predictions, labels))
            acc = correct / len(predictions) * 100
            elapsed = time.perf_counter() - t_start
            print(f"  [{img_idx + 1}/{n_images}] acc: {acc:.1f}%, "
                  f"{(img_idx + 1) / elapsed:.1f} img/s")

    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(predictions) * 100

def classify_dot(sim, test_set, n_exc, assignments, exc_gid_t, pixel_gid_t,
                 max_images=None):
    n_images = len(test_set) if max_images is None else min(max_images, len(test_set))
    dev = sim.device
    predictions, labels = [], []

    for img_idx in range(n_images):
        image, label = test_set[img_idx]
        exc_input = dot_product_batch(sim, image.view(-1).to(dev), pixel_gid_t, exc_gid_t)
        class_votes = np.zeros(10)
        for ni, response in enumerate(exc_input):
            class_votes[assignments[ni]] += response
        predictions.append(int(np.argmax(class_votes)))
        labels.append(label)

    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(predictions) * 100

def visualize_receptive_fields(sim, input_pop, exc_pop, n_exc, assignments,
                               output_dir="results"):
    if not HAS_MATPLOTLIB:
        print("matplotlib not available")
        return
    os.makedirs(output_dir, exist_ok=True)
    placement = sim._compiled.placement

    pixel_gid_to_px = {}
    for px in range(784):
        cn = placement.neuron_map.get((input_pop.id, px))
        if cn:
            pixel_gid_to_px[cn[0] * NEURONS_PER_CORE + cn[1]] = px

    exc_gid_to_idx = {}
    for i in range(n_exc):
        cn = placement.neuron_map.get((exc_pop.id, i))
        if cn:
            exc_gid_to_idx[cn[0] * NEURONS_PER_CORE + cn[1]] = i

    crow = sim._soma_crow.cpu().numpy()
    col = sim._soma_col.cpu().numpy()
    val = sim._W_soma.values().cpu().numpy()

    W = np.zeros((n_exc, 784))
    for tgt_gid in range(sim._n):
        if tgt_gid not in exc_gid_to_idx:
            continue
        ei = exc_gid_to_idx[tgt_gid]
        start, end = int(crow[tgt_gid]), int(crow[tgt_gid + 1])
        for idx in range(start, end):
            src_gid = int(col[idx])
            if src_gid in pixel_gid_to_px:
                W[ei, pixel_gid_to_px[src_gid]] = val[idx]

    cols = min(10, n_exc)
    rows = (n_exc + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n_exc:
            rf = W[i].reshape(28, 28)
            ax.imshow(rf, cmap='hot', interpolation='nearest')
            ax.set_title(f"d={assignments[i]}", fontsize=7)
        ax.axis('off')

    plt.suptitle("Receptive Fields (d=assigned digit)", fontsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, "receptive_fields.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(W.flatten(), bins=100, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Weight")
    ax.set_ylabel("Count")
    ax.set_title("Weight Distribution")
    path = os.path.join(output_dir, "weight_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

def main():
    builtins.print = functools.partial(print, flush=True)

    parser = argparse.ArgumentParser(description="MNIST SNN Classification")
    parser.add_argument("--n-exc", type=int, default=39)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-images", type=int, default=10000)
    parser.add_argument("--test-images", type=int, default=1000)
    parser.add_argument("--presentation-time", type=int, default=50)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--eta-ltp", type=float, default=0.05)
    parser.add_argument("--eta-ltd", type=float, default=0.005)
    parser.add_argument("--k-winners", type=int, default=1)
    parser.add_argument("--weight-norm", type=float, default=10000)
    parser.add_argument("--ior-frac", type=float, default=0.0)
    parser.add_argument("--ior-decay", type=float, default=0.95)
    parser.add_argument("--exc-threshold", type=int, default=5000)
    parser.add_argument("--inh-weight", type=int, default=-300)
    parser.add_argument("--stim-current", type=int, default=200)
    args = parser.parse_args()

    n_exc = args.n_exc

    print("MNIST SNN (prototype init + IOR competitive learning)")
    print(f"  n_exc={n_exc}, epochs={args.epochs}, "
          f"train={args.train_images}/epoch, test={args.test_images}")
    print(f"  eta_ltp={args.eta_ltp}, eta_ltd={args.eta_ltd}, "
          f"k={args.k_winners}, ior={args.ior_frac}/{args.ior_decay}")
    print()

    print("Loading MNIST...")
    train_set, test_set = load_mnist(args.data_dir)

    print("\nBuilding network...")
    net, input_pop, exc_pop, inh_pop = build_mnist_network(
        n_exc=n_exc, exc_threshold=args.exc_threshold,
        inh_weight=args.inh_weight)

    print("\nDeploying to GPU...")
    if not torch.cuda.is_available():
        print("CUDA not available!")
        sys.exit(1)
    device = torch.device(args.device) if args.device else None
    sim = nc.GpuSimulator(device=device)
    sim.deploy(net)
    print(f"  GPU: {torch.cuda.get_device_name(sim.device)}")

    exc_gid_np, exc_gid_t, pixel_gid_np, pixel_gid_t = \
        compute_gid_arrays(sim, input_pop, exc_pop)

    sim.set_stdp_mask(set(pixel_gid_np.tolist()))

    print("\n  Initializing with prototype images...")
    prototype_initialize(sim, train_set, n_exc, exc_gid_t, pixel_gid_t,
                         args.weight_norm)

    test_img, test_label = train_set[0]
    test_input = dot_product_batch(sim, test_img.view(-1).to(sim.device),
                                   pixel_gid_t, exc_gid_t)
    top3 = np.argsort(test_input)[-3:][::-1]
    print(f"  Dynamics check (digit {test_label}): "
          f"max_dot={test_input[top3[0]]:.0f}, "
          f"min_dot={test_input.min():.0f}, "
          f"ratio={test_input[top3[0]] / max(1, test_input.min()):.1f}x")

    ior = None
    accuracies_dot = []
    accuracies_snn = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        winner_class_counts, ior = train_epoch(
            sim, train_set, n_exc, exc_gid_t, pixel_gid_t,
            max_images=args.train_images, epoch=epoch,
            weight_norm_target=args.weight_norm,
            eta_ltp=args.eta_ltp, eta_ltd=args.eta_ltd,
            k_winners=args.k_winners,
            ior=ior, ior_frac=args.ior_frac, ior_decay=args.ior_decay,
        )
        sim.normalize_learnable_weights(args.weight_norm, target_gids=exc_gid_t)

        print("\n  Winner-count assignment:")
        assign_wc = assign_neurons(winner_class_counts, n_exc)

        print("\n  Dot-product assignment:")
        assign_dp = assign_neurons_dot(sim, train_set, n_exc, exc_gid_t,
                                        pixel_gid_t, n_images=5000)

        acc_wc = classify_dot(sim, test_set, n_exc, assign_wc,
                              exc_gid_t, pixel_gid_t,
                              max_images=args.test_images)
        acc_dp = classify_dot(sim, test_set, n_exc, assign_dp,
                              exc_gid_t, pixel_gid_t,
                              max_images=args.test_images)
        print(f"  Dot accuracy: winner-count={acc_wc:.1f}%, "
              f"dot-assign={acc_dp:.1f}%")

        assignments = assign_dp if acc_dp >= acc_wc else assign_wc
        acc_dot = max(acc_wc, acc_dp)
        accuracies_dot.append(acc_dot)

        print(f"\n  SNN inference ({args.test_images} images)...")
        sim._build_weight_matrices(sim._n)
        acc_snn = classify_snn(sim, test_set, n_exc, assignments,
                               exc_gid_np, pixel_gid_np,
                               presentation_time=args.presentation_time,
                               max_images=args.test_images,
                               stim_current=args.stim_current)
        accuracies_snn.append(acc_snn)
        print(f"  SNN accuracy: {acc_snn:.1f}%")

    print(f"\nResults:")
    for i in range(len(accuracies_dot)):
        print(f"  Epoch {i + 1}: dot={accuracies_dot[i]:.1f}%, snn={accuracies_snn[i]:.1f}%")
    print(f"  Best: dot={max(accuracies_dot):.1f}%, snn={max(accuracies_snn):.1f}%")

    if args.visualize:
        print("\nVisualization...")
        output_dir = os.path.join(os.path.dirname(__file__), "..", "results")
        visualize_receptive_fields(sim, input_pop, exc_pop, n_exc,
                                   assignments, output_dir)

    sim.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
