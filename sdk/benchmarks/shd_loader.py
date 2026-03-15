"""SHD (Spiking Heidelberg Digits) dataset loader for neuromorphic benchmarks.

Downloads HDF5 files from Zenodo, converts variable-length spike events
to fixed-size dense binary tensors suitable for PyTorch training.

700 input channels (cochlea model), 20 classes (digits 0-9 in German+English).
"""

import os
import urllib.request
import gzip
import shutil
import numpy as np

try:
    import h5py
except ImportError:
    raise ImportError("h5py required: pip install h5py")

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    raise ImportError("PyTorch required: pip install torch")


SHD_URLS = {
    "train": "https://compneuro.net/datasets/shd_train.h5.gz",
    "test": "https://compneuro.net/datasets/shd_test.h5.gz",
}

N_CHANNELS = 700   # SHD cochlea channels
N_CLASSES = 20      # spoken digits 0-9 in German + English


def download_shd(data_dir="data/shd"):
    """Download SHD train/test HDF5 files from Zenodo if not present."""
    os.makedirs(data_dir, exist_ok=True)

    for split, url in SHD_URLS.items():
        h5_path = os.path.join(data_dir, f"shd_{split}.h5")
        gz_path = h5_path + ".gz"

        if os.path.exists(h5_path):
            continue

        print(f"Downloading SHD {split} set from {url} ...")
        try:
            urllib.request.urlretrieve(url, gz_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download {url}: {e}\n"
                f"Download manually from https://zenodo.org/records/4319560 "
                f"and place shd_train.h5 / shd_test.h5 in {data_dir}/")

        print(f"Extracting {gz_path} ...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(h5_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)
        print(f"  Saved to {h5_path}")

    return data_dir


def spikes_to_dense(times, units, n_channels=N_CHANNELS, dt=4e-3, max_time=1.0):
    """Convert spike event lists to a dense binary tensor.

    Args:
        times: array of spike times in seconds
        units: array of channel indices (0 to n_channels-1)
        n_channels: number of input channels (700 for SHD)
        dt: time bin width in seconds (4ms -> 250 bins)
        max_time: maximum time window in seconds

    Returns:
        dense: (T, n_channels) float32 array with 1.0 at spike locations
    """
    n_bins = int(max_time / dt)
    dense = np.zeros((n_bins, n_channels), dtype=np.float32)

    if not times:
        return dense

    bin_indices = np.clip((times / dt).astype(int), 0, n_bins - 1)
    unit_indices = np.clip(units.astype(int), 0, n_channels - 1)
    dense[bin_indices, unit_indices] = 1.0
    return dense


class SHDDataset(Dataset):
    """PyTorch Dataset for Spiking Heidelberg Digits.

    Each sample is converted to a dense binary tensor (T, 700) on first access.
    """

    def __init__(self, data_dir="data/shd", split="train", dt=4e-3, max_time=1.0):
        h5_path = os.path.join(data_dir, f"shd_{split}.h5")
        if not os.path.exists(h5_path):
            download_shd(data_dir)

        with h5py.File(h5_path, 'r') as f:
            self.times = [np.array(t) for t in f['spikes']['times']]
            self.units = [np.array(u) for u in f['spikes']['units']]
            self.labels = np.array(f['labels'])

        self.dt = dt
        self.max_time = max_time
        self.n_bins = int(max_time / dt)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        dense = spikes_to_dense(
            self.times[idx], self.units[idx],
            dt=self.dt, max_time=self.max_time,
        )
        return torch.from_numpy(dense), int(self.labels[idx])


def collate_fn(batch):
    """Collate with uniform time length (all samples use same max_time)."""
    inputs, labels = zip(*batch)
    return torch.stack(inputs), torch.tensor(labels, dtype=torch.long)
