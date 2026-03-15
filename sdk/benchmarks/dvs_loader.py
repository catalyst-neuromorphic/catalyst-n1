import os
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    raise ImportError("PyTorch required: pip install torch")

try:
    import tonic
    import tonic.transforms as transforms
except ImportError:
    raise ImportError("tonic required: pip install tonic")

N_CHANNELS = 2048
N_CLASSES = 11
SENSOR_SIZE = (128, 128, 2)
DS_FACTOR = 4
DS_SIZE = (32, 32, 2)

def get_dvs_transform(dt=10e-3, duration=1.5):
    n_bins = int(duration / dt)
    return transforms.Compose([
        transforms.Downsample(spatial_factor=1.0 / DS_FACTOR),
        transforms.ToFrame(
            sensor_size=DS_SIZE,
            n_time_bins=n_bins,
        ),
    ])

class DVSGestureDataset(Dataset):

    def __init__(self, data_dir="data/dvs_gesture", train=True, dt=10e-3, duration=1.5):
        transform = get_dvs_transform(dt=dt, duration=duration)

        self._tonic_ds = tonic.datasets.DVSGesture(
            save_to=data_dir,
            train=train,
            transform=transform,
        )

        self.n_bins = int(duration / dt)
        self.dt = dt
        self.duration = duration

    def __len__(self):
        return len(self._tonic_ds)

    def __getitem__(self, idx):
        frames, label = self._tonic_ds[idx]
        frames = np.array(frames, dtype=np.float32)

        if frames.ndim == 4:
            T = frames.shape[0]
            frames = frames.reshape(T, -1)
        elif frames.ndim == 3:
            T = frames.shape[0]
            frames = frames.reshape(T, -1)

        if frames.shape[0] > self.n_bins:
            frames = frames[:self.n_bins]
        elif frames.shape[0] < self.n_bins:
            pad = np.zeros((self.n_bins - frames.shape[0], frames.shape[1]), dtype=np.float32)
            frames = np.concatenate([frames, pad], axis=0)

        frames = (frames > 0).astype(np.float32)

        return torch.from_numpy(frames), int(label)

def collate_fn(batch):
    inputs, labels = zip(*batch)
    return torch.stack(inputs), torch.tensor(labels, dtype=torch.long)
