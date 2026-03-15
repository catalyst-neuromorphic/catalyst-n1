# Neurocore SDK

Python SDK for the Catalyst N1 neuromorphic processor.

## Installation

```bash
pip install -e .
```

For GPU simulation (optional):
```bash
pip install torch  # PyTorch with CUDA support
```

## Quick Start

```python
import neurocore as nc

# Build a network
net = nc.Network()
inp = net.population(100, params={'threshold': 1000, 'leak': 10}, label='input')
hid = net.population(50, params={'threshold': 1000, 'leak': 5}, label='hidden')
out = net.population(10, params={'threshold': 1000, 'leak': 5}, label='output')

net.connect(inp, hid, weight=500, probability=0.3)
net.connect(hid, out, weight=400, probability=0.5)

# Simulate
sim = nc.Simulator()
sim.deploy(net)

# Inject spikes and run
for t in range(100):
    sim.inject(inp, neuron_ids=[0, 5, 10], current=1500)
    sim.step()

# Analyze results
result = sim.get_result()
result.raster_plot(show=True)
```

## Backends

| Backend | Import | Description |
|---------|--------|-------------|
| `Simulator` | `neurocore.Simulator` | CPU reference simulator (LIF neurons) |
| `GpuSimulator` | `neurocore.GpuSimulator` | PyTorch CUDA accelerated (4-8x speedup at 4K+ neurons) |
| `Chip` | `neurocore.Chip` | UART interface to FPGA (Arty A7) |
| `F2Backend` | `neurocore.f2.F2Backend` | AWS F2 FPGA via PCIe MMIO |

All backends share the same `deploy(net)` / `step()` / `get_result()` API.

## Package Structure

```
neurocore/
  __init__.py          # Public API exports
  network.py           # Network, Population, Connection
  compiler.py          # Network -> hardware instructions
  simulator.py         # CPU LIF simulator
  gpu_simulator.py     # PyTorch GPU simulator
  chip.py              # UART FPGA backend
  f2.py                # AWS F2 PCIe backend
  result.py            # Spike recording and analysis
  analysis.py          # Raster plots, firing rates, ISI
  topology.py          # all_to_all, random, small_world, ring
  microcode.py         # Learning rule microcode compiler
  constants.py         # Hardware limits (WEIGHT_MIN/MAX, etc.)
  exceptions.py        # NeuroError, CompileError, etc.
```

## Benchmarks

```
benchmarks/
  shd_train.py         # Spiking Heidelberg Digits (surrogate gradient)
  shd_deploy.py        # SHD model quantization and deployment
  shd_loader.py        # SHD dataset loader (HDF5)
  stress_test.py       # SDK stress tests (saturation, stability, fan-out)
  scaling_benchmark.py # Neuron count scaling performance
  gpu_benchmark.py     # CPU vs GPU simulator comparison
```

### SHD Benchmark

Train a spiking neural network on spoken digit classification:

```bash
# Download dataset (first run)
python benchmarks/shd_train.py --data-dir benchmarks/data/shd --epochs 200

# Evaluate quantization for hardware deployment
python benchmarks/shd_deploy.py --checkpoint benchmarks/shd_model.pt --data-dir benchmarks/data/shd
```

## Tests

```bash
pytest tests/ -v        # 168 tests
pytest tests/ -v -k gpu # GPU tests only (requires CUDA)
```

## Hardware Requirements

- **CPU Simulator**: Python 3.9+, NumPy
- **GPU Simulator**: PyTorch 2.0+ with CUDA
- **Chip backend**: pyserial, FPGA with UART connection
- **F2 backend**: AWS F2 instance, fpga_mgmt library
