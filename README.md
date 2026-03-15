# Catalyst N1

Open source 128-core neuromorphic processor with full mesh NoC, STDP learning, and RISC-V management. Verilog RTL, validated on FPGA.

## Specifications

| Parameter | Value |
|-----------|-------|
| Cores | 128 |
| Neurons per core | 1,024 |
| Total neurons | 131,072 |
| Neuron model | Leaky Integrate-and-Fire (16-bit fixed-point) |
| Synapse pool | 131,072 entries per core |
| Learning | STDP, 14-opcode programmable learning ISA |
| Network-on-Chip | Configurable XY mesh with multicast |
| Host interface | UART (FPGA) / AXI-Lite (F2) / PCIe MMIO |
| Management | RV32IM RISC-V cluster |
| Multi-chip | Chip link with routing table |
| Clock | 100 MHz (simulation default) |

## Directory Structure

```
catalyst-n1/
  rtl/           25 Verilog modules (core, NoC, memory, host, RISC-V)
  tb/            46 testbenches (unit, integration, regression)
  sdk/           Python SDK with CPU, GPU, and FPGA backends
  fpga/          FPGA build files (Arty A7, AWS F2, Kria K26)
  sim/           Simulation scripts and visualization
  Makefile       Compile and run simulation
```

## Simulation

Requires [Icarus Verilog](https://github.com/steveicarus/iverilog) (v12+).

```bash
# Compile and run basic simulation
make sim

# Run full regression (25 testbenches)
bash run_regression.sh

# Run a single testbench
iverilog -g2012 -DSIMULATION -o out.vvp \
  rtl/sram.v rtl/spike_fifo.v rtl/uart_tx.v rtl/uart_rx.v \
  rtl/scalable_core_v2.v rtl/neuromorphic_mesh.v \
  rtl/host_interface.v rtl/neuromorphic_top.v rtl/sync_tree.v \
  rtl/rv32i_core.v rtl/mmio_bridge.v rtl/rv32im_cluster.v \
  tb/tb_p24_final.v
vvp out.vvp

# View waveforms (requires GTKWave)
make waves
```

## SDK

Python SDK for building, simulating, and deploying spiking neural networks. See [`sdk/README.md`](sdk/README.md) for full documentation.

```bash
cd sdk
pip install -e .
```

```python
import neurocore as nc

net = nc.Network()
inp = net.population(100, params={'threshold': 1000, 'leak': 10}, label='input')
hid = net.population(50, params={'threshold': 1000, 'leak': 5}, label='hidden')
out = net.population(10, params={'threshold': 1000, 'leak': 5}, label='output')

net.connect(inp, hid, weight=500, probability=0.3)
net.connect(hid, out, weight=400, probability=0.5)

sim = nc.Simulator()
sim.deploy(net)

for t in range(100):
    sim.inject(inp, neuron_ids=[0, 5, 10], current=1500)
    sim.step()

result = sim.get_result()
result.raster_plot(show=True)
```

Four backends: CPU simulator, GPU simulator (PyTorch CUDA), FPGA via UART (Arty A7), AWS F2 via PCIe. All share the same API.

## FPGA

### Arty A7

```bash
# Vivado batch build
vivado -mode batch -source fpga/build_vivado.tcl
```

Constraints: `fpga/arty_a7.xdc`. Top module: `fpga/fpga_top.v`.

### AWS F2

```bash
# Build on F2 build instance
cd fpga/f2
bash run_build.sh
```

CL wrapper: `fpga/f2/cl_neuromorphic.sv`. Host driver: `fpga/f2_host.py`.

### Kria K26

```bash
vivado -mode batch -source fpga/kria/build_kria.tcl
```

Wrapper: `fpga/kria/kria_neuromorphic.v`.

## Benchmarks

SHD (Spiking Heidelberg Digits) spoken digit classification:

```bash
cd sdk
python benchmarks/shd_train.py --data-dir benchmarks/data/shd --epochs 200
python benchmarks/shd_deploy.py --checkpoint benchmarks/shd_model.pt --data-dir benchmarks/data/shd
```

Additional benchmarks in `sdk/benchmarks/`: DVS gesture recognition, XOR classification, temporal patterns, scaling, stress tests.

## Links

- [catalyst-neuromorphic.com](https://catalyst-neuromorphic.com) (work in progress)
- [Cloud API](https://github.com/catalyst-neuromorphic/catalyst-cloud-python) (work in progress)
- [Catalyst-Neurocore](https://github.com/catalyst-neuromorphic/catalyst-neurocore)

## License

Apache 2.0. See [LICENSE](LICENSE).
