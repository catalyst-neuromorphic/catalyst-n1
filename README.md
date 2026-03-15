# Catalyst N1

Open source parameterised neuromorphic processor. LIF neurons, STDP learning, XY mesh NoC, RISC-V management. Verilog RTL, validated on AWS F2 FPGA.

## Specifications

Core count, neuron count, and pool depth are compile-time parameters. Actual configuration depends on FPGA target.

| Parameter | Value |
|-----------|-------|
| Neuron model | Leaky Integrate-and-Fire (16-bit fixed-point) |
| Learning | STDP, 14-opcode programmable learning ISA |
| Network-on-Chip | Configurable XY mesh with multicast |
| Host interface | UART (FPGA) / AXI-Lite (F2) |
| Management | RV32IM RISC-V cluster |
| Clock | 62.5 MHz (F2), 100 MHz (simulation) |

### FPGA configurations

| Target | Cores | Neurons/core | Total neurons |
|--------|-------|-------------|---------------|
| AWS F2 (VU47P) | 16 | 1,024 | 16,384 |
| Arty A7 | 4 | 256 | 1,024 |
| Kria K26 | 2 | 256 | 512 |

## Directory Structure

```
catalyst-n1/
  rtl/           Verilog modules (core, NoC, memory, host, RISC-V)
  tb/            Testbenches
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
  rtl/host_interface.v rtl/neuromorphic_top.v \
  rtl/rv32i_core.v rtl/rv32im_cluster.v \
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

net.connect(inp, hid, weight=500, p=0.3)
net.connect(hid, out, weight=400, p=0.5)

sim = nc.Simulator()
sim.deploy(net)
sim.inject(inp, current=1500)
result = sim.run(100)
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
