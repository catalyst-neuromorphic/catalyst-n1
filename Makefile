# Neuromorphic Chip - Build & Simulation Makefile
# Usage:
#   make sim      - Compile and run simulation
#   make waves    - Open waveform viewer
#   make synth    - Synthesize with Yosys (gate-level)
#   make clean    - Clean build artifacts

# Source files
RTL_DIR = rtl
TB_DIR  = tb
SIM_DIR = sim

RTL_SRC = $(RTL_DIR)/lif_neuron.v $(RTL_DIR)/synapse.v $(RTL_DIR)/neuron_core.v
TB_SRC  = $(TB_DIR)/tb_neuron_core.v

# Simulation
SIM_OUT = $(SIM_DIR)/neuron_core_sim
VCD_OUT = $(SIM_DIR)/neuron_core.vcd

.PHONY: sim waves synth clean

sim: $(RTL_SRC) $(TB_SRC)
	@mkdir -p $(SIM_DIR)
	iverilog -o $(SIM_OUT) -I $(RTL_DIR) $(RTL_SRC) $(TB_SRC)
	cd $(SIM_DIR) && vvp ../$(SIM_OUT)

waves: $(VCD_OUT)
	gtkwave $(VCD_OUT) &

synth:
	@mkdir -p synth
	yosys -p "read_verilog $(RTL_SRC); synth -top neuron_core; stat; write_json synth/neuron_core.json" 2>&1 | tail -30

clean:
	rm -rf $(SIM_DIR)/*.vcd $(SIM_DIR)/neuron_core_sim synth/*.json
