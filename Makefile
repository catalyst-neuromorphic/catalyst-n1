RTL_DIR = rtl
TB_DIR  = tb
SIM_DIR = sim

RTL_SRC = $(wildcard $(RTL_DIR)/*.v)
TB_SRC  = $(TB_DIR)/tb_p24_final.v

SIM_OUT = $(SIM_DIR)/sim.vvp

.PHONY: sim waves clean

sim: $(RTL_SRC) $(TB_SRC)
	@mkdir -p $(SIM_DIR)
	iverilog -g2012 -DSIMULATION -o $(SIM_OUT) -I $(RTL_DIR) $(RTL_SRC) $(TB_SRC)
	vvp $(SIM_OUT)

waves:
	gtkwave $(SIM_DIR)/*.vcd &

clean:
	rm -rf $(SIM_DIR)/*.vcd $(SIM_DIR)/*.vvp
