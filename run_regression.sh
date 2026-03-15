#!/bin/bash
RTL="rtl/sram.v rtl/spike_fifo.v rtl/uart_tx.v rtl/uart_rx.v rtl/scalable_core_v2.v rtl/neuromorphic_mesh.v rtl/host_interface.v rtl/neuromorphic_top.v rtl/async_router.v rtl/async_noc_mesh.v rtl/rv32i_core.v rtl/multi_chip_router.v rtl/rv32im_cluster.v rtl/async_fifo.v rtl/axi_uart_bridge.v rtl/chip_link.v fpga/fpga_top.v"

PASS=0
FAIL=0
for tb in tb/tb_p13a.v tb/tb_p13b.v tb/tb_p13c.v tb/tb_p14_noise.v tb/tb_p15_traces.v tb/tb_p17_delays.v tb/tb_p18_formats.v tb/tb_p19_microcode.v tb/tb_p20_hierarchical.v tb/tb_p21a_dendrites.v tb/tb_p21b_observe.v tb/tb_p21c_power.v tb/tb_p21d_learning.v tb/tb_p21e_chiplink.v tb/tb_p22a_cuba.v tb/tb_p22b_compartments.v tb/tb_p22c_learning.v tb/tb_p22d_axontypes.v tb/tb_p22e_noc.v tb/tb_p22f_riscv.v tb/tb_p22g_multichip.v tb/tb_p22h_power.v tb/tb_p23a_neuron_arith.v tb/tb_p23b_comp_synapse.v tb/tb_p23c_scale.v tb/tb_p23d_riscv.v tb/tb_p24_final.v tb/tb_p25_final.v tb/tb_stress.v; do
    tb_mod=$(basename "$tb" .v)
    iverilog -g2012 -DSIMULATION -s "$tb_mod" -o test_reg.vvp $RTL $tb 2>&1
    if [ $? -eq 0 ]; then
        timeout 120 vvp test_reg.vvp 2>&1 | grep -iE "pass|fail|result"
        PASS=$((PASS+1))
    else
        echo "COMPILE ERROR: $tb"
        FAIL=$((FAIL+1))
    fi
done
echo "$PASS compiled, $FAIL failed"
