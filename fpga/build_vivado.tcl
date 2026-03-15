# ============================================================================
# Vivado Non-Project Mode Build Script
# ============================================================================
# Target: Arty A7-100T (xc7a100tcsg324-1)
# Usage:  vivado -mode batch -source fpga/build_vivado.tcl
# ============================================================================

# ---- Configuration ----
set part        "xc7a100tcsg324-1"
set top         "fpga_top"
set build_dir   "fpga/build"
set bit_file    "${build_dir}/neuromorphic.bit"

# ---- Create build directory ----
file mkdir $build_dir

# ---- Read RTL sources ----
read_verilog {
    rtl/sram.v
    rtl/spike_fifo.v
    rtl/uart_tx.v
    rtl/uart_rx.v
    rtl/scalable_core_v2.v
    rtl/neuromorphic_mesh.v
    rtl/async_noc_mesh.v
    rtl/async_router.v
    rtl/sync_tree.v
    rtl/chip_link.v
    rtl/host_interface.v
    rtl/neuromorphic_top.v
    fpga/fpga_top.v
}

# ---- Read constraints ----
read_xdc fpga/arty_a7.xdc

# ---- Synthesis ----
puts "========================================"
puts "  SYNTHESIS"
puts "========================================"
synth_design -top $top -part $part \
    -flatten_hierarchy rebuilt \
    -directive Default

# Report utilization after synthesis
report_utilization -file ${build_dir}/synth_utilization.rpt
report_timing_summary -file ${build_dir}/synth_timing.rpt

# ---- Optimization ----
puts "========================================"
puts "  OPTIMIZATION"
puts "========================================"
opt_design

# ---- Placement ----
puts "========================================"
puts "  PLACEMENT"
puts "========================================"
place_design -directive Explore

# Report utilization after placement
report_utilization -file ${build_dir}/place_utilization.rpt

# ---- Routing ----
puts "========================================"
puts "  ROUTING"
puts "========================================"
route_design -directive Explore

# ---- Reports ----
puts "========================================"
puts "  REPORTS"
puts "========================================"
report_utilization -file ${build_dir}/route_utilization.rpt
report_timing_summary -file ${build_dir}/route_timing.rpt -max_paths 10
report_power -file ${build_dir}/power.rpt
report_drc -file ${build_dir}/drc.rpt
report_methodology -file ${build_dir}/methodology.rpt

# Check timing
set timing_slack [get_property SLACK [get_timing_paths -max_paths 1]]
puts "Worst slack: ${timing_slack} ns"
if {$timing_slack < 0} {
    puts "WARNING: Timing not met! Worst negative slack: ${timing_slack} ns"
}

# ---- Generate Bitstream ----
puts "========================================"
puts "  BITSTREAM"
puts "========================================"
write_bitstream -force $bit_file

# ---- Summary ----
puts ""
puts "========================================"
puts "  BUILD COMPLETE"
puts "========================================"
puts "  Bitstream: $bit_file"
puts "  Reports:   ${build_dir}/"
puts ""
puts "  To program the FPGA:"
puts "    open_hw_manager"
puts "    connect_hw_server"
puts "    open_hw_target"
puts "    set_property PROGRAM.FILE {${bit_file}} [current_hw_device]"
puts "    program_hw_devices"
puts "========================================"
