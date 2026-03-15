# ============================================================================
# F2 Build Script — Source File List
# ============================================================================
#
# This script is sourced by the AWS HDK build flow.
# It adds our CL design sources to the Vivado project.
#
# Usage (within HDK environment):
#   source $CL_DIR/build/scripts/aws_build_dcp_from_cl.tcl
#
# The HDK flow expects CL sources in $CL_DIR/design/
# Copy all .v files there before running the build.
# ============================================================================

# ---- CL wrapper + bridge ----
set cl_design_files [list \
    $CL_DIR/design/cl_neuromorphic_defines.vh \
    $CL_DIR/design/cl_neuromorphic.v \
    $CL_DIR/design/axi_uart_bridge.v \
]

# ---- Neuromorphic RTL ----
set neuro_rtl_files [list \
    $CL_DIR/design/sram.v \
    $CL_DIR/design/spike_fifo.v \
    $CL_DIR/design/scalable_core_v2.v \
    $CL_DIR/design/neuromorphic_mesh.v \
    $CL_DIR/design/async_noc_mesh.v \
    $CL_DIR/design/async_router.v \
    $CL_DIR/design/sync_tree.v \
    $CL_DIR/design/chip_link.v \
    $CL_DIR/design/host_interface.v \
    $CL_DIR/design/neuromorphic_top.v \
    $CL_DIR/design/rv32i_core.v \
    $CL_DIR/design/rv32im_cluster.v \
    $CL_DIR/design/mmio_bridge.v \
    $CL_DIR/design/multi_chip_router.v \
]

# Note: uart_rx.v and uart_tx.v are NOT needed (BYPASS_UART=1).
# They would be optimized away anyway, but omitting them prevents
# Vivado lint warnings about unconnected modules.

# ---- Add all sources ----
foreach f [concat $cl_design_files $neuro_rtl_files] {
    if {[file exists $f]} {
        read_verilog $f
    } else {
        puts "WARNING: File not found: $f"
    }
}

# ---- Include path for defines ----
set_property verilog_define {} [current_fileset]
set_property include_dirs [list $CL_DIR/design] [current_fileset]
