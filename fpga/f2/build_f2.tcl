set cl_design_files [list \
    $CL_DIR/design/cl_neuromorphic_defines.vh \
    $CL_DIR/design/cl_neuromorphic.sv \
    $CL_DIR/design/axi_uart_bridge.v \
]

set neuro_rtl_files [list \
    $CL_DIR/design/sram.v \
    $CL_DIR/design/spike_fifo.v \
    $CL_DIR/design/async_fifo.v \
    $CL_DIR/design/scalable_core_v2.v \
    $CL_DIR/design/neuromorphic_mesh.v \
    $CL_DIR/design/async_noc_mesh.v \
    $CL_DIR/design/async_router.v \
    $CL_DIR/design/chip_link.v \
    $CL_DIR/design/host_interface.v \
    $CL_DIR/design/neuromorphic_top.v \
    $CL_DIR/design/rv32i_core.v \
    $CL_DIR/design/rv32im_cluster.v \
    $CL_DIR/design/multi_chip_router.v \
]

foreach f [concat $cl_design_files $neuro_rtl_files] {
    if {[file exists $f]} {
        read_verilog $f
    } else {
        puts "WARNING: File not found: $f"
    }
}

set_property verilog_define {} [current_fileset]
set_property include_dirs [list $CL_DIR/design] [current_fileset]
