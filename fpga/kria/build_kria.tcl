# ============================================================================
# Vivado Build Script — Kria KV260 Target — Catalyst N1 (Loihi 1 Parity)
# ============================================================================
# Usage: vivado -mode batch -source fpga/kria/build_kria.tcl -tclargs synth_only
# ============================================================================

set script_dir  [file dirname [file normalize [info script]]]
set project_dir "${script_dir}/build"
set part        "xczu5ev-sfvc784-2-i"
set rtl_dir     "[file normalize ${script_dir}/../../rtl]"
set kria_dir    $script_dir

set mode "full"
if {[llength $argv] > 0} {
    set mode [lindex $argv 0]
}

puts "============================================"
puts "  Catalyst N1 — Kria KV260 Build"
puts "  Mode: $mode"
puts "  Part: $part"
puts "============================================"

file mkdir $project_dir
create_project catalyst_kria_n1 $project_dir -part $part -force

set rtl_files [list \
    ${rtl_dir}/sram.v \
    ${rtl_dir}/spike_fifo.v \
    ${rtl_dir}/async_fifo.v \
    ${rtl_dir}/uart_tx.v \
    ${rtl_dir}/uart_rx.v \
    ${rtl_dir}/scalable_core_v2.v \
    ${rtl_dir}/neuromorphic_mesh.v \
    ${rtl_dir}/async_noc_mesh.v \
    ${rtl_dir}/async_router.v \
    ${rtl_dir}/sync_tree.v \
    ${rtl_dir}/chip_link.v \
    ${rtl_dir}/host_interface.v \
    ${rtl_dir}/axi_uart_bridge.v \
    ${rtl_dir}/neuromorphic_top.v \
    ${kria_dir}/kria_neuromorphic.v \
]
add_files -norecurse $rtl_files
update_compile_order -fileset sources_1

if {$mode eq "synth_only"} {
    puts "============================================"
    puts "  SYNTHESIS-ONLY MODE"
    puts "============================================"

    set_property top kria_neuromorphic [current_fileset]
    update_compile_order -fileset sources_1

    launch_runs synth_1 -jobs 4
    wait_on_run synth_1
    open_run synth_1

    report_utilization -file ${project_dir}/synth_utilization.rpt
    report_utilization -hierarchical -file ${project_dir}/synth_utilization_hier.rpt
    report_timing_summary -file ${project_dir}/synth_timing.rpt

    puts ""
    puts "============================================"
    puts "  N1 SYNTHESIS COMPLETE"
    puts "============================================"
    report_utilization -return_string

    close_project
    exit
}

close_project
