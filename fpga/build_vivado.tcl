set part        "xc7a100tcsg324-1"
set top         "fpga_top"
set build_dir   "fpga/build"
set bit_file    "${build_dir}/neuromorphic.bit"

file mkdir $build_dir

read_verilog [glob rtl/*.v]
read_verilog fpga/fpga_top.v

read_xdc fpga/arty_a7.xdc

synth_design -top $top -part $part -flatten_hierarchy rebuilt -directive Default
report_utilization -file ${build_dir}/synth_utilization.rpt
report_timing_summary -file ${build_dir}/synth_timing.rpt

opt_design
place_design -directive Explore
report_utilization -file ${build_dir}/place_utilization.rpt

route_design -directive Explore
report_utilization -file ${build_dir}/route_utilization.rpt
report_timing_summary -file ${build_dir}/route_timing.rpt -max_paths 10
report_power -file ${build_dir}/power.rpt
report_drc -file ${build_dir}/drc.rpt

set timing_slack [get_property SLACK [get_timing_paths -max_paths 1]]
puts "Worst slack: ${timing_slack} ns"
if {$timing_slack < 0} {
    puts "WARNING: Timing not met! Worst negative slack: ${timing_slack} ns"
}

write_bitstream -force $bit_file
puts "Bitstream: $bit_file"
