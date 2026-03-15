set script_dir  [file dirname [file normalize [info script]]]
set project_dir "${script_dir}/build"
set synth_dcp   "${project_dir}/catalyst_kria_n1.runs/synth_1/kria_neuromorphic.dcp"
set out_dir     "${project_dir}/impl_results"

file mkdir $out_dir

open_checkpoint $synth_dcp

create_clock -period 10.000 -name sys_clk [get_ports s_axi_aclk]
set_input_delay -clock sys_clk -max 2.0 [get_ports -filter {DIRECTION == IN && NAME != "s_axi_aclk"}]
set_output_delay -clock sys_clk -max 2.0 [get_ports -filter {DIRECTION == OUT}]

opt_design
place_design
phys_opt_design
route_design

write_checkpoint -force ${out_dir}/kria_n1_impl.dcp

report_timing_summary -file ${out_dir}/timing_summary.rpt
report_timing -max_paths 20 -file ${out_dir}/timing_paths.rpt
report_utilization -file ${out_dir}/utilization.rpt
report_utilization -hierarchical -file ${out_dir}/utilization_hier.rpt
report_power -file ${out_dir}/power.rpt

report_timing_summary -return_string
report_utilization -return_string

close_design
exit
