# ============================================================================
# Vivado Implementation Script — Kria K26 — Catalyst N1 (Loihi 1 Parity)
# ============================================================================
# Opens existing synthesis checkpoint and runs Place & Route + reports
# Usage: vivado -mode batch -source fpga/kria/run_impl.tcl
# ============================================================================

set script_dir  [file dirname [file normalize [info script]]]
set project_dir "${script_dir}/build"
set synth_dcp   "${project_dir}/catalyst_kria_n1.runs/synth_1/kria_neuromorphic.dcp"
set out_dir     "${project_dir}/impl_results"

file mkdir $out_dir

puts "============================================"
puts "  Catalyst N1 — Kria K26 Implementation"
puts "  Loading: $synth_dcp"
puts "============================================"

# Open synthesis checkpoint
open_checkpoint $synth_dcp

# Add clock constraint — Kria K26 PS provides 100 MHz PL clock
create_clock -period 10.000 -name sys_clk [get_ports s_axi_aclk]

# Set IO delay constraints (generic, for timing closure)
set_input_delay -clock sys_clk -max 2.0 [get_ports -filter {DIRECTION == IN && NAME != "s_axi_aclk"}]
set_output_delay -clock sys_clk -max 2.0 [get_ports -filter {DIRECTION == OUT}]

# Run implementation
puts "Running opt_design..."
opt_design

puts "Running place_design..."
place_design

puts "Running phys_opt_design..."
phys_opt_design

puts "Running route_design..."
route_design

# Save implemented checkpoint
write_checkpoint -force ${out_dir}/kria_n1_impl.dcp

# Generate reports
puts "Generating reports..."
report_timing_summary -file ${out_dir}/timing_summary.rpt
report_timing -max_paths 20 -file ${out_dir}/timing_paths.rpt
report_utilization -file ${out_dir}/utilization.rpt
report_utilization -hierarchical -file ${out_dir}/utilization_hier.rpt
report_power -file ${out_dir}/power.rpt
report_clock_utilization -file ${out_dir}/clock_utilization.rpt
report_design_analysis -file ${out_dir}/design_analysis.rpt

puts ""
puts "============================================"
puts "  N1 IMPLEMENTATION COMPLETE"
puts "============================================"
puts "Reports in: $out_dir"

# Print summary to console
report_timing_summary -return_string
report_utilization -return_string
report_power -return_string

close_design
exit
