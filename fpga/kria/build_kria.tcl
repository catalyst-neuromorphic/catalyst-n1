set script_dir  [file dirname [file normalize [info script]]]
set project_dir "${script_dir}/build"
set part        "xczu5ev-sfvc784-2-i"
set rtl_dir     "[file normalize ${script_dir}/../../rtl]"
set kria_dir    $script_dir

set mode "full"
if {[llength $argv] > 0} {
    set mode [lindex $argv 0]
}

file mkdir $project_dir
create_project catalyst_kria_n1 $project_dir -part $part -force

add_files -norecurse [glob ${rtl_dir}/*.v]
add_files -norecurse ${kria_dir}/kria_neuromorphic.v
update_compile_order -fileset sources_1

if {$mode eq "synth_only"} {
    set_property top kria_neuromorphic [current_fileset]
    update_compile_order -fileset sources_1

    launch_runs synth_1 -jobs 4
    wait_on_run synth_1
    open_run synth_1

    report_utilization -file ${project_dir}/synth_utilization.rpt
    report_utilization -hierarchical -file ${project_dir}/synth_utilization_hier.rpt
    report_timing_summary -file ${project_dir}/synth_timing.rpt
    report_utilization -return_string

    close_project
    exit
}

close_project
