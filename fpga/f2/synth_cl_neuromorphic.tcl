source ${HDK_SHELL_DIR}/build/scripts/synth_cl_header.tcl

read_verilog -sv [ list \
  ${src_post_enc_dir}/cl_neuromorphic.sv \
]

read_verilog [ list \
  ${src_post_enc_dir}/cl_neuromorphic_defines.vh \
  ${src_post_enc_dir}/async_fifo.v \
  ${src_post_enc_dir}/axi_uart_bridge.v \
  ${src_post_enc_dir}/sram.v \
  ${src_post_enc_dir}/spike_fifo.v \
  ${src_post_enc_dir}/scalable_core_v2.v \
  ${src_post_enc_dir}/neuromorphic_mesh.v \
  ${src_post_enc_dir}/async_noc_mesh.v \
  ${src_post_enc_dir}/async_router.v \
  ${src_post_enc_dir}/chip_link.v \
  ${src_post_enc_dir}/host_interface.v \
  ${src_post_enc_dir}/neuromorphic_top.v \
  ${src_post_enc_dir}/rv32i_core.v \
  ${src_post_enc_dir}/rv32im_cluster.v \
  ${src_post_enc_dir}/multi_chip_router.v \
]

read_xdc [ list \
  ${constraints_dir}/cl_synth_user.xdc \
  ${constraints_dir}/cl_timing_user.xdc \
]
set_property PROCESSING_ORDER LATE [get_files cl_synth_user.xdc]
set_property PROCESSING_ORDER LATE [get_files cl_timing_user.xdc]

update_compile_order -fileset sources_1

synth_design -mode out_of_context \
             -top ${CL} \
             -verilog_define XSDB_SLV_DIS \
             -part ${DEVICE_TYPE} \
             -keep_equivalent_registers

source ${HDK_SHELL_DIR}/build/scripts/synth_cl_footer.tcl
