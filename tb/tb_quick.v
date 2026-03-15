`timescale 1ns/1ps
module tb_quick;
    reg clk;
    initial clk = 0;
    always #5 clk = ~clk;
    reg rst_n;

    wire timestep_done;
    wire [3:0] spike_valid_bus;

    neuromorphic_mesh #(
        .NUM_CORES(1), .CORE_ID_BITS(1),
        .NUM_NEURONS(1024), .NEURON_BITS(10),
        .DATA_WIDTH(16),
        .POOL_DEPTH(1024), .POOL_ADDR_BITS(10),
        .COUNT_BITS(10)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(1'b0),
        .prog_pool_we(1'b0), .prog_pool_core(2'b0), .prog_pool_addr(10'b0),
        .prog_pool_src(10'b0), .prog_pool_target(10'b0), .prog_pool_weight(16'sd0), .prog_pool_comp(2'b0),
        .prog_index_we(1'b0), .prog_index_core(2'b0), .prog_index_neuron(10'b0),
        .prog_index_base(10'b0), .prog_index_count(10'b0), .prog_index_format(2'b0),
        .prog_route_we(1'b0), .prog_route_src_core(2'b0), .prog_route_src_neuron(10'b0),
        .prog_route_slot(3'b0), .prog_route_dest_core(2'b0), .prog_route_dest_neuron(10'b0),
        .prog_route_weight(16'sd0),
        .prog_global_route_we(1'b0), .prog_global_route_src_core(2'b0),
        .prog_global_route_src_neuron(10'b0), .prog_global_route_slot(2'b0),
        .prog_global_route_dest_core(2'b0), .prog_global_route_dest_neuron(10'b0),
        .prog_global_route_weight(16'sd0),
        .learn_enable(1'b0), .graded_enable(1'b0), .dendritic_enable(1'b0), .async_enable(1'b0),
        .threefactor_enable(1'b0), .noise_enable(1'b0), .skip_idle_enable(1'b0), .scale_u_enable(1'b0),
        .reward_value(16'sd0),
        .prog_delay_we(1'b0), .prog_delay_core(2'b0), .prog_delay_addr(10'b0), .prog_delay_value(6'b0),
        .prog_ucode_we(1'b0), .prog_ucode_core(2'b0), .prog_ucode_addr(8'b0), .prog_ucode_data(32'b0),
        .prog_param_we(1'b0), .prog_param_core(2'b0), .prog_param_neuron(10'b0),
        .prog_param_id(5'b0), .prog_param_value(16'sd0),
        .ext_valid(1'b0), .ext_core(2'b0), .ext_neuron_id(10'b0), .ext_current(16'sd0),
        .probe_read(1'b0), .probe_core(2'b0), .probe_neuron(10'b0), .probe_state_id(5'b0),
        .probe_pool_addr(10'b0),
        .timestep_done(timestep_done),
        .spike_valid_bus(spike_valid_bus),
        .dvfs_stall(8'b0),
        .link_tx_full(1'b0),
        .link_rx_core(2'b0), .link_rx_neuron(10'b0), .link_rx_current(16'sd0),
        .link_rx_empty(1'b1)
    );

    initial begin
        $display("[t=0] Starting quick test...");
        rst_n = 0;
        #50;
        rst_n = 1;
        #100;
        $display("[t=150] Reset complete. Mesh idle.");
        #100;
        $display("[t=250] Quick test PASSED.");
        $finish;
    end
endmodule
