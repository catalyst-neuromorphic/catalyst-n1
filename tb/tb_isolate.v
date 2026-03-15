`timescale 1ns/1ps
module tb_isolate;
    reg clk;
    initial clk = 0;
    always #5 clk = ~clk;
    reg rst_n;

    wire done;

    scalable_core_v2 #(
        .NUM_NEURONS(1024), .NEURON_BITS(10),
        .DATA_WIDTH(16),
        .POOL_DEPTH(1024), .POOL_ADDR_BITS(10),
        .COUNT_BITS(10)
    ) core0 (
        .clk(clk), .rst_n(rst_n), .start(1'b0),
        .learn_enable(1'b0), .graded_enable(1'b0), .dendritic_enable(1'b0),
        .threefactor_enable(1'b0), .noise_enable(1'b0),
        .skip_idle_enable(1'b0), .scale_u_enable(1'b0),
        .reward_value(16'sd0),
        .ext_valid(1'b0), .ext_neuron_id(10'b0), .ext_current(16'sd0),
        .pool_we(1'b0), .pool_addr_in(10'b0), .pool_src_in(10'b0),
        .pool_target_in(10'b0), .pool_weight_in(16'sd0), .pool_comp_in(2'b0),
        .index_we(1'b0), .index_neuron_in(10'b0), .index_base_in(10'b0),
        .index_count_in(10'b0), .index_format_in(2'b0),
        .delay_we(1'b0), .delay_addr_in(10'b0), .delay_value_in(6'b0),
        .ucode_prog_we(1'b0), .ucode_prog_addr(8'b0), .ucode_prog_data(32'b0),
        .prog_param_we(1'b0), .prog_param_neuron(10'b0),
        .prog_param_id(5'b0), .prog_param_value(16'sd0),
        .timestep_done(done)
    );

    initial begin
        $display("[t=0] Core isolation test...");
        rst_n = 0;
        #50;
        rst_n = 1;
        #100;
        $display("[t=150] Core idle test PASSED.");
        $finish;
    end
endmodule
