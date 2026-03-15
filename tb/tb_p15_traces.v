`timescale 1ns/1ps

module tb_p15_traces;

    parameter NUM_CORES      = 4;
    parameter CORE_ID_BITS   = 2;
    parameter NUM_NEURONS    = 256;
    parameter NEURON_BITS    = 8;
    parameter DATA_WIDTH     = 16;
    parameter POOL_DEPTH     = 1024;
    parameter POOL_ADDR_BITS = 10;
    parameter COUNT_BITS     = 6;
    parameter REV_FANIN      = 16;
    parameter REV_SLOT_BITS  = 4;
    parameter ROUTE_FANOUT   = 8;
    parameter ROUTE_SLOT_BITS = 3;
    parameter CLK_PERIOD     = 10;

    reg clk, rst_n;
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    reg                         start;

    reg                         prog_pool_we;
    reg  [CORE_ID_BITS-1:0]    prog_pool_core;
    reg  [POOL_ADDR_BITS-1:0]  prog_pool_addr;
    reg  [NEURON_BITS-1:0]     prog_pool_src;
    reg  [NEURON_BITS-1:0]     prog_pool_target;
    reg  signed [DATA_WIDTH-1:0] prog_pool_weight;
    reg  [1:0]                  prog_pool_comp;

    reg                         prog_index_we;
    reg  [CORE_ID_BITS-1:0]    prog_index_core;
    reg  [NEURON_BITS-1:0]     prog_index_neuron;
    reg  [POOL_ADDR_BITS-1:0]  prog_index_base;
    reg  [COUNT_BITS-1:0]      prog_index_count;

    reg                         prog_route_we;
    reg  [CORE_ID_BITS-1:0]    prog_route_src_core;
    reg  [NEURON_BITS-1:0]     prog_route_src_neuron;
    reg  [ROUTE_SLOT_BITS-1:0] prog_route_slot;
    reg  [CORE_ID_BITS-1:0]    prog_route_dest_core;
    reg  [NEURON_BITS-1:0]     prog_route_dest_neuron;
    reg  signed [DATA_WIDTH-1:0] prog_route_weight;

    reg                         learn_enable;
    reg                         graded_enable;
    reg                         dendritic_enable;
    reg                         async_enable;
    reg                         threefactor_enable;
    reg  signed [DATA_WIDTH-1:0] reward_value;
    reg                         noise_enable;

    reg                         prog_param_we;
    reg  [CORE_ID_BITS-1:0]    prog_param_core;
    reg  [NEURON_BITS-1:0]     prog_param_neuron;
    reg  [2:0]                  prog_param_id;
    reg  signed [DATA_WIDTH-1:0] prog_param_value;

    reg                         ext_valid;
    reg  [CORE_ID_BITS-1:0]    ext_core;
    reg  [NEURON_BITS-1:0]     ext_neuron_id;
    reg  signed [DATA_WIDTH-1:0] ext_current;

    wire                        timestep_done;
    wire [NUM_CORES-1:0]        spike_valid_bus;
    wire [NUM_CORES*NEURON_BITS-1:0] spike_id_bus;
    wire [4:0]                  mesh_state_out;
    wire [31:0]                 total_spikes;
    wire [31:0]                 timestep_count;

    neuromorphic_mesh #(
        .NUM_CORES      (NUM_CORES),
        .CORE_ID_BITS   (CORE_ID_BITS),
        .NUM_NEURONS    (NUM_NEURONS),
        .NEURON_BITS    (NEURON_BITS),
        .DATA_WIDTH     (DATA_WIDTH),
        .POOL_DEPTH     (POOL_DEPTH),
        .POOL_ADDR_BITS (POOL_ADDR_BITS),
        .COUNT_BITS     (COUNT_BITS),
        .REV_FANIN      (REV_FANIN),
        .REV_SLOT_BITS  (REV_SLOT_BITS),
        .ROUTE_FANOUT   (ROUTE_FANOUT),
        .ROUTE_SLOT_BITS(ROUTE_SLOT_BITS),
        .THRESHOLD      (16'sd1000),
        .LEAK_RATE      (16'sd3),
        .REFRAC_CYCLES  (3)
    ) dut (
        .clk               (clk),
        .rst_n             (rst_n),
        .start             (start),
        .prog_pool_we      (prog_pool_we),
        .prog_pool_core    (prog_pool_core),
        .prog_pool_addr    (prog_pool_addr),
        .prog_pool_src     (prog_pool_src),
        .prog_pool_target  (prog_pool_target),
        .prog_pool_weight  (prog_pool_weight),
        .prog_pool_comp    (prog_pool_comp),
        .prog_index_we     (prog_index_we),
        .prog_index_core   (prog_index_core),
        .prog_index_neuron (prog_index_neuron),
        .prog_index_base   (prog_index_base),
        .prog_index_count  (prog_index_count),
        .prog_index_format (2'd0),
        .prog_route_we         (prog_route_we),
        .prog_route_src_core   (prog_route_src_core),
        .prog_route_src_neuron (prog_route_src_neuron),
        .prog_route_slot       (prog_route_slot),
        .prog_route_dest_core  (prog_route_dest_core),
        .prog_route_dest_neuron(prog_route_dest_neuron),
        .prog_route_weight     (prog_route_weight),
        .prog_global_route_we(1'b0),
        .prog_global_route_src_core({CORE_ID_BITS{1'b0}}),
        .prog_global_route_src_neuron({NEURON_BITS{1'b0}}),
        .prog_global_route_slot(2'b0),
        .prog_global_route_dest_core({CORE_ID_BITS{1'b0}}),
        .prog_global_route_dest_neuron({NEURON_BITS{1'b0}}),
        .prog_global_route_weight({DATA_WIDTH{1'b0}}),
        .learn_enable      (learn_enable),
        .graded_enable     (graded_enable),
        .dendritic_enable  (dendritic_enable),
        .async_enable      (async_enable),
        .threefactor_enable(threefactor_enable),
        .reward_value      (reward_value),
        .noise_enable      (noise_enable),
        .prog_delay_we     (1'b0),
        .prog_delay_core   ({CORE_ID_BITS{1'b0}}),
        .prog_delay_addr   ({POOL_ADDR_BITS{1'b0}}),
        .prog_delay_value  (6'd0),
        .prog_ucode_we     (1'b0),
        .prog_ucode_core   ({CORE_ID_BITS{1'b0}}),
        .prog_ucode_addr   (6'd0),
        .prog_ucode_data   (32'd0),
        .prog_param_we     (prog_param_we),
        .prog_param_core   (prog_param_core),
        .prog_param_neuron (prog_param_neuron),
        .prog_param_id     (prog_param_id),
        .prog_param_value  (prog_param_value),
        .ext_valid         (ext_valid),
        .ext_core          (ext_core),
        .ext_neuron_id     (ext_neuron_id),
        .ext_current       (ext_current),
        .timestep_done     (timestep_done),
        .spike_valid_bus   (spike_valid_bus),
        .spike_id_bus      (spike_id_bus),
        .mesh_state_out    (mesh_state_out),
        .total_spikes      (total_spikes),
        .timestep_count    (timestep_count)
    );

    initial begin
        $dumpfile("tb_p15_traces.vcd");
        $dumpvars(0, tb_p15_traces);
    end

    task run_timestep_stim;
        input [CORE_ID_BITS-1:0]     core;
        input [NEURON_BITS-1:0]      neuron;
        input signed [DATA_WIDTH-1:0] current;
    begin
        @(posedge clk);
        ext_valid     <= 1;
        ext_core      <= core;
        ext_neuron_id <= neuron;
        ext_current   <= current;
        @(posedge clk);
        ext_valid <= 0;
        start     <= 1;
        @(posedge clk);
        start <= 0;
        wait (timestep_done);
        @(posedge clk);
    end
    endtask

    task run_empty;
    begin
        @(posedge clk);
        start <= 1;
        @(posedge clk);
        start <= 0;
        wait (timestep_done);
        @(posedge clk);
    end
    endtask

    task set_tau;
        input [CORE_ID_BITS-1:0]    core;
        input [NEURON_BITS-1:0]     neuron;
        input [2:0]                 param_id;
        input [3:0]                 tau_val;
    begin
        @(posedge clk);
        prog_param_we     <= 1;
        prog_param_core   <= core;
        prog_param_neuron <= neuron;
        prog_param_id     <= param_id;
        prog_param_value  <= {12'd0, tau_val};
        @(posedge clk);
        prog_param_we <= 0;
    end
    endtask

    task prog_pool_entry;
        input [CORE_ID_BITS-1:0]    core;
        input [POOL_ADDR_BITS-1:0]  addr;
        input [NEURON_BITS-1:0]     src;
        input [NEURON_BITS-1:0]     target;
        input signed [DATA_WIDTH-1:0] weight;
    begin
        @(posedge clk);
        prog_pool_we     <= 1;
        prog_pool_core   <= core;
        prog_pool_addr   <= addr;
        prog_pool_src    <= src;
        prog_pool_target <= target;
        prog_pool_weight <= weight;
        prog_pool_comp   <= 2'd0;
        @(posedge clk);
        prog_pool_we <= 0;
    end
    endtask

    task prog_index_entry;
        input [CORE_ID_BITS-1:0]    core;
        input [NEURON_BITS-1:0]     neuron;
        input [POOL_ADDR_BITS-1:0]  base;
        input [COUNT_BITS-1:0]      count;
    begin
        @(posedge clk);
        prog_index_we     <= 1;
        prog_index_core   <= core;
        prog_index_neuron <= neuron;
        prog_index_base   <= base;
        prog_index_count  <= count;
        @(posedge clk);
        prog_index_we <= 0;
    end
    endtask

    integer pass_count;
    integer fail_count;
    reg [7:0] trace1_val, trace2_val;
    reg [7:0] trace1_prev, trace2_prev;
    reg [7:0] expected_trace;
    integer ts;

    initial begin
        start = 0;
        prog_pool_we = 0; prog_pool_core = 0; prog_pool_addr = 0;
        prog_pool_src = 0; prog_pool_target = 0; prog_pool_weight = 0; prog_pool_comp = 0;
        prog_index_we = 0; prog_index_core = 0; prog_index_neuron = 0;
        prog_index_base = 0; prog_index_count = 0;
        prog_route_we = 0; prog_route_src_core = 0; prog_route_src_neuron = 0;
        prog_route_slot = 0;
        prog_route_dest_core = 0; prog_route_dest_neuron = 0; prog_route_weight = 0;
        learn_enable = 0; graded_enable = 0; dendritic_enable = 0; async_enable = 0;
        threefactor_enable = 0; reward_value = 0; noise_enable = 0;
        prog_param_we = 0; prog_param_core = 0; prog_param_neuron = 0;
        prog_param_id = 0; prog_param_value = 0;
        ext_valid = 0; ext_core = 0; ext_neuron_id = 0; ext_current = 0;

        pass_count = 0;
        fail_count = 0;

        rst_n = 0;
        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 5);

        $display("\ntest 1: Default tau exponential decay");

        run_timestep_stim(0, 0, 16'sd1003);

        trace1_val = dut.gen_core[0].core.trace_mem.mem[0];
        trace2_val = dut.gen_core[0].core.trace2_mem.mem[0];
        $display("  After spike: trace1=%0d, trace2=%0d", trace1_val, trace2_val);

        if (trace1_val == 100 && trace2_val == 100) begin
            $display("  PASS: Both traces set to TRACE_MAX (100)");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected trace1=100, trace2=100, got trace1=%0d, trace2=%0d",
                     trace1_val, trace2_val);
            fail_count = fail_count + 1;
        end

        for (ts = 0; ts < 5; ts = ts + 1) begin
            run_empty;
        end

        trace1_val = dut.gen_core[0].core.trace_mem.mem[0];
        trace2_val = dut.gen_core[0].core.trace2_mem.mem[0];
        $display("  After 5 decay steps: trace1=%0d, trace2=%0d", trace1_val, trace2_val);

        if (trace1_val < trace2_val && trace1_val > 0 && trace2_val > 0) begin
            $display("  PASS: trace1 (tau=3) decayed faster than trace2 (tau=4)");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected trace1 < trace2 (both > 0)");
            fail_count = fail_count + 1;
        end

        $display("\ntest 2: Custom tau values");

        set_tau(0, 1, 3'd6, 4'd2);
        set_tau(0, 1, 3'd7, 4'd6);
        #(CLK_PERIOD * 2);

        for (ts = 0; ts < 5; ts = ts + 1)
            run_empty;

        run_timestep_stim(0, 1, 16'sd1003);

        trace1_val = dut.gen_core[0].core.trace_mem.mem[1];
        trace2_val = dut.gen_core[0].core.trace2_mem.mem[1];
        $display("  After spike N1: trace1=%0d, trace2=%0d", trace1_val, trace2_val);

        for (ts = 0; ts < 10; ts = ts + 1) begin
            run_empty;
        end

        trace1_val = dut.gen_core[0].core.trace_mem.mem[1];
        trace2_val = dut.gen_core[0].core.trace2_mem.mem[1];
        $display("  After 10 decay steps: trace1=%0d (tau=2), trace2=%0d (tau=6)", trace1_val, trace2_val);

        if (trace1_val < trace2_val) begin
            $display("  PASS: Fast tau=2 decayed more than slow tau=6");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected trace1 (tau=2) < trace2 (tau=6)");
            fail_count = fail_count + 1;
        end

        $display("\ntest 3: Min-step-1 convergence to zero");

        set_tau(0, 2, 3'd6, 4'd8);
        #(CLK_PERIOD * 2);

        for (ts = 0; ts < 5; ts = ts + 1)
            run_empty;

        run_timestep_stim(0, 2, 16'sd1003);

        trace1_val = dut.gen_core[0].core.trace_mem.mem[2];
        $display("  After spike N2: trace1=%0d", trace1_val);

        for (ts = 0; ts < 120; ts = ts + 1)
            run_empty;

        trace1_val = dut.gen_core[0].core.trace_mem.mem[2];
        $display("  After 120 decay steps (tau=8): trace1=%0d", trace1_val);

        if (trace1_val == 0) begin
            $display("  PASS: Trace decayed to zero via min-step-1");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Trace should be 0, got %0d", trace1_val);
            fail_count = fail_count + 1;
        end

        $display("\ntest 4: STDP learning uses trace1");

        prog_pool_entry(0, 100, 10, 11, 16'sd1200);
        prog_index_entry(0, 10, 100, 1);
        #(CLK_PERIOD * 2);

        learn_enable = 1;

        for (ts = 0; ts < 5; ts = ts + 1)
            run_empty;

        run_timestep_stim(0, 11, 16'sd1003);

        trace1_val = dut.gen_core[0].core.trace_mem.mem[11];
        trace2_val = dut.gen_core[0].core.trace2_mem.mem[11];
        $display("  N11 post-spike: trace1=%0d, trace2=%0d", trace1_val, trace2_val);

        for (ts = 0; ts < 4; ts = ts + 1)
            run_empty;

        $display("  Weight[100] before LTD: %0d",
                 $signed(dut.gen_core[0].core.pool_weight_mem.mem[100]));

        run_timestep_stim(0, 10, 16'sd1003);

        $display("  Weight[100] after LTD:  %0d",
                 $signed(dut.gen_core[0].core.pool_weight_mem.mem[100]));

        if ($signed(dut.gen_core[0].core.pool_weight_mem.mem[100]) < 16'sd1200) begin
            $display("  PASS: LTD decreased weight using trace1");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Weight should have decreased from 1200");
            fail_count = fail_count + 1;
        end

        learn_enable = 0;

        $display("\ntest 5: Independent trace values");

        set_tau(0, 20, 3'd6, 4'd3);
        set_tau(0, 20, 3'd7, 4'd1);
        #(CLK_PERIOD * 2);

        for (ts = 0; ts < 5; ts = ts + 1)
            run_empty;

        run_timestep_stim(0, 20, 16'sd1003);

        trace1_val = dut.gen_core[0].core.trace_mem.mem[20];
        trace2_val = dut.gen_core[0].core.trace2_mem.mem[20];
        $display("  After spike N20: trace1=%0d, trace2=%0d", trace1_val, trace2_val);

        for (ts = 0; ts < 3; ts = ts + 1)
            run_empty;

        trace1_val = dut.gen_core[0].core.trace_mem.mem[20];
        trace2_val = dut.gen_core[0].core.trace2_mem.mem[20];
        $display("  After 3 steps: trace1=%0d (tau=3), trace2=%0d (tau=1)", trace1_val, trace2_val);

        if (trace2_val < trace1_val && trace1_val > 40 && trace2_val < 20) begin
            $display("  PASS: Traces decayed independently at different rates");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Unexpected trace values (expected trace1>40, trace2<20)");
            fail_count = fail_count + 1;
        end

        $display("  P15 Trace Tests: %0d PASSED, %0d FAILED", pass_count, fail_count);

        if (fail_count > 0)
            $display("*** SOME TESTS FAILED ***");
        else
            $display("*** ALL TESTS PASSED ***");

        #(CLK_PERIOD * 10);
        $finish;
    end

    initial begin
        #(CLK_PERIOD * 3000000);
        $display("ERROR: Simulation timed out!");
        $finish;
    end

endmodule
