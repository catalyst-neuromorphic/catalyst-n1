`timescale 1ns / 1ps

module tb_p18_formats;

    parameter NUM_CORES      = 4;
    parameter CORE_ID_BITS   = 2;
    parameter NUM_NEURONS    = 1024;
    parameter NEURON_BITS    = 10;
    parameter DATA_WIDTH     = 16;
    parameter POOL_DEPTH     = 1024;
    parameter POOL_ADDR_BITS = 10;
    parameter COUNT_BITS     = 10;
    parameter CLK_PERIOD     = 10;

    localparam FMT_SPARSE = 2'd0;
    localparam FMT_DENSE  = 2'd1;
    localparam FMT_POP    = 2'd2;

    reg                          clk, rst_n;
    reg                          start;

    reg                          prog_pool_we;
    reg  [CORE_ID_BITS-1:0]     prog_pool_core;
    reg  [POOL_ADDR_BITS-1:0]   prog_pool_addr;
    reg  [NEURON_BITS-1:0]      prog_pool_src;
    reg  [NEURON_BITS-1:0]      prog_pool_target;
    reg  signed [DATA_WIDTH-1:0] prog_pool_weight;
    reg  [1:0]                   prog_pool_comp;

    reg                          prog_index_we;
    reg  [CORE_ID_BITS-1:0]     prog_index_core;
    reg  [NEURON_BITS-1:0]      prog_index_neuron;
    reg  [POOL_ADDR_BITS-1:0]   prog_index_base;
    reg  [COUNT_BITS-1:0]       prog_index_count;
    reg  [1:0]                   prog_index_format;

    reg                          ext_valid;
    reg  [CORE_ID_BITS-1:0]     ext_core;
    reg  [NEURON_BITS-1:0]      ext_neuron_id;
    reg  signed [DATA_WIDTH-1:0] ext_current;

    wire                         timestep_done;
    wire [NUM_CORES-1:0]         spike_valid_bus;
    wire [NUM_CORES*NEURON_BITS-1:0] spike_id_bus;
    wire [4:0]                   mesh_state_out;
    wire [31:0]                  total_spikes;
    wire [31:0]                  timestep_count;

    integer spike_count_arr [0:NUM_NEURONS-1];
    integer i;

    neuromorphic_mesh #(
        .NUM_CORES      (NUM_CORES),
        .CORE_ID_BITS   (CORE_ID_BITS),
        .NUM_NEURONS    (NUM_NEURONS),
        .NEURON_BITS    (NEURON_BITS),
        .DATA_WIDTH     (DATA_WIDTH),
        .POOL_DEPTH     (POOL_DEPTH),
        .POOL_ADDR_BITS (POOL_ADDR_BITS),
        .COUNT_BITS     (COUNT_BITS),
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
        .prog_index_format (prog_index_format),
        .prog_route_we         (1'b0),
        .prog_route_src_core   ({CORE_ID_BITS{1'b0}}),
        .prog_route_src_neuron ({NEURON_BITS{1'b0}}),
        .prog_route_slot       (3'd0),
        .prog_route_dest_core  ({CORE_ID_BITS{1'b0}}),
        .prog_route_dest_neuron({NEURON_BITS{1'b0}}),
        .prog_route_weight     (16'sd0),
        .prog_global_route_we(1'b0),
        .prog_global_route_src_core({CORE_ID_BITS{1'b0}}),
        .prog_global_route_src_neuron({NEURON_BITS{1'b0}}),
        .prog_global_route_slot(2'b0),
        .prog_global_route_dest_core({CORE_ID_BITS{1'b0}}),
        .prog_global_route_dest_neuron({NEURON_BITS{1'b0}}),
        .prog_global_route_weight({DATA_WIDTH{1'b0}}),
        .learn_enable      (1'b0),
        .graded_enable     (1'b0),
        .dendritic_enable  (1'b0),
        .async_enable      (1'b0),
        .threefactor_enable(1'b0),
        .noise_enable      (1'b0),
        .reward_value      (16'sd0),
        .prog_delay_we     (1'b0),
        .prog_delay_core   ({CORE_ID_BITS{1'b0}}),
        .prog_delay_addr   ({POOL_ADDR_BITS{1'b0}}),
        .prog_delay_value  (6'd0),
        .prog_ucode_we     (1'b0),
        .prog_ucode_core   ({CORE_ID_BITS{1'b0}}),
        .prog_ucode_addr   (6'd0),
        .prog_ucode_data   (32'd0),
        .prog_param_we     (1'b0),
        .prog_param_core   ({CORE_ID_BITS{1'b0}}),
        .prog_param_neuron ({NEURON_BITS{1'b0}}),
        .prog_param_id     (3'd0),
        .prog_param_value  (16'sd0),
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

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    always @(posedge clk) begin
        if (spike_valid_bus[0]) begin
            spike_count_arr[spike_id_bus[NEURON_BITS-1:0]] =
                spike_count_arr[spike_id_bus[NEURON_BITS-1:0]] + 1;
            $display("  [t=%0d] Core 0 N%0d spiked",
                timestep_count, spike_id_bus[NEURON_BITS-1:0]);
        end
    end

    initial begin
        $dumpfile("p18_formats.vcd");
        $dumpvars(0, tb_p18_formats);
    end

    task prog_pool_entry;
        input [CORE_ID_BITS-1:0]     core;
        input [POOL_ADDR_BITS-1:0]   addr;
        input [NEURON_BITS-1:0]      src;
        input [NEURON_BITS-1:0]      target;
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
        prog_pool_we     <= 0;
    end
    endtask

    task prog_idx_entry;
        input [CORE_ID_BITS-1:0]     core;
        input [NEURON_BITS-1:0]      neuron;
        input [POOL_ADDR_BITS-1:0]   base;
        input [COUNT_BITS-1:0]       count;
        input [1:0]                  fmt;
    begin
        @(posedge clk);
        prog_index_we     <= 1;
        prog_index_core   <= core;
        prog_index_neuron <= neuron;
        prog_index_base   <= base;
        prog_index_count  <= count;
        prog_index_format <= fmt;
        @(posedge clk);
        prog_index_we     <= 0;
    end
    endtask

    task run_stim;
        input [CORE_ID_BITS-1:0]     core;
        input [NEURON_BITS-1:0]      neuron;
        input signed [DATA_WIDTH-1:0] current;
    begin
        ext_valid     <= 1;
        ext_core      <= core;
        ext_neuron_id <= neuron;
        ext_current   <= current;
        @(posedge clk);
        ext_valid     <= 0;
        @(posedge clk);
        start <= 1;
        @(posedge clk);
        start <= 0;
        wait(timestep_done);
        @(posedge clk);
    end
    endtask

    task run_empty;
    begin
        @(posedge clk);
        start <= 1;
        @(posedge clk);
        start <= 0;
        wait(timestep_done);
        @(posedge clk);
    end
    endtask

    task reset_tracking;
    begin
        for (i = 0; i < NUM_NEURONS; i = i + 1)
            spike_count_arr[i] = 0;
    end
    endtask

    integer t, tests_passed, tests_total;
    initial begin
        tests_passed = 0;
        tests_total  = 0;

        for (i = 0; i < NUM_NEURONS; i = i + 1)
            spike_count_arr[i] = 0;

        rst_n = 0; start = 0;
        ext_valid = 0; ext_core = 0; ext_neuron_id = 0; ext_current = 0;
        prog_pool_we = 0; prog_pool_core = 0; prog_pool_addr = 0;
        prog_pool_src = 0; prog_pool_target = 0; prog_pool_weight = 0; prog_pool_comp = 0;
        prog_index_we = 0; prog_index_core = 0; prog_index_neuron = 0;
        prog_index_base = 0; prog_index_count = 0; prog_index_format = 0;

        $display("  Phase 18: Synapse Format Tests");

        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 5);

        $display("test 1: Sparse backward compat");
        tests_total = tests_total + 1;

        prog_pool_entry(0, 0, 10'd0, 10'd1, 16'sd1200);
        prog_idx_entry(0, 10'd0, 10'd0, 10'd1, FMT_SPARSE);

        reset_tracking();
        for (t = 0; t < 20; t = t + 1)
            run_stim(0, 10'd0, 16'sd200);

        $display("  N0 spikes: %0d, N1 spikes: %0d", spike_count_arr[0], spike_count_arr[1]);
        if (spike_count_arr[0] > 0 && spike_count_arr[1] > 0) begin
            $display("TEST 1 PASSED");
            tests_passed = tests_passed + 1;
        end else
            $display("TEST 1 FAILED");

        $display("test 2: Dense format (5 implicit targets)");
        tests_total = tests_total + 1;

        prog_pool_entry(0, 100, 10'd100, 10'd101, 16'sd1200);
        prog_pool_entry(0, 101, 10'd100, 10'd0,   16'sd1200);
        prog_pool_entry(0, 102, 10'd100, 10'd0,   16'sd1200);
        prog_pool_entry(0, 103, 10'd100, 10'd0,   16'sd1200);
        prog_pool_entry(0, 104, 10'd100, 10'd0,   16'sd1200);

        prog_idx_entry(0, 10'd100, 10'd100, 10'd5, FMT_DENSE);

        reset_tracking();
        for (t = 0; t < 20; t = t + 1)
            run_stim(0, 10'd100, 16'sd200);

        $display("  N100 spikes: %0d", spike_count_arr[100]);
        $display("  N101 spikes: %0d (base+0)", spike_count_arr[101]);
        $display("  N102 spikes: %0d (base+1)", spike_count_arr[102]);
        $display("  N103 spikes: %0d (base+2)", spike_count_arr[103]);
        $display("  N104 spikes: %0d (base+3)", spike_count_arr[104]);
        $display("  N105 spikes: %0d (base+4)", spike_count_arr[105]);

        if (spike_count_arr[100] > 0 &&
            spike_count_arr[101] > 0 && spike_count_arr[102] > 0 &&
            spike_count_arr[103] > 0 && spike_count_arr[104] > 0 &&
            spike_count_arr[105] > 0) begin
            $display("TEST 2 PASSED (all 5 dense targets fired)");
            tests_passed = tests_passed + 1;
        end else
            $display("TEST 2 FAILED");

        $display("test 3: Population format (8 targets, 1 weight)");
        tests_total = tests_total + 1;

        prog_pool_entry(0, 200, 10'd200, 10'd201, 16'sd1200);

        prog_idx_entry(0, 10'd200, 10'd200, 10'd8, FMT_POP);

        reset_tracking();
        for (t = 0; t < 20; t = t + 1)
            run_stim(0, 10'd200, 16'sd200);

        $display("  N200 spikes: %0d", spike_count_arr[200]);
        begin : pop_check
            integer all_fired, pop_i;
            all_fired = 1;
            for (pop_i = 201; pop_i <= 208; pop_i = pop_i + 1) begin
                $display("  N%0d spikes: %0d", pop_i, spike_count_arr[pop_i]);
                if (spike_count_arr[pop_i] == 0) all_fired = 0;
            end
            if (spike_count_arr[200] > 0 && all_fired) begin
                $display("TEST 3 PASSED (all 8 pop targets fired with 1 pool entry)");
                tests_passed = tests_passed + 1;
            end else
                $display("TEST 3 FAILED");
        end

        $display("test 4: Mixed formats in same core");
        tests_total = tests_total + 1;

        prog_pool_entry(0, 300, 10'd300, 10'd301, 16'sd1200);
        prog_idx_entry(0, 10'd300, 10'd300, 10'd1, FMT_SPARSE);

        prog_pool_entry(0, 310, 10'd310, 10'd311, 16'sd1200);
        prog_pool_entry(0, 311, 10'd310, 10'd0,   16'sd1200);
        prog_pool_entry(0, 312, 10'd310, 10'd0,   16'sd1200);
        prog_idx_entry(0, 10'd310, 10'd310, 10'd3, FMT_DENSE);

        prog_pool_entry(0, 320, 10'd320, 10'd321, 16'sd1200);
        prog_idx_entry(0, 10'd320, 10'd320, 10'd4, FMT_POP);

        reset_tracking();
        for (t = 0; t < 20; t = t + 1) begin
            ext_valid <= 1; ext_core <= 0; ext_neuron_id <= 10'd300; ext_current <= 16'sd200;
            @(posedge clk); ext_valid <= 0; @(posedge clk);
            ext_valid <= 1; ext_core <= 0; ext_neuron_id <= 10'd310; ext_current <= 16'sd200;
            @(posedge clk); ext_valid <= 0; @(posedge clk);
            ext_valid <= 1; ext_core <= 0; ext_neuron_id <= 10'd320; ext_current <= 16'sd200;
            @(posedge clk); ext_valid <= 0; @(posedge clk);
            start <= 1; @(posedge clk); start <= 0;
            wait(timestep_done); @(posedge clk);
        end

        $display("  Sparse: N300→N301: src=%0d tgt=%0d", spike_count_arr[300], spike_count_arr[301]);
        $display("  Dense:  N310→N311..313: src=%0d, 311=%0d 312=%0d 313=%0d",
            spike_count_arr[310], spike_count_arr[311], spike_count_arr[312], spike_count_arr[313]);
        $display("  Pop:    N320→N321..324: src=%0d, 321=%0d 322=%0d 323=%0d 324=%0d",
            spike_count_arr[320], spike_count_arr[321], spike_count_arr[322],
            spike_count_arr[323], spike_count_arr[324]);

        if (spike_count_arr[301] > 0 &&
            spike_count_arr[311] > 0 && spike_count_arr[312] > 0 && spike_count_arr[313] > 0 &&
            spike_count_arr[321] > 0 && spike_count_arr[322] > 0 &&
            spike_count_arr[323] > 0 && spike_count_arr[324] > 0) begin
            $display("TEST 4 PASSED (all formats coexist)");
            tests_passed = tests_passed + 1;
        end else
            $display("TEST 4 FAILED");

        $display("P18 RESULTS: %0d/%0d passed", tests_passed, tests_total);
        if (tests_passed == tests_total)
            $display("All tests passed!");
        else
            $display("SOME TESTS FAILED");

        #(CLK_PERIOD * 10);
        $finish;
    end

    initial begin
        #(CLK_PERIOD * 5000000);
        $display("TIMEOUT at state=%0d, ts=%0d", mesh_state_out, timestep_count);
        $finish;
    end

endmodule
