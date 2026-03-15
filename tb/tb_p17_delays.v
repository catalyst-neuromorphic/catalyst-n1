`timescale 1ns / 1ps

module tb_p17_delays;

    parameter NUM_CORES      = 4;
    parameter CORE_ID_BITS   = 2;
    parameter NUM_NEURONS    = 1024;
    parameter NEURON_BITS    = 10;
    parameter DATA_WIDTH     = 16;
    parameter POOL_DEPTH     = 1024;
    parameter POOL_ADDR_BITS = 10;
    parameter COUNT_BITS     = 10;
    parameter CLK_PERIOD     = 10;

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

    reg                          prog_delay_we;
    reg  [CORE_ID_BITS-1:0]     prog_delay_core;
    reg  [POOL_ADDR_BITS-1:0]   prog_delay_addr;
    reg  [5:0]                   prog_delay_value;

    reg                          prog_route_we;
    reg  [CORE_ID_BITS-1:0]     prog_route_src_core;
    reg  [NEURON_BITS-1:0]      prog_route_src_neuron;
    reg  [2:0]                   prog_route_slot;
    reg  [CORE_ID_BITS-1:0]     prog_route_dest_core;
    reg  [NEURON_BITS-1:0]      prog_route_dest_neuron;
    reg  signed [DATA_WIDTH-1:0] prog_route_weight;

    reg                          prog_param_we;
    reg  [CORE_ID_BITS-1:0]     prog_param_core;
    reg  [NEURON_BITS-1:0]      prog_param_neuron;
    reg  [2:0]                   prog_param_id;
    reg  signed [DATA_WIDTH-1:0] prog_param_value;

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

    integer spike_ts [0:NUM_NEURONS-1];
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
        .learn_enable      (1'b0),
        .graded_enable     (1'b0),
        .dendritic_enable  (1'b0),
        .async_enable      (1'b0),
        .threefactor_enable(1'b0),
        .noise_enable      (1'b0),
        .reward_value      (16'sd0),
        .prog_delay_we     (prog_delay_we),
        .prog_delay_core   (prog_delay_core),
        .prog_delay_addr   (prog_delay_addr),
        .prog_delay_value  (prog_delay_value),
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

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    always @(posedge clk) begin
        if (spike_valid_bus[0]) begin
            spike_ts[spike_id_bus[NEURON_BITS-1:0]] = timestep_count;
            spike_count_arr[spike_id_bus[NEURON_BITS-1:0]] =
                spike_count_arr[spike_id_bus[NEURON_BITS-1:0]] + 1;
            $display("  [t=%0d] Core 0 Neuron %0d spiked",
                timestep_count, spike_id_bus[NEURON_BITS-1:0]);
        end
    end

    initial begin
        $dumpfile("p17_delays.vcd");
        $dumpvars(0, tb_p17_delays);
    end

    task prog_pool;
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

    task prog_idx;
        input [CORE_ID_BITS-1:0]     core;
        input [NEURON_BITS-1:0]      neuron;
        input [POOL_ADDR_BITS-1:0]   base;
        input [COUNT_BITS-1:0]       count;
    begin
        @(posedge clk);
        prog_index_we     <= 1;
        prog_index_core   <= core;
        prog_index_neuron <= neuron;
        prog_index_base   <= base;
        prog_index_count  <= count;
        @(posedge clk);
        prog_index_we     <= 0;
    end
    endtask

    task prog_dly;
        input [CORE_ID_BITS-1:0]     core;
        input [POOL_ADDR_BITS-1:0]   addr;
        input [5:0]                  delay_val;
    begin
        @(posedge clk);
        prog_delay_we    <= 1;
        prog_delay_core  <= core;
        prog_delay_addr  <= addr;
        prog_delay_value <= delay_val;
        @(posedge clk);
        prog_delay_we    <= 0;
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
        for (i = 0; i < NUM_NEURONS; i = i + 1) begin
            spike_ts[i] = -1;
            spike_count_arr[i] = 0;
        end
    end
    endtask

    integer tests_passed, tests_total;

    integer t, src_spike_ts, tgt_spike_ts;
    initial begin
        tests_passed = 0;
        tests_total  = 0;

        for (i = 0; i < NUM_NEURONS; i = i + 1) begin
            spike_ts[i] = -1;
            spike_count_arr[i] = 0;
        end
        rst_n = 0; start = 0;
        ext_valid = 0; ext_core = 0; ext_neuron_id = 0; ext_current = 0;
        prog_pool_we = 0; prog_pool_core = 0; prog_pool_addr = 0;
        prog_pool_src = 0; prog_pool_target = 0; prog_pool_weight = 0; prog_pool_comp = 0;
        prog_index_we = 0; prog_index_core = 0; prog_index_neuron = 0;
        prog_index_base = 0; prog_index_count = 0;
        prog_delay_we = 0; prog_delay_core = 0; prog_delay_addr = 0; prog_delay_value = 0;
        prog_route_we = 0; prog_route_src_core = 0; prog_route_src_neuron = 0;
        prog_route_slot = 0; prog_route_dest_core = 0; prog_route_dest_neuron = 0;
        prog_route_weight = 0;
        prog_param_we = 0; prog_param_core = 0; prog_param_neuron = 0;
        prog_param_id = 0; prog_param_value = 0;

        $display("  Phase 17: Axon Delay Tests");

        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 5);

        $display("test 1: Delay=0 backward compat");
        tests_total = tests_total + 1;

        prog_pool(0, 0, 10'd0, 10'd1, 16'sd1200);
        prog_idx(0, 10'd0, 10'd0, 10'd1);

        reset_tracking();
        for (t = 0; t < 20; t = t + 1) begin
            run_stim(0, 10'd0, 16'sd200);
        end

        $display("  N0 first spike: t=%0d", spike_ts[0]);
        $display("  N1 first spike: t=%0d", spike_ts[1]);
        $display("  N0 total spikes: %0d", spike_count_arr[0]);
        $display("  N1 total spikes: %0d", spike_count_arr[1]);

        if (spike_count_arr[0] > 0 && spike_count_arr[1] > 0) begin
            $display("TEST 1 PASSED (delay=0 delivers immediately)");
            tests_passed = tests_passed + 1;
        end else begin
            $display("TEST 1 FAILED");
        end

        $display("test 2: Delay=3");
        tests_total = tests_total + 1;

        prog_pool(0, 10, 10'd10, 10'd11, 16'sd1200);
        prog_idx(0, 10'd10, 10'd10, 10'd1);
        prog_dly(0, 10, 6'd3);

        reset_tracking();
        for (t = 0; t < 30; t = t + 1) begin
            run_stim(0, 10'd10, 16'sd200);
        end

        src_spike_ts = spike_ts[10];
        tgt_spike_ts = spike_ts[11];
        $display("  N10 first spike: t=%0d", src_spike_ts);
        $display("  N11 first spike: t=%0d", tgt_spike_ts);

        if (spike_count_arr[10] > 0 && spike_count_arr[11] > 0 &&
            tgt_spike_ts > src_spike_ts + 1) begin
            $display("TEST 2 PASSED (delay=3 causes later delivery, delta=%0d)",
                     tgt_spike_ts - src_spike_ts);
            tests_passed = tests_passed + 1;
        end else begin
            $display("TEST 2 FAILED (src_ts=%0d, tgt_ts=%0d)", src_spike_ts, tgt_spike_ts);
        end

        $display("test 3: Mixed delays (delay=1 and delay=5)");
        tests_total = tests_total + 1;

        prog_pool(0, 20, 10'd20, 10'd21, 16'sd1200);
        prog_pool(0, 21, 10'd20, 10'd22, 16'sd1200);
        prog_idx(0, 10'd20, 10'd20, 10'd2);
        prog_dly(0, 20, 6'd1);
        prog_dly(0, 21, 6'd5);

        reset_tracking();
        for (t = 0; t < 30; t = t + 1) begin
            run_stim(0, 10'd20, 16'sd200);
        end

        $display("  N20 first spike: t=%0d", spike_ts[20]);
        $display("  N21 first spike: t=%0d (delay=1)", spike_ts[21]);
        $display("  N22 first spike: t=%0d (delay=5)", spike_ts[22]);

        if (spike_count_arr[21] > 0 && spike_count_arr[22] > 0 &&
            spike_ts[21] < spike_ts[22]) begin
            $display("TEST 3 PASSED (N21 fires before N22: delta=%0d)",
                     spike_ts[22] - spike_ts[21]);
            tests_passed = tests_passed + 1;
        end else begin
            $display("TEST 3 FAILED");
        end

        $display("test 4: Delay=0 vs Delay=3 comparison");
        tests_total = tests_total + 1;

        prog_pool(0, 30, 10'd30, 10'd31, 16'sd1200);
        prog_idx(0, 10'd30, 10'd30, 10'd1);

        prog_pool(0, 40, 10'd40, 10'd41, 16'sd1200);
        prog_idx(0, 10'd40, 10'd40, 10'd1);
        prog_dly(0, 40, 6'd3);

        reset_tracking();
        for (t = 0; t < 30; t = t + 1) begin
            ext_valid     <= 1;
            ext_core      <= 0;
            ext_neuron_id <= 10'd30;
            ext_current   <= 16'sd200;
            @(posedge clk);
            ext_valid     <= 0;
            @(posedge clk);
            ext_valid     <= 1;
            ext_core      <= 0;
            ext_neuron_id <= 10'd40;
            ext_current   <= 16'sd200;
            @(posedge clk);
            ext_valid     <= 0;
            @(posedge clk);

            start <= 1;
            @(posedge clk);
            start <= 0;
            wait(timestep_done);
            @(posedge clk);
        end

        $display("  N30 first spike: t=%0d", spike_ts[30]);
        $display("  N31 first spike: t=%0d (delay=0)", spike_ts[31]);
        $display("  N40 first spike: t=%0d", spike_ts[40]);
        $display("  N41 first spike: t=%0d (delay=3)", spike_ts[41]);

        if (spike_count_arr[31] > 0 && spike_count_arr[41] > 0) begin
            if (spike_ts[41] - spike_ts[40] > spike_ts[31] - spike_ts[30]) begin
                $display("TEST 4 PASSED (delay=3 path has %0d extra timestep delay)",
                         (spike_ts[41] - spike_ts[40]) - (spike_ts[31] - spike_ts[30]));
                tests_passed = tests_passed + 1;
            end else begin
                $display("TEST 4 FAILED (no measurable delay difference)");
            end
        end else begin
            $display("TEST 4 FAILED (spikes missing: N31=%0d, N41=%0d)",
                     spike_count_arr[31], spike_count_arr[41]);
        end

        $display("P17 RESULTS: %0d/%0d passed", tests_passed, tests_total);
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
