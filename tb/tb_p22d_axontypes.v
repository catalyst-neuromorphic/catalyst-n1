`timescale 1ns/1ps

module tb_p22d_axontypes;

    parameter NUM_CORES      = 2;
    parameter CORE_ID_BITS   = 1;
    parameter NUM_NEURONS    = 1024;
    parameter NEURON_BITS    = 10;
    parameter DATA_WIDTH     = 16;
    parameter POOL_DEPTH     = 1024;
    parameter POOL_ADDR_BITS = 10;
    parameter COUNT_BITS     = 10;
    parameter REV_FANIN      = 32;
    parameter REV_SLOT_BITS  = 5;
    parameter CLK_PERIOD     = 10;
    parameter ROUTE_FANOUT    = 8;
    parameter ROUTE_SLOT_BITS = 3;

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
    reg                         noise_enable;
    reg                         skip_idle_enable;
    reg  signed [DATA_WIDTH-1:0] reward_value;

    reg                         prog_param_we;
    reg  [CORE_ID_BITS-1:0]    prog_param_core;
    reg  [NEURON_BITS-1:0]     prog_param_neuron;
    reg  [4:0]                  prog_param_id;
    reg  signed [DATA_WIDTH-1:0] prog_param_value;

    reg                         ext_valid;
    reg  [CORE_ID_BITS-1:0]    ext_core;
    reg  [NEURON_BITS-1:0]     ext_neuron_id;
    reg  signed [DATA_WIDTH-1:0] ext_current;

    reg                         probe_read;
    reg  [CORE_ID_BITS-1:0]    probe_core;
    reg  [NEURON_BITS-1:0]     probe_neuron;
    reg  [3:0]                  probe_state_id;
    reg  [POOL_ADDR_BITS-1:0]  probe_pool_addr;
    wire signed [DATA_WIDTH-1:0] probe_data;
    wire                         probe_valid;

    wire                        timestep_done;
    wire [NUM_CORES-1:0]        spike_valid_bus;
    wire [NUM_CORES*NEURON_BITS-1:0] spike_id_bus;
    wire [5:0]                  mesh_state_out;
    wire [31:0]                 total_spikes;
    wire [31:0]                 timestep_count;
    wire [NUM_CORES-1:0]        core_idle_bus;

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
        .THRESHOLD      (16'sd5000),
        .LEAK_RATE      (16'sd0),
        .REFRAC_CYCLES  (0)
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
        .noise_enable      (noise_enable),
        .skip_idle_enable  (skip_idle_enable),
        .scale_u_enable    (1'b0),
        .reward_value      (reward_value),
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
        .probe_read        (probe_read),
        .probe_core        (probe_core),
        .probe_neuron      (probe_neuron),
        .probe_state_id    (probe_state_id),
        .probe_pool_addr   (probe_pool_addr),
        .probe_data        (probe_data),
        .probe_valid       (probe_valid),
        .ext_valid         (ext_valid),
        .ext_core          (ext_core),
        .ext_neuron_id     (ext_neuron_id),
        .ext_current       (ext_current),
        .timestep_done     (timestep_done),
        .spike_valid_bus   (spike_valid_bus),
        .spike_id_bus      (spike_id_bus),
        .mesh_state_out    (mesh_state_out),
        .total_spikes      (total_spikes),
        .timestep_count    (timestep_count),
        .core_idle_bus     (core_idle_bus),
        .link_tx_push      (),
        .link_tx_core      (),
        .link_tx_neuron    (),
        .link_tx_payload   (),
        .link_tx_full      (1'b0),
        .link_rx_core      ({CORE_ID_BITS{1'b0}}),
        .link_rx_neuron    ({NEURON_BITS{1'b0}}),
        .link_rx_current   ({DATA_WIDTH{1'b0}}),
        .link_rx_pop       (),
        .link_rx_empty     (1'b1)
    );

    task set_param;
        input [CORE_ID_BITS-1:0]     core;
        input [NEURON_BITS-1:0]      neuron;
        input [4:0]                   pid;
        input signed [DATA_WIDTH-1:0] value;
    begin
        @(posedge clk);
        prog_param_we     <= 1;
        prog_param_core   <= core;
        prog_param_neuron <= neuron;
        prog_param_id     <= pid;
        prog_param_value  <= value;
        @(posedge clk);
        prog_param_we <= 0;
    end
    endtask

    task add_pool;
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
        prog_pool_we <= 0;
    end
    endtask

    task set_index;
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
        prog_index_we <= 0;
    end
    endtask

    task run_timestep;
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

    task do_probe;
        input [CORE_ID_BITS-1:0]     core;
        input [NEURON_BITS-1:0]      neuron;
        input [3:0]                   sid;
        input [POOL_ADDR_BITS-1:0]   paddr;
    begin
        probe_read      <= 1;
        probe_core      <= core;
        probe_neuron    <= neuron;
        probe_state_id  <= sid;
        probe_pool_addr <= paddr;
        @(posedge clk);
        probe_read <= 0;
        wait(probe_valid);
        @(posedge clk);
    end
    endtask

    task reset_all;
    begin
        rst_n <= 0;
        start <= 0;
        prog_pool_we <= 0; prog_index_we <= 0; prog_route_we <= 0;
        prog_param_we <= 0; ext_valid <= 0;
        repeat (5) @(posedge clk);
        rst_n <= 1;
        repeat (2) @(posedge clk);
        repeat (4) begin
            @(posedge clk); start <= 1;
            @(posedge clk); start <= 0;
            wait (timestep_done);
            @(posedge clk);
        end
    end
    endtask

    integer pass_count, fail_count;
    reg signed [15:0] probed_v;

    initial begin
        clk = 0; rst_n = 0;
        start = 0;
        prog_pool_we = 0; prog_pool_core = 0; prog_pool_addr = 0;
        prog_pool_src = 0; prog_pool_target = 0; prog_pool_weight = 0; prog_pool_comp = 0;
        prog_index_we = 0; prog_index_core = 0; prog_index_neuron = 0;
        prog_index_base = 0; prog_index_count = 0;
        prog_route_we = 0; prog_route_src_core = 0; prog_route_src_neuron = 0;
        prog_route_slot = 0;
        prog_route_dest_core = 0; prog_route_dest_neuron = 0; prog_route_weight = 0;
        learn_enable = 0; graded_enable = 0; dendritic_enable = 0;
        async_enable = 0; threefactor_enable = 0; noise_enable = 0;
        skip_idle_enable = 0; reward_value = 0;
        prog_param_we = 0; prog_param_core = 0; prog_param_neuron = 0;
        prog_param_id = 0; prog_param_value = 0;
        ext_valid = 0; ext_core = 0; ext_neuron_id = 0; ext_current = 0;
        probe_read = 0; probe_core = 0; probe_neuron = 0;
        probe_state_id = 0; probe_pool_addr = 0;

        pass_count = 0; fail_count = 0;

        #100 rst_n = 1;
        @(posedge clk); @(posedge clk);

        $display("\ntest 1: Two Axon Types (passthrough vs 4-bit+exp)");

        set_param(0, 10'd0, 5'd0, 16'sd100);

        set_param(0, 10'd1, 5'd26, 16'h0420);

        set_param(0, 10'd11, 5'd25, 16'd1);

        add_pool(0, 10'd0, 10'd0, 10'd10, 16'sd13);
        add_pool(0, 10'd1, 10'd0, 10'd11, 16'sd13);
        set_index(0, 10'd0, 10'd0, 10'd2);

        run_timestep(0, 10'd0, 16'sd200);

        run_empty;

        do_probe(0, 10'd10, 4'd0, 0);
        probed_v = $signed(probe_data);
        $display("  Neuron 10 (type 0, passthrough): v = %0d (expected 13)", probed_v);

        do_probe(0, 10'd11, 4'd0, 0);
        begin : test1_check
            reg signed [15:0] v10, v11;
            v10 = probed_v;
        end

        do_probe(0, 10'd10, 4'd0, 0);
        begin : test1_eval
            reg signed [15:0] v10, v11;
            v10 = $signed(probe_data);
            do_probe(0, 10'd11, 4'd0, 0);
            v11 = $signed(probe_data);
            $display("  Neuron 10 (passthrough): v = %0d", v10);
            $display("  Neuron 11 (4-bit exp=2): v = %0d", v11);
            if (v11 > v10 && v11 != v10) begin
                $display("TEST 1 PASSED (type 1 delivers more: v11=%0d > v10=%0d)", v11, v10);
                pass_count = pass_count + 1;
            end else begin
                $display("TEST 1 FAILED (expected v11 > v10, got v11=%0d, v10=%0d)", v11, v10);
                fail_count = fail_count + 1;
            end
        end

        $display("\ntest 2: Weight Decompression (4-bit, exp=3)");
        reset_all;

        set_param(0, 10'd50, 5'd0, 16'sd100);

        set_param(0, 10'd2, 5'd26, 16'h0430);

        set_param(0, 10'd60, 5'd25, 16'd2);

        add_pool(0, 10'd0, 10'd50, 10'd60, 16'sd7);
        set_index(0, 10'd50, 10'd0, 10'd1);

        run_timestep(0, 10'd50, 16'sd200);

        run_empty;
        run_empty;

        do_probe(0, 10'd60, 4'd0, 0);
        probed_v = $signed(probe_data);
        $display("  Neuron 60 v = %0d (expected 56 = 7 << 3)", probed_v);

        do_probe(0, 10'd60, 4'd13, 0);
        $display("  Neuron 60 u = %0d (expected 56)", $signed(probe_data));

        if (probed_v >= 50 && probed_v <= 62) begin
            $display("TEST 2 PASSED (decompressed weight = %0d, expected ~56)", probed_v);
            pass_count = pass_count + 1;
        end else begin
            do_probe(0, 10'd60, 4'd13, 0);
            if ($signed(probe_data) >= 50 && $signed(probe_data) <= 62) begin
                $display("TEST 2 PASSED (u = %0d, expected ~56)", $signed(probe_data));
                pass_count = pass_count + 1;
            end else begin
                $display("TEST 2 FAILED (v=%0d, u=%0d, expected ~56)", probed_v, $signed(probe_data));
                fail_count = fail_count + 1;
            end
        end

        $display("\ntest 3: Excitatory/Inhibitory Flag");
        reset_all;

        set_param(0, 10'd70, 5'd0, 16'sd100);

        set_param(0, 10'd3, 5'd26, 16'h0804);

        set_param(0, 10'd80, 5'd25, 16'd3);

        add_pool(0, 10'd0, 10'd70, 10'd80, 16'sd100);
        add_pool(0, 10'd1, 10'd70, 10'd81, 16'sd100);
        set_index(0, 10'd70, 10'd0, 10'd2);

        run_timestep(0, 10'd70, 16'sd200);

        run_empty;

        do_probe(0, 10'd80, 4'd0, 0);
        begin : test3_eval
            reg signed [15:0] v80, v81;
            v80 = $signed(probe_data);
            do_probe(0, 10'd81, 4'd0, 0);
            v81 = $signed(probe_data);
            $display("  Neuron 80 (isExc): v = %0d (expected 0, clamped from -100)", v80);
            $display("  Neuron 81 (passthrough): v = %0d (expected 100)", v81);
            if (v80 <= 0 && v81 > 0 && v81 != v80) begin
                $display("TEST 3 PASSED (isExc: v80=%0d <= 0, passthrough: v81=%0d > 0)", v80, v81);
                pass_count = pass_count + 1;
            end else begin
                $display("TEST 3 FAILED (v80=%0d, v81=%0d)", v80, v81);
                fail_count = fail_count + 1;
            end
        end

        $display("\ntest 4: backward compat (passthrough)");
        reset_all;

        set_param(0, 10'd90, 5'd0, 16'sd100);

        add_pool(0, 10'd0, 10'd90, 10'd100, 16'sd500);
        set_index(0, 10'd90, 10'd0, 10'd1);

        run_timestep(0, 10'd90, 16'sd200);

        run_empty;
        run_empty;

        do_probe(0, 10'd100, 4'd0, 0);
        probed_v = $signed(probe_data);
        $display("  Neuron 100 (default passthrough): v = %0d (expected ~500)", probed_v);
        do_probe(0, 10'd100, 4'd13, 0);
        $display("  Neuron 100 u = %0d (expected 500)", $signed(probe_data));

        if (probed_v >= 490 && probed_v <= 510) begin
            $display("TEST 4 PASSED (passthrough weight delivery: v=%0d)", probed_v);
            pass_count = pass_count + 1;
        end else begin
            do_probe(0, 10'd100, 4'd13, 0);
            if ($signed(probe_data) >= 490 && $signed(probe_data) <= 510) begin
                $display("TEST 4 PASSED (u=%0d matches expected 500)", $signed(probe_data));
                pass_count = pass_count + 1;
            end else begin
                $display("TEST 4 FAILED (v=%0d, u=%0d, expected ~500)", probed_v, $signed(probe_data));
                fail_count = fail_count + 1;
            end
        end

        $display("\nP22D RESULTS: %0d/4 passed", pass_count);
        if (fail_count == 0)
            $display("All tests passed!");
        else
            $display("%0d tests FAILED", fail_count);
        $finish;
    end

    initial begin
        #5000000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
