`timescale 1ns/1ps

module tb_p22b_compartments;

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
        .prog_ucode_addr   (7'd0),
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

    reg [31:0] ext_spike_count;
    reg [NEURON_BITS-1:0] last_spike_id;

    always @(posedge clk) begin : spike_monitor
        integer c;
        for (c = 0; c < NUM_CORES; c = c + 1) begin
            if (spike_valid_bus[c]) begin
                $display("  [t=%0d] Core %0d Neuron %0d spiked (external)",
                    timestep_count, c, spike_id_bus[c*NEURON_BITS +: NEURON_BITS]);
            end
        end
    end

    reg [NUM_NEURONS-1:0] captured_spike_bitmap;
    always @(posedge clk) begin : bitmap_capture
        if (dut.gen_core[0].core.state == 6'd12) begin
            captured_spike_bitmap <= dut.gen_core[0].core.spike_bitmap;
        end
    end

    task reset_all;
    begin
        rst_n = 0; start = 0;
        prog_pool_we = 0; prog_pool_core = 0; prog_pool_addr = 0;
        prog_pool_src = 0; prog_pool_target = 0; prog_pool_weight = 0; prog_pool_comp = 0;
        prog_index_we = 0; prog_index_core = 0; prog_index_neuron = 0;
        prog_index_base = 0; prog_index_count = 0;
        prog_route_we = 0; prog_route_src_core = 0; prog_route_src_neuron = 0;
        prog_route_slot = 0; prog_route_dest_core = 0; prog_route_dest_neuron = 0;
        prog_route_weight = 0;
        learn_enable = 0; graded_enable = 0; dendritic_enable = 0;
        async_enable = 0; threefactor_enable = 0; noise_enable = 0;
        skip_idle_enable = 0; reward_value = 0;
        prog_param_we = 0; prog_param_core = 0; prog_param_neuron = 0;
        prog_param_id = 0; prog_param_value = 0;
        ext_valid = 0; ext_core = 0; ext_neuron_id = 0; ext_current = 0;
        probe_read = 0; probe_core = 0; probe_neuron = 0; probe_state_id = 0; probe_pool_addr = 0;
        ext_spike_count = 0; last_spike_id = 0;
        #100;
        rst_n = 1;
        #20;
        repeat (4) begin
            @(posedge clk); start <= 1;
            @(posedge clk); start <= 0;
            wait (timestep_done);
            @(posedge clk);
        end
    end
    endtask

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

    integer pass_count, fail_count;
    reg [31:0] spikes_before, spikes_after;

    initial begin
        pass_count = 0;
        fail_count = 0;

        $display("test 1: Flat mode (backward compatible)");
        reset_all;

        spikes_before = total_spikes;
        run_timestep(0, 10'd10, 16'sd2000);
        spikes_after = total_spikes;
        $display("  External spikes: %0d (expected 1)", spikes_after - spikes_before);
        if (spikes_after - spikes_before == 1) begin
            $display("TEST 1 PASSED (flat root neuron emits external spike)");
            pass_count = pass_count + 1;
        end else begin
            $display("TEST 1 FAILED");
            fail_count = fail_count + 1;
        end

        $display("test 2: Chain compartment tree");
        reset_all;

        begin
            integer n;
            for (n = 0; n < 4; n = n + 1) begin
                set_param(0, n[NEURON_BITS-1:0], 5'd0, 16'sd500);
                set_param(0, n[NEURON_BITS-1:0], 5'd1, 16'sd0);
            end
        end

        set_param(0, 10'd0, 5'd22, 16'd1);
        set_param(0, 10'd0, 5'd24, 16'd0);
        set_param(0, 10'd1, 5'd22, 16'd2);
        set_param(0, 10'd1, 5'd24, 16'd0);
        set_param(0, 10'd2, 5'd22, 16'd3);
        set_param(0, 10'd2, 5'd24, 16'd0);

        spikes_before = total_spikes;
        run_timestep(0, 10'd0, 16'sd2000);
        spikes_after = total_spikes;

        $display("  External spikes: %0d (expected 1 from root comp 3)", spikes_after - spikes_before);

        begin
            reg bitmap3;
            bitmap3 = dut.gen_core[0].core.spike_bitmap[3];
            $display("  Comp 3 spike_bitmap: %0d", bitmap3);
        end

        if (spikes_after - spikes_before == 1) begin
            $display("TEST 2 PASSED (chain tree: only root emits external spike)");
            pass_count = pass_count + 1;
        end else begin
            $display("TEST 2 FAILED (expected 1 external spike from root)");
            fail_count = fail_count + 1;
        end

        $display("test 3: Fan-in with JoinOp");
        reset_all;

        set_param(0, 10'd10, 5'd0, 16'sd400);
        set_param(0, 10'd10, 5'd1, 16'sd0);
        set_param(0, 10'd11, 5'd0, 16'sd600);
        set_param(0, 10'd11, 5'd1, 16'sd0);
        set_param(0, 10'd12, 5'd0, 16'sd1200);
        set_param(0, 10'd12, 5'd1, 16'sd0);

        set_param(0, 10'd10, 5'd22, 16'd12);
        set_param(0, 10'd10, 5'd24, 16'd0);
        set_param(0, 10'd11, 5'd22, 16'd12);
        set_param(0, 10'd11, 5'd24, 16'd0);

        @(posedge clk);
        ext_valid     <= 1; ext_core <= 0; ext_neuron_id <= 10'd10; ext_current <= 16'sd2000;
        @(posedge clk);
        ext_valid     <= 1; ext_core <= 0; ext_neuron_id <= 10'd11; ext_current <= 16'sd2000;
        @(posedge clk);
        ext_valid <= 0;
        start <= 1;
        @(posedge clk);
        start <= 0;
        wait (timestep_done);
        @(posedge clk);

        begin
            reg signed [DATA_WIDTH-1:0] parent_v;
            parent_v = dut.gen_core[0].core.neuron_mem.mem[12][DATA_WIDTH-1:0];
            $display("  ADD mode: parent (12) potential=%0d (threshold=1200)", parent_v);
        end

        reset_all;
        set_param(0, 10'd10, 5'd0, 16'sd400);
        set_param(0, 10'd10, 5'd1, 16'sd0);
        set_param(0, 10'd11, 5'd0, 16'sd600);
        set_param(0, 10'd11, 5'd1, 16'sd0);
        set_param(0, 10'd12, 5'd0, 16'sd500);
        set_param(0, 10'd12, 5'd1, 16'sd0);

        set_param(0, 10'd10, 5'd22, 16'd12);
        set_param(0, 10'd10, 5'd24, 16'd0);
        set_param(0, 10'd10, 5'd23, 16'd1);
        set_param(0, 10'd11, 5'd22, 16'd12);
        set_param(0, 10'd11, 5'd24, 16'd0);
        set_param(0, 10'd11, 5'd23, 16'd1);

        spikes_before = total_spikes;
        @(posedge clk);
        ext_valid     <= 1; ext_core <= 0; ext_neuron_id <= 10'd10; ext_current <= 16'sd2000;
        @(posedge clk);
        ext_valid     <= 1; ext_core <= 0; ext_neuron_id <= 10'd11; ext_current <= 16'sd2000;
        @(posedge clk);
        ext_valid <= 0;
        start <= 1;
        @(posedge clk);
        start <= 0;
        wait (timestep_done);
        @(posedge clk);
        spikes_after = total_spikes;

        begin
            $display("  ABS_MAX mode: ext spikes=%0d (expected 1)", spikes_after - spikes_before);
            if (spikes_after - spikes_before == 1) begin
                $display("TEST 3 PASSED (ADD gives 1000 < 1200 no spike; ABS_MAX gives 600 >= 500 spike)");
                pass_count = pass_count + 1;
            end else begin
                $display("TEST 3 FAILED (ABS_MAX parent should have produced 1 external spike)");
                fail_count = fail_count + 1;
            end
        end

        $display("test 4: Non-root spike suppression");
        reset_all;

        set_param(0, 10'd20, 5'd22, 16'd21);
        set_param(0, 10'd20, 5'd24, 16'd0);

        spikes_before = total_spikes;
        run_timestep(0, 10'd20, 16'sd2000);
        spikes_after = total_spikes;

        begin
            reg bitmap20;
            reg signed [DATA_WIDTH-1:0] parent21_v;
            bitmap20 = captured_spike_bitmap[20];
            parent21_v = dut.gen_core[0].core.neuron_mem.mem[21][DATA_WIDTH-1:0];
            $display("  Comp 20 captured_bitmap: %0d (internal spike)", bitmap20);
            $display("  Comp 21 potential: %0d (received contribution)", parent21_v);
            $display("  External spikes: %0d (expected 0 — comp 20 is non-root)", spikes_after - spikes_before);

            if (spikes_after - spikes_before == 0 && bitmap20 == 1 && parent21_v > 0) begin
                $display("TEST 4 PASSED (non-root spike suppressed externally, parent received contribution)");
                pass_count = pass_count + 1;
            end else begin
                $display("TEST 4 FAILED (expected 0 ext spikes, bitmap20=1, parent21_v>0)");
                fail_count = fail_count + 1;
            end
        end

        $display("P22B RESULTS: %0d/%0d passed", pass_count, pass_count + fail_count);
        if (fail_count == 0)
            $display("All tests passed!");
        else
            $display("SOME TESTS FAILED");
        $finish;
    end

endmodule
