`timescale 1ns/1ps

module tb_p22e_noc;

    parameter NUM_CORES      = 4;
    parameter CORE_ID_BITS   = 2;
    parameter NUM_NEURONS    = 1024;
    parameter NEURON_BITS    = 10;
    parameter DATA_WIDTH     = 16;
    parameter POOL_DEPTH     = 1024;
    parameter POOL_ADDR_BITS = 10;
    parameter COUNT_BITS     = 10;
    parameter REV_FANIN      = 32;
    parameter REV_SLOT_BITS  = 5;
    parameter CLK_PERIOD     = 10;
    parameter ROUTE_FANOUT     = 8;
    parameter ROUTE_SLOT_BITS  = 3;
    parameter MESH_X = 2;
    parameter MESH_Y = 2;

    reg clk, rst_n;
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    reg                         start;

    reg                         prog_route_we;
    reg  [CORE_ID_BITS-1:0]    prog_route_src_core;
    reg  [NEURON_BITS-1:0]     prog_route_src_neuron;
    reg  [ROUTE_SLOT_BITS-1:0] prog_route_slot;
    reg  [CORE_ID_BITS-1:0]    prog_route_dest_core;
    reg  [NEURON_BITS-1:0]     prog_route_dest_neuron;
    reg  signed [DATA_WIDTH-1:0] prog_route_weight;

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

    async_noc_mesh #(
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
        .THRESHOLD      (16'sd1000),
        .LEAK_RATE      (16'sd3),
        .REFRAC_CYCLES  (3),
        .ROUTE_FANOUT   (ROUTE_FANOUT),
        .ROUTE_SLOT_BITS(ROUTE_SLOT_BITS),
        .MESH_X         (MESH_X),
        .MESH_Y         (MESH_Y)
    ) dut (
        .clk               (clk),
        .rst_n             (rst_n),
        .start             (start),
        .prog_pool_we      (1'b0),
        .prog_pool_core    ({CORE_ID_BITS{1'b0}}),
        .prog_pool_addr    ({POOL_ADDR_BITS{1'b0}}),
        .prog_pool_src     ({NEURON_BITS{1'b0}}),
        .prog_pool_target  ({NEURON_BITS{1'b0}}),
        .prog_pool_weight  ({DATA_WIDTH{1'b0}}),
        .prog_pool_comp    (2'd0),
        .prog_index_we     (1'b0),
        .prog_index_core   ({CORE_ID_BITS{1'b0}}),
        .prog_index_neuron ({NEURON_BITS{1'b0}}),
        .prog_index_base   ({POOL_ADDR_BITS{1'b0}}),
        .prog_index_count  ({COUNT_BITS{1'b0}}),
        .prog_index_format (2'd0),
        .prog_route_we         (prog_route_we),
        .prog_route_src_core   (prog_route_src_core),
        .prog_route_src_neuron (prog_route_src_neuron),
        .prog_route_slot       (prog_route_slot),
        .prog_route_dest_core  (prog_route_dest_core),
        .prog_route_dest_neuron(prog_route_dest_neuron),
        .prog_route_weight     (prog_route_weight),
        .prog_global_route_we          (1'b0),
        .prog_global_route_src_core    ({CORE_ID_BITS{1'b0}}),
        .prog_global_route_src_neuron  ({NEURON_BITS{1'b0}}),
        .prog_global_route_slot        (2'b0),
        .prog_global_route_dest_core   ({CORE_ID_BITS{1'b0}}),
        .prog_global_route_dest_neuron ({NEURON_BITS{1'b0}}),
        .prog_global_route_weight      ({DATA_WIDTH{1'b0}}),
        .learn_enable      (1'b0),
        .graded_enable     (1'b0),
        .dendritic_enable  (1'b0),
        .async_enable      (1'b0),
        .threefactor_enable(1'b0),
        .noise_enable      (1'b0),
        .skip_idle_enable  (1'b0),
        .scale_u_enable    (1'b0),
        .reward_value      ({DATA_WIDTH{1'b0}}),
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
        .ext_valid         (ext_valid),
        .ext_core          (ext_core),
        .ext_neuron_id     (ext_neuron_id),
        .ext_current       (ext_current),
        .probe_read        (probe_read),
        .probe_core        (probe_core),
        .probe_neuron      (probe_neuron),
        .probe_state_id    (probe_state_id),
        .probe_pool_addr   (probe_pool_addr),
        .probe_data        (probe_data),
        .probe_valid       (probe_valid),
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

    always @(posedge clk) begin : spike_monitor
        integer c;
        for (c = 0; c < NUM_CORES; c = c + 1) begin
            if (spike_valid_bus[c]) begin
                $display("  [ts=%0d] Core %0d Neuron %0d spiked",
                    timestep_count, c, spike_id_bus[c*NEURON_BITS +: NEURON_BITS]);
            end
        end
    end

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

    task add_route;
        input [CORE_ID_BITS-1:0]     src_core;
        input [NEURON_BITS-1:0]      src_neuron;
        input [ROUTE_SLOT_BITS-1:0]  slot;
        input [CORE_ID_BITS-1:0]     dest_core;
        input [NEURON_BITS-1:0]      dest_neuron;
        input signed [DATA_WIDTH-1:0] weight;
    begin
        @(posedge clk);
        prog_route_we          <= 1;
        prog_route_src_core    <= src_core;
        prog_route_src_neuron  <= src_neuron;
        prog_route_slot        <= slot;
        prog_route_dest_core   <= dest_core;
        prog_route_dest_neuron <= dest_neuron;
        prog_route_weight      <= weight;
        @(posedge clk);
        prog_route_we <= 0;
    end
    endtask

    task inject_stim;
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
    end
    endtask

    task run_start;
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
    reg [31:0] spk_before, spk_after;

    initial begin
        #2000000;
        $display("TIMEOUT - simulation exceeded 2ms");
        $finish;
    end

    initial begin
        clk = 0; rst_n = 0;
        start = 0;
        prog_route_we = 0; prog_route_src_core = 0; prog_route_src_neuron = 0;
        prog_route_slot = 0; prog_route_dest_core = 0;
        prog_route_dest_neuron = 0; prog_route_weight = 0;
        prog_param_we = 0; prog_param_core = 0; prog_param_neuron = 0;
        prog_param_id = 0; prog_param_value = 0;
        ext_valid = 0; ext_core = 0; ext_neuron_id = 0; ext_current = 0;
        probe_read = 0; probe_core = 0; probe_neuron = 0;
        probe_state_id = 0; probe_pool_addr = 0;
        pass_count = 0; fail_count = 0;

        #100;
        rst_n = 1;
        #100;

        $display("\nTest 1: Point-to-point XY routing");

        set_param(2'd0, 10'd0, 5'd0, 16'sd100);
        set_param(2'd3, 10'd5, 5'd0, 16'sd100);

        add_route(2'd0, 10'd0, 3'd0, 2'd3, 10'd5, 16'sd200);

        spk_before = total_spikes;
        inject_stim(2'd0, 10'd0, 16'sd200);
        run_start;
        $display("  After TS1: total_spikes=%0d", total_spikes);

        run_start;
        spk_after = total_spikes;
        $display("  After TS2: total_spikes=%0d", total_spikes);

        if ((spk_after - spk_before) >= 2) begin
            $display("  PASSED: point-to-point delivered (%0d spikes)", spk_after - spk_before);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: expected >= 2 spikes, got %0d", spk_after - spk_before);
            fail_count = fail_count + 1;
        end

        $display("\nTest 2: Multicast routing");

        set_param(2'd0, 10'd1, 5'd0, 16'sd100);
        set_param(2'd1, 10'd10, 5'd0, 16'sd100);
        set_param(2'd2, 10'd10, 5'd0, 16'sd100);
        set_param(2'd3, 10'd10, 5'd0, 16'sd100);

        add_route(2'd0, 10'd1, 3'd0, 2'd1, 10'd10, 16'sd200);
        add_route(2'd0, 10'd1, 3'd1, 2'd2, 10'd10, 16'sd200);
        add_route(2'd0, 10'd1, 3'd2, 2'd3, 10'd10, 16'sd200);

        spk_before = total_spikes;
        inject_stim(2'd0, 10'd1, 16'sd200);
        run_start;
        $display("  After TS1: total_spikes=%0d (source spike)", total_spikes);

        run_start;
        spk_after = total_spikes;
        $display("  After TS2: total_spikes=%0d", total_spikes);

        if ((spk_after - spk_before) >= 4) begin
            $display("  PASSED: multicast delivered (%0d spikes, expect 4)", spk_after - spk_before);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: expected >= 4 spikes, got %0d", spk_after - spk_before);
            fail_count = fail_count + 1;
        end

        $display("\nTest 3: Contention resolution");

        set_param(2'd0, 10'd2, 5'd0, 16'sd100);
        set_param(2'd1, 10'd2, 5'd0, 16'sd100);
        set_param(2'd3, 10'd20, 5'd0, 16'sd100);

        add_route(2'd0, 10'd2, 3'd0, 2'd3, 10'd20, 16'sd200);
        add_route(2'd1, 10'd2, 3'd0, 2'd3, 10'd20, 16'sd200);

        spk_before = total_spikes;
        inject_stim(2'd0, 10'd2, 16'sd200);
        inject_stim(2'd1, 10'd2, 16'sd200);
        run_start;
        $display("  After TS1: total_spikes=%0d (2 source spikes)", total_spikes);

        run_start;
        spk_after = total_spikes;
        $display("  After TS2: total_spikes=%0d", total_spikes);

        if ((spk_after - spk_before) >= 3) begin
            $display("  PASSED: contention resolved (%0d spikes, expect 3)", spk_after - spk_before);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: expected >= 3 spikes, got %0d", spk_after - spk_before);
            fail_count = fail_count + 1;
        end

        $display("\nTest 4: Chain propagation");

        set_param(2'd0, 10'd3, 5'd0, 16'sd100);
        set_param(2'd1, 10'd3, 5'd0, 16'sd100);
        set_param(2'd2, 10'd3, 5'd0, 16'sd100);
        set_param(2'd3, 10'd3, 5'd0, 16'sd100);

        add_route(2'd0, 10'd3, 3'd0, 2'd1, 10'd3, 16'sd200);
        add_route(2'd1, 10'd3, 3'd0, 2'd2, 10'd3, 16'sd200);
        add_route(2'd2, 10'd3, 3'd0, 2'd3, 10'd3, 16'sd200);

        spk_before = total_spikes;

        inject_stim(2'd0, 10'd3, 16'sd200);
        run_start;
        $display("  After TS1: total_spikes=%0d (chain hop 1)", total_spikes);

        run_start;
        $display("  After TS2: total_spikes=%0d (chain hop 2)", total_spikes);

        run_start;
        $display("  After TS3: total_spikes=%0d (chain hop 3)", total_spikes);

        run_start;
        spk_after = total_spikes;
        $display("  After TS4: total_spikes=%0d (chain hop 4)", total_spikes);

        if ((spk_after - spk_before) >= 4) begin
            $display("  PASSED: chain propagated (%0d spikes over 4 TS)", spk_after - spk_before);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: expected >= 4 chain spikes, got %0d", spk_after - spk_before);
            fail_count = fail_count + 1;
        end

        $display("P22E RESULTS: %0d/%0d passed", pass_count, pass_count + fail_count);

        if (fail_count > 0)
            $display("SOME TESTS FAILED");

        $finish;
    end

endmodule
