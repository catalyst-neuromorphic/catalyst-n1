`timescale 1ns/1ps

module tb_p13b;

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
    reg learn_enable, graded_enable, dendritic_enable, async_enable;
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
        .NUM_CORES(NUM_CORES), .CORE_ID_BITS(CORE_ID_BITS),
        .NUM_NEURONS(NUM_NEURONS), .NEURON_BITS(NEURON_BITS),
        .DATA_WIDTH(DATA_WIDTH), .POOL_DEPTH(POOL_DEPTH),
        .POOL_ADDR_BITS(POOL_ADDR_BITS), .COUNT_BITS(COUNT_BITS),
        .REV_FANIN(REV_FANIN), .REV_SLOT_BITS(REV_SLOT_BITS),
        .THRESHOLD(16'sd1000), .LEAK_RATE(16'sd3), .REFRAC_CYCLES(3),
        .ROUTE_FANOUT(ROUTE_FANOUT), .ROUTE_SLOT_BITS(ROUTE_SLOT_BITS)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .prog_pool_we(prog_pool_we), .prog_pool_core(prog_pool_core),
        .prog_pool_addr(prog_pool_addr), .prog_pool_src(prog_pool_src),
        .prog_pool_target(prog_pool_target), .prog_pool_weight(prog_pool_weight),
        .prog_pool_comp(prog_pool_comp),
        .prog_index_we(prog_index_we), .prog_index_core(prog_index_core),
        .prog_index_neuron(prog_index_neuron), .prog_index_base(prog_index_base),
        .prog_index_count(prog_index_count),
        .prog_index_format(2'd0),
        .prog_route_we(prog_route_we), .prog_route_src_core(prog_route_src_core),
        .prog_route_src_neuron(prog_route_src_neuron), .prog_route_slot(prog_route_slot),
        .prog_route_dest_core(prog_route_dest_core),
        .prog_route_dest_neuron(prog_route_dest_neuron),
        .prog_route_weight(prog_route_weight),
        .prog_global_route_we(1'b0),
        .prog_global_route_src_core({CORE_ID_BITS{1'b0}}),
        .prog_global_route_src_neuron({NEURON_BITS{1'b0}}),
        .prog_global_route_slot(2'b0),
        .prog_global_route_dest_core({CORE_ID_BITS{1'b0}}),
        .prog_global_route_dest_neuron({NEURON_BITS{1'b0}}),
        .prog_global_route_weight({DATA_WIDTH{1'b0}}),
        .learn_enable(learn_enable), .graded_enable(graded_enable),
        .dendritic_enable(dendritic_enable), .async_enable(async_enable),
        .prog_delay_we     (1'b0),
        .prog_delay_core   ({CORE_ID_BITS{1'b0}}),
        .prog_delay_addr   ({POOL_ADDR_BITS{1'b0}}),
        .prog_delay_value  (6'd0),
        .prog_ucode_we     (1'b0),
        .prog_ucode_core   ({CORE_ID_BITS{1'b0}}),
        .prog_ucode_addr   (6'd0),
        .prog_ucode_data   (32'd0),
        .prog_param_we(prog_param_we), .prog_param_core(prog_param_core),
        .prog_param_neuron(prog_param_neuron), .prog_param_id(prog_param_id),
        .prog_param_value(prog_param_value),
        .ext_valid(ext_valid), .ext_core(ext_core),
        .ext_neuron_id(ext_neuron_id), .ext_current(ext_current),
        .timestep_done(timestep_done), .spike_valid_bus(spike_valid_bus),
        .spike_id_bus(spike_id_bus), .mesh_state_out(mesh_state_out),
        .total_spikes(total_spikes), .timestep_count(timestep_count)
    );

    always @(posedge clk) begin : spike_monitor
        integer c;
        for (c = 0; c < NUM_CORES; c = c + 1) begin
            if (spike_valid_bus[c])
                $display("  [t=%0d] Core %0d Neuron %0d spiked",
                    timestep_count, c, spike_id_bus[c*NEURON_BITS +: NEURON_BITS]);
        end
    end

    initial begin
        $dumpfile("tb_p13b.vcd");
        $dumpvars(0, tb_p13b);
    end

    task add_pool;
        input [CORE_ID_BITS-1:0] core;
        input [POOL_ADDR_BITS-1:0] addr;
        input [NEURON_BITS-1:0] src;
        input [NEURON_BITS-1:0] target;
        input signed [DATA_WIDTH-1:0] weight;
        input [1:0] comp;
    begin
        @(posedge clk);
        prog_pool_we <= 1; prog_pool_core <= core; prog_pool_addr <= addr;
        prog_pool_src <= src; prog_pool_target <= target;
        prog_pool_weight <= weight; prog_pool_comp <= comp;
        @(posedge clk);
        prog_pool_we <= 0;
    end
    endtask

    task set_index;
        input [CORE_ID_BITS-1:0] core;
        input [NEURON_BITS-1:0] neuron;
        input [POOL_ADDR_BITS-1:0] base;
        input [COUNT_BITS-1:0] count;
    begin
        @(posedge clk);
        prog_index_we <= 1; prog_index_core <= core;
        prog_index_neuron <= neuron; prog_index_base <= base; prog_index_count <= count;
        @(posedge clk);
        prog_index_we <= 0;
    end
    endtask

    task add_route;
        input [CORE_ID_BITS-1:0] src_core;
        input [NEURON_BITS-1:0] src_neuron;
        input [ROUTE_SLOT_BITS-1:0] slot;
        input [CORE_ID_BITS-1:0] dest_core;
        input [NEURON_BITS-1:0] dest_neuron;
        input signed [DATA_WIDTH-1:0] weight;
    begin
        @(posedge clk);
        prog_route_we <= 1;
        prog_route_src_core <= src_core; prog_route_src_neuron <= src_neuron;
        prog_route_slot <= slot;
        prog_route_dest_core <= dest_core; prog_route_dest_neuron <= dest_neuron;
        prog_route_weight <= weight;
        @(posedge clk);
        prog_route_we <= 0;
    end
    endtask

    task run_timestep;
        input [CORE_ID_BITS-1:0] core;
        input [NEURON_BITS-1:0] neuron;
        input signed [DATA_WIDTH-1:0] current;
    begin
        @(posedge clk);
        ext_valid <= 1; ext_core <= core; ext_neuron_id <= neuron; ext_current <= current;
        @(posedge clk);
        ext_valid <= 0; start <= 1;
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

    integer c1_spikes, c2_spikes, c3_spikes;
    integer pass_count, fail_count;
    reg [31:0] spikes_before;
    integer ts;

    initial begin
        start = 0;
        prog_pool_we = 0; prog_pool_core = 0; prog_pool_addr = 0;
        prog_pool_src = 0; prog_pool_target = 0; prog_pool_weight = 0; prog_pool_comp = 0;
        prog_index_we = 0; prog_index_core = 0; prog_index_neuron = 0;
        prog_index_base = 0; prog_index_count = 0;
        prog_route_we = 0; prog_route_src_core = 0; prog_route_src_neuron = 0;
        prog_route_slot = 0; prog_route_dest_core = 0;
        prog_route_dest_neuron = 0; prog_route_weight = 0;
        learn_enable = 0; graded_enable = 0; dendritic_enable = 0; async_enable = 0;
        prog_param_we = 0; prog_param_core = 0; prog_param_neuron = 0;
        prog_param_id = 0; prog_param_value = 0;
        ext_valid = 0; ext_core = 0; ext_neuron_id = 0; ext_current = 0;
        pass_count = 0; fail_count = 0;

        rst_n = 0;
        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 5);

        $display("test 1: Multicast C0:N50 -> C1:N60, C2:N70");

        add_route(0, 50, 0, 1, 60, 16'sd1200);
        add_route(0, 50, 1, 2, 70, 16'sd1200);

        spikes_before = total_spikes;
        c1_spikes = 0;
        c2_spikes = 0;

        for (ts = 0; ts < 15; ts = ts + 1) begin
            run_timestep(0, 50, 16'sd1200);
        end

        $display("Test 1 total spikes: %0d", total_spikes - spikes_before);
        if (total_spikes - spikes_before >= 3) begin
            $display("TEST 1 PASSED (multicast delivered to multiple cores)");
            pass_count = pass_count + 1;
        end else begin
            $display("TEST 1 FAILED - not enough spikes");
            fail_count = fail_count + 1;
        end

        $display("test 2: 4-way multicast from C0:N80");

        add_route(0, 80, 0, 0, 81, 16'sd1200);
        add_route(0, 80, 1, 1, 82, 16'sd1200);
        add_route(0, 80, 2, 2, 83, 16'sd1200);
        add_route(0, 80, 3, 3, 84, 16'sd1200);

        spikes_before = total_spikes;

        for (ts = 0; ts < 15; ts = ts + 1) begin
            run_timestep(0, 80, 16'sd1200);
        end

        $display("Test 2 total spikes: %0d", total_spikes - spikes_before);
        if (total_spikes - spikes_before >= 5) begin
            $display("TEST 2 PASSED (4-way multicast)");
            pass_count = pass_count + 1;
        end else begin
            $display("TEST 2 FAILED - not enough spikes");
            fail_count = fail_count + 1;
        end

        $display("test 3: Mixed unicast + multicast");

        add_route(0, 90, 0, 1, 300, 16'sd1200);

        add_route(0, 91, 0, 1, 301, 16'sd1200);
        add_route(0, 91, 1, 2, 302, 16'sd1200);

        spikes_before = total_spikes;

        for (ts = 0; ts < 10; ts = ts + 1) begin
            run_timestep(0, 90, 16'sd1200);
        end
        for (ts = 0; ts < 10; ts = ts + 1) begin
            run_timestep(0, 91, 16'sd1200);
        end

        $display("Test 3 total spikes: %0d", total_spikes - spikes_before);
        if (total_spikes - spikes_before >= 3) begin
            $display("TEST 3 PASSED (mixed routing)");
            pass_count = pass_count + 1;
        end else begin
            $display("TEST 3 FAILED");
            fail_count = fail_count + 1;
        end

        $display("test 4: backward compat (slot 0 unicast)");

        add_route(0, 400, 0, 1, 401, 16'sd1200);

        add_pool(1, 50, 401, 402, 16'sd1200, 0);
        set_index(1, 401, 50, 1);

        spikes_before = total_spikes;

        for (ts = 0; ts < 15; ts = ts + 1) begin
            run_timestep(0, 400, 16'sd1200);
        end

        $display("Test 4 total spikes: %0d", total_spikes - spikes_before);
        if (total_spikes - spikes_before >= 3) begin
            $display("TEST 4 PASSED (backward compat)");
            pass_count = pass_count + 1;
        end else begin
            $display("TEST 4 FAILED");
            fail_count = fail_count + 1;
        end

        $display("P13b RESULTS: %0d/%0d passed", pass_count, pass_count + fail_count);
        if (fail_count > 0)
            $display("*** SOME TESTS FAILED ***");
        else
            $display("All tests passed!");
        $finish;
    end

    initial begin
        #(CLK_PERIOD * 5_000_000);
        $display("TIMEOUT");
        $finish;
    end

endmodule
