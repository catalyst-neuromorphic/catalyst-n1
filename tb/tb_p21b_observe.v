`timescale 1ps/1ps

module tb_p21b_observe;

    localparam NUM_CORES    = 4;
    localparam CORE_ID_BITS = 2;
    localparam NUM_NEURONS  = 1024;
    localparam NEURON_BITS  = 10;
    localparam DATA_WIDTH   = 16;
    localparam POOL_DEPTH   = 1024;
    localparam POOL_ADDR_BITS = 10;
    localparam COUNT_BITS   = 10;
    localparam THRESHOLD    = 16'sd1000;
    localparam LEAK_RATE    = 16'sd3;
    localparam ROUTE_FANOUT = 8;
    localparam ROUTE_SLOT_BITS = 3;
    localparam GLOBAL_ROUTE_SLOTS = 4;
    localparam GLOBAL_ROUTE_SLOT_BITS = 2;

    reg clk, rst_n;

    always #5000 clk = ~clk;

    reg start;
    reg prog_pool_we;
    reg [CORE_ID_BITS-1:0] prog_pool_core;
    reg [POOL_ADDR_BITS-1:0] prog_pool_addr;
    reg [NEURON_BITS-1:0] prog_pool_src, prog_pool_target;
    reg signed [DATA_WIDTH-1:0] prog_pool_weight;
    reg [1:0] prog_pool_comp;

    reg prog_index_we;
    reg [CORE_ID_BITS-1:0] prog_index_core;
    reg [NEURON_BITS-1:0] prog_index_neuron;
    reg [POOL_ADDR_BITS-1:0] prog_index_base;
    reg [COUNT_BITS-1:0] prog_index_count;
    reg [1:0] prog_index_format;

    reg prog_route_we;
    reg [CORE_ID_BITS-1:0] prog_route_src_core;
    reg [NEURON_BITS-1:0] prog_route_src_neuron;
    reg [ROUTE_SLOT_BITS-1:0] prog_route_slot;
    reg [CORE_ID_BITS-1:0] prog_route_dest_core;
    reg [NEURON_BITS-1:0] prog_route_dest_neuron;
    reg signed [DATA_WIDTH-1:0] prog_route_weight;

    reg prog_global_route_we;
    reg [CORE_ID_BITS-1:0] prog_global_route_src_core;
    reg [NEURON_BITS-1:0] prog_global_route_src_neuron;
    reg [GLOBAL_ROUTE_SLOT_BITS-1:0] prog_global_route_slot;
    reg [CORE_ID_BITS-1:0] prog_global_route_dest_core;
    reg [NEURON_BITS-1:0] prog_global_route_dest_neuron;
    reg signed [DATA_WIDTH-1:0] prog_global_route_weight;

    reg learn_enable, graded_enable, dendritic_enable, async_enable;
    reg threefactor_enable, noise_enable;
    reg signed [DATA_WIDTH-1:0] reward_value;

    reg prog_delay_we;
    reg [CORE_ID_BITS-1:0] prog_delay_core;
    reg [POOL_ADDR_BITS-1:0] prog_delay_addr;
    reg [5:0] prog_delay_value;

    reg prog_ucode_we;
    reg [CORE_ID_BITS-1:0] prog_ucode_core;
    reg [5:0] prog_ucode_addr;
    reg [31:0] prog_ucode_data;

    reg prog_param_we;
    reg [CORE_ID_BITS-1:0] prog_param_core;
    reg [NEURON_BITS-1:0] prog_param_neuron;
    reg [3:0] prog_param_id;
    reg signed [DATA_WIDTH-1:0] prog_param_value;

    reg ext_valid;
    reg [CORE_ID_BITS-1:0] ext_core;
    reg [NEURON_BITS-1:0] ext_neuron_id;
    reg signed [DATA_WIDTH-1:0] ext_current;

    reg probe_read;
    reg [CORE_ID_BITS-1:0] probe_core;
    reg [NEURON_BITS-1:0] probe_neuron;
    reg [3:0] probe_state_id;
    reg [POOL_ADDR_BITS-1:0] probe_pool_addr;
    wire signed [DATA_WIDTH-1:0] probe_data;
    wire probe_valid;

    wire timestep_done;
    wire [NUM_CORES-1:0] spike_valid_bus;
    wire [NUM_CORES*NEURON_BITS-1:0] spike_id_bus;
    wire [5:0] mesh_state_out;
    wire [31:0] total_spikes, timestep_count;

    neuromorphic_mesh #(
        .NUM_CORES(NUM_CORES), .CORE_ID_BITS(CORE_ID_BITS),
        .NUM_NEURONS(NUM_NEURONS), .NEURON_BITS(NEURON_BITS),
        .DATA_WIDTH(DATA_WIDTH), .POOL_DEPTH(POOL_DEPTH),
        .POOL_ADDR_BITS(POOL_ADDR_BITS), .COUNT_BITS(COUNT_BITS),
        .THRESHOLD(THRESHOLD), .LEAK_RATE(LEAK_RATE),
        .ROUTE_FANOUT(ROUTE_FANOUT), .ROUTE_SLOT_BITS(ROUTE_SLOT_BITS),
        .GLOBAL_ROUTE_SLOTS(GLOBAL_ROUTE_SLOTS),
        .GLOBAL_ROUTE_SLOT_BITS(GLOBAL_ROUTE_SLOT_BITS)
    ) uut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .prog_pool_we(prog_pool_we), .prog_pool_core(prog_pool_core),
        .prog_pool_addr(prog_pool_addr), .prog_pool_src(prog_pool_src),
        .prog_pool_target(prog_pool_target), .prog_pool_weight(prog_pool_weight),
        .prog_pool_comp(prog_pool_comp),
        .prog_index_we(prog_index_we), .prog_index_core(prog_index_core),
        .prog_index_neuron(prog_index_neuron), .prog_index_base(prog_index_base),
        .prog_index_count(prog_index_count), .prog_index_format(prog_index_format),
        .prog_route_we(prog_route_we), .prog_route_src_core(prog_route_src_core),
        .prog_route_src_neuron(prog_route_src_neuron), .prog_route_slot(prog_route_slot),
        .prog_route_dest_core(prog_route_dest_core),
        .prog_route_dest_neuron(prog_route_dest_neuron),
        .prog_route_weight(prog_route_weight),
        .prog_global_route_we(prog_global_route_we),
        .prog_global_route_src_core(prog_global_route_src_core),
        .prog_global_route_src_neuron(prog_global_route_src_neuron),
        .prog_global_route_slot(prog_global_route_slot),
        .prog_global_route_dest_core(prog_global_route_dest_core),
        .prog_global_route_dest_neuron(prog_global_route_dest_neuron),
        .prog_global_route_weight(prog_global_route_weight),
        .learn_enable(learn_enable), .graded_enable(graded_enable),
        .dendritic_enable(dendritic_enable), .async_enable(async_enable),
        .threefactor_enable(threefactor_enable), .noise_enable(noise_enable),
        .reward_value(reward_value),
        .prog_delay_we(prog_delay_we), .prog_delay_core(prog_delay_core),
        .prog_delay_addr(prog_delay_addr), .prog_delay_value(prog_delay_value),
        .prog_ucode_we(prog_ucode_we), .prog_ucode_core(prog_ucode_core),
        .prog_ucode_addr(prog_ucode_addr), .prog_ucode_data(prog_ucode_data),
        .prog_param_we(prog_param_we), .prog_param_core(prog_param_core),
        .prog_param_neuron(prog_param_neuron), .prog_param_id(prog_param_id),
        .prog_param_value(prog_param_value),
        .probe_read(probe_read), .probe_core(probe_core),
        .probe_neuron(probe_neuron), .probe_state_id(probe_state_id),
        .probe_pool_addr(probe_pool_addr),
        .probe_data(probe_data), .probe_valid(probe_valid),
        .ext_valid(ext_valid), .ext_core(ext_core),
        .ext_neuron_id(ext_neuron_id), .ext_current(ext_current),
        .timestep_done(timestep_done), .spike_valid_bus(spike_valid_bus),
        .spike_id_bus(spike_id_bus), .mesh_state_out(mesh_state_out),
        .total_spikes(total_spikes), .timestep_count(timestep_count)
    );

    task clear_prog;
        begin
            prog_pool_we <= 0; prog_index_we <= 0; prog_route_we <= 0;
            prog_global_route_we <= 0; prog_delay_we <= 0; prog_ucode_we <= 0;
            prog_param_we <= 0; ext_valid <= 0;
        end
    endtask

    task run_timestep;
        begin
            start <= 1; @(posedge clk); start <= 0;
            wait(timestep_done); @(posedge clk);
        end
    endtask

    task do_probe(input [CORE_ID_BITS-1:0] core, input [NEURON_BITS-1:0] neuron,
                  input [3:0] sid, input [POOL_ADDR_BITS-1:0] paddr);
        begin
            probe_read <= 1;
            probe_core <= core;
            probe_neuron <= neuron;
            probe_state_id <= sid;
            probe_pool_addr <= paddr;
            @(posedge clk);
            probe_read <= 0;
            wait(probe_valid);
            @(posedge clk);
        end
    endtask

    integer pass_count, fail_count;

    initial begin

        clk = 0; rst_n = 0;
        start = 0;
        clear_prog;
        learn_enable = 0; graded_enable = 0; dendritic_enable = 0;
        async_enable = 0; threefactor_enable = 0; noise_enable = 0;
        reward_value = 0;
        probe_read = 0; probe_core = 0; probe_neuron = 0;
        probe_state_id = 0; probe_pool_addr = 0;

        pass_count = 0; fail_count = 0;

        #20000 rst_n = 1;
        @(posedge clk); @(posedge clk);

        $display("test 1: Read membrane potential after stimulus");
        ext_valid <= 1; ext_core <= 0; ext_neuron_id <= 5; ext_current <= 600;
        @(posedge clk); ext_valid <= 0;
        @(posedge clk);

        run_timestep;

        do_probe(0, 5, 4'd0, 0);
        $display("  Probe: membrane potential of core 0, neuron 5 = %0d", $signed(probe_data));
        if ($signed(probe_data) > 0 && $signed(probe_data) < 700) begin
            $display("TEST 1 PASSED (membrane potential = %0d, expected ~597)", $signed(probe_data));
            pass_count = pass_count + 1;
        end else begin
            $display("TEST 1 FAILED (membrane potential = %0d, expected ~597)", $signed(probe_data));
            fail_count = fail_count + 1;
        end

        $display("test 2: Read weight from pool");
        prog_pool_we <= 1; prog_pool_core <= 0; prog_pool_addr <= 0;
        prog_pool_src <= 10; prog_pool_target <= 20;
        prog_pool_weight <= 500; prog_pool_comp <= 0;
        @(posedge clk); prog_pool_we <= 0;
        @(posedge clk); @(posedge clk);

        do_probe(0, 0, 4'd11, 10'd0);
        $display("  Probe: pool weight at addr 0, core 0 = %0d", $signed(probe_data));
        if ($signed(probe_data) == 500) begin
            $display("TEST 2 PASSED (weight = 500)");
            pass_count = pass_count + 1;
        end else begin
            $display("TEST 2 FAILED (weight = %0d, expected 500)", $signed(probe_data));
            fail_count = fail_count + 1;
        end

        $display("test 3: Read threshold parameter");
        prog_param_we <= 1; prog_param_core <= 0; prog_param_neuron <= 50;
        prog_param_id <= 0; prog_param_value <= 1234;
        @(posedge clk); prog_param_we <= 0;
        @(posedge clk); @(posedge clk);

        do_probe(0, 50, 4'd1, 0);
        $display("  Probe: threshold of core 0, neuron 50 = %0d", $signed(probe_data));
        if ($signed(probe_data) == 1234) begin
            $display("TEST 3 PASSED (threshold = 1234)");
            pass_count = pass_count + 1;
        end else begin
            $display("TEST 3 FAILED (threshold = %0d, expected 1234)", $signed(probe_data));
            fail_count = fail_count + 1;
        end

        $display("test 4: Read trace after spiking");
        ext_valid <= 1; ext_core <= 0; ext_neuron_id <= 100; ext_current <= 2000;
        @(posedge clk); ext_valid <= 0;
        @(posedge clk);

        run_timestep;

        do_probe(0, 100, 4'd2, 0);
        $display("  Probe: trace1 of core 0, neuron 100 = %0d", probe_data);
        if (probe_data > 0) begin
            $display("TEST 4 PASSED (trace1 = %0d, non-zero after spike)", probe_data);
            pass_count = pass_count + 1;
        end else begin
            $display("TEST 4 FAILED (trace1 = %0d, expected > 0 after spike)", probe_data);
            fail_count = fail_count + 1;
        end

        $display("P21B RESULTS: %0d/%0d passed", pass_count, pass_count + fail_count);
        if (fail_count == 0)
            $display("All tests passed!");
        else
            $display("SOME TESTS FAILED");
        $finish;
    end

endmodule
