`timescale 1ps/1ps

module tb_p21a_dendrites;

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

    task prog_conn(input [CORE_ID_BITS-1:0] core,
                   input [POOL_ADDR_BITS-1:0] addr,
                   input [NEURON_BITS-1:0] src, target,
                   input signed [DATA_WIDTH-1:0] weight,
                   input [1:0] comp);
        begin
            prog_pool_we <= 1; prog_pool_core <= core;
            prog_pool_addr <= addr; prog_pool_src <= src;
            prog_pool_target <= target; prog_pool_weight <= weight;
            prog_pool_comp <= comp;
            @(posedge clk); prog_pool_we <= 0; @(posedge clk);
        end
    endtask

    task prog_idx(input [CORE_ID_BITS-1:0] core,
                  input [NEURON_BITS-1:0] neuron,
                  input [POOL_ADDR_BITS-1:0] base,
                  input [COUNT_BITS-1:0] count);
        begin
            prog_index_we <= 1; prog_index_core <= core;
            prog_index_neuron <= neuron; prog_index_base <= base;
            prog_index_count <= count; prog_index_format <= 2'd0;
            @(posedge clk); prog_index_we <= 0; @(posedge clk);
        end
    endtask

    task prog_param(input [CORE_ID_BITS-1:0] core,
                    input [NEURON_BITS-1:0] neuron,
                    input [3:0] param_id,
                    input signed [DATA_WIDTH-1:0] value);
        begin
            prog_param_we <= 1; prog_param_core <= core;
            prog_param_neuron <= neuron; prog_param_id <= param_id;
            prog_param_value <= value;
            @(posedge clk); prog_param_we <= 0; @(posedge clk);
        end
    endtask

    task inject(input [CORE_ID_BITS-1:0] core,
                input [NEURON_BITS-1:0] neuron,
                input signed [DATA_WIDTH-1:0] current);
        begin
            ext_valid <= 1; ext_core <= core;
            ext_neuron_id <= neuron; ext_current <= current;
            @(posedge clk); ext_valid <= 0; @(posedge clk);
        end
    endtask

    integer pass_count, fail_count;
    reg signed [DATA_WIDTH-1:0] probed_val;

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

        dendritic_enable = 1;

        $display("test 1: Flat mode (all parent=0, default)");

        prog_conn(0, 0, 0, 10, 16'sd300, 2'd1);
        prog_idx(0, 0, 0, 1);

        prog_conn(0, 1, 1, 10, 16'sd200, 2'd2);
        prog_idx(0, 1, 1, 1);

        inject(0, 0, 16'sd1500);
        inject(0, 1, 16'sd1500);

        run_timestep;
        run_timestep;

        do_probe(0, 10, 4'd0, 0);
        probed_val = $signed(probe_data);
        $display("  Neuron 10 membrane potential = %0d", probed_val);
        if (probed_val > 16'sd400 && probed_val < 16'sd600) begin
            $display("TEST 1 PASSED (flat dendrites, potential=%0d, expected ~497)", probed_val);
            pass_count = pass_count + 1;
        end else begin
            $display("TEST 1 FAILED (potential=%0d, expected ~497)", probed_val);
            fail_count = fail_count + 1;
        end

        $display("test 2: Chain mode (dend3->dend2->dend1->soma)");

        prog_param(0, 20, 4'd15, 16'sd36);

        prog_param(0, 20, 4'd8,  16'sd100);
        prog_param(0, 20, 4'd9,  16'sd50);
        prog_param(0, 20, 4'd10, 16'sd20);

        prog_conn(0, 2, 5, 20, 16'sd500, 2'd3);
        prog_idx(0, 5, 2, 1);

        inject(0, 5, 16'sd1500);

        run_timestep;
        run_timestep;

        do_probe(0, 20, 4'd0, 0);
        probed_val = $signed(probe_data);
        $display("  Neuron 20 membrane potential = %0d", probed_val);
        if (probed_val > 16'sd250 && probed_val < 16'sd400) begin
            $display("TEST 2 PASSED (chain dendrites, potential=%0d, expected ~327)", probed_val);
            pass_count = pass_count + 1;
        end else begin
            $display("TEST 2 FAILED (potential=%0d, expected ~327)", probed_val);
            fail_count = fail_count + 1;
        end

        $display("test 3: Fan-in mode (dend2,dend3->dend1->soma)");

        prog_param(0, 30, 4'd15, 16'sd20);

        prog_param(0, 30, 4'd8,  16'sd50);
        prog_param(0, 30, 4'd9,  16'sd0);
        prog_param(0, 30, 4'd10, 16'sd0);

        prog_conn(0, 3, 6, 30, 16'sd200, 2'd2);
        prog_idx(0, 6, 3, 1);

        prog_conn(0, 4, 7, 30, 16'sd150, 2'd3);
        prog_idx(0, 7, 4, 1);

        inject(0, 6, 16'sd1500);
        inject(0, 7, 16'sd1500);

        run_timestep;
        run_timestep;

        do_probe(0, 30, 4'd0, 0);
        probed_val = $signed(probe_data);
        $display("  Neuron 30 membrane potential = %0d", probed_val);
        if (probed_val > 16'sd220 && probed_val < 16'sd380) begin
            $display("TEST 3 PASSED (fan-in dendrites, potential=%0d, expected ~297)", probed_val);
            pass_count = pass_count + 1;
        end else begin
            $display("TEST 3 FAILED (potential=%0d, expected ~297)", probed_val);
            fail_count = fail_count + 1;
        end

        $display("test 4: Tree dendrites cause spike");

        prog_param(0, 40, 4'd15, 16'sd36);

        prog_param(0, 40, 4'd8,  16'sd100);
        prog_param(0, 40, 4'd9,  16'sd50);
        prog_param(0, 40, 4'd10, 16'sd20);

        prog_conn(0, 5, 8, 40, 16'sd1200, 2'd3);
        prog_idx(0, 8, 5, 1);

        inject(0, 8, 16'sd1500);

        run_timestep;

        begin : test4_block
            reg [31:0] spikes_before;
            spikes_before = total_spikes;

            run_timestep;

            $display("  Spikes in delivery timestep = %0d", total_spikes - spikes_before);
            if (total_spikes > spikes_before) begin
                $display("TEST 4 PASSED (tree dendrite spike, new spikes=%0d)", total_spikes - spikes_before);
                pass_count = pass_count + 1;
            end else begin
                $display("TEST 4 FAILED (expected spike from neuron 40, got 0 new spikes)");
                fail_count = fail_count + 1;
            end
        end

        $display("P21A RESULTS: %0d/%0d passed", pass_count, pass_count + fail_count);
        if (fail_count == 0)
            $display("All tests passed!");
        else
            $display("SOME TESTS FAILED");
        $finish;
    end

endmodule
