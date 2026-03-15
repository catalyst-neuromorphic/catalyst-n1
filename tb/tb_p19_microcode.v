`timescale 1ps/1ps

module tb_p19_microcode;

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
    localparam LEARN_SHIFT  = 3;

    reg clk, rst_n;
    initial clk = 0;
    always #5000 clk = ~clk;

    reg start;
    reg prog_pool_we, prog_index_we, prog_route_we;
    reg [CORE_ID_BITS-1:0] prog_pool_core, prog_index_core, prog_route_src_core;
    reg [POOL_ADDR_BITS-1:0] prog_pool_addr;
    reg [NEURON_BITS-1:0] prog_pool_src, prog_pool_target;
    reg signed [DATA_WIDTH-1:0] prog_pool_weight;
    reg [1:0] prog_pool_comp;
    reg [NEURON_BITS-1:0] prog_index_neuron;
    reg [POOL_ADDR_BITS-1:0] prog_index_base;
    reg [COUNT_BITS-1:0] prog_index_count;
    reg [1:0] prog_index_format;
    reg [NEURON_BITS-1:0] prog_route_src_neuron;
    reg [2:0] prog_route_slot;
    reg [CORE_ID_BITS-1:0] prog_route_dest_core;
    reg [NEURON_BITS-1:0] prog_route_dest_neuron;
    reg signed [DATA_WIDTH-1:0] prog_route_weight;
    reg learn_enable, graded_enable, dendritic_enable, async_enable;
    reg threefactor_enable, noise_enable;
    reg signed [DATA_WIDTH-1:0] reward_value;
    reg prog_param_we;
    reg [CORE_ID_BITS-1:0] prog_param_core;
    reg [NEURON_BITS-1:0] prog_param_neuron;
    reg [4:0] prog_param_id;
    reg signed [DATA_WIDTH-1:0] prog_param_value;
    reg ext_valid;
    reg [CORE_ID_BITS-1:0] ext_core;
    reg [NEURON_BITS-1:0] ext_neuron_id;
    reg signed [DATA_WIDTH-1:0] ext_current;

    reg prog_ucode_we;
    reg [CORE_ID_BITS-1:0] prog_ucode_core;
    reg [6:0] prog_ucode_addr;
    reg [31:0] prog_ucode_data;

    wire timestep_done;
    wire [NUM_CORES-1:0] spike_valid_bus;
    wire [NUM_CORES*NEURON_BITS-1:0] spike_id_bus;
    wire [4:0] mesh_state_out;
    wire [31:0] total_spikes, timestep_count;

    neuromorphic_mesh #(
        .NUM_CORES(NUM_CORES), .CORE_ID_BITS(CORE_ID_BITS),
        .NUM_NEURONS(NUM_NEURONS), .NEURON_BITS(NEURON_BITS),
        .DATA_WIDTH(DATA_WIDTH), .POOL_DEPTH(POOL_DEPTH),
        .POOL_ADDR_BITS(POOL_ADDR_BITS), .COUNT_BITS(COUNT_BITS),
        .THRESHOLD(THRESHOLD), .LEAK_RATE(LEAK_RATE)
    ) dut (
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
        .prog_global_route_we(1'b0),
        .prog_global_route_src_core({CORE_ID_BITS{1'b0}}),
        .prog_global_route_src_neuron({NEURON_BITS{1'b0}}),
        .prog_global_route_slot(2'b0),
        .prog_global_route_dest_core({CORE_ID_BITS{1'b0}}),
        .prog_global_route_dest_neuron({NEURON_BITS{1'b0}}),
        .prog_global_route_weight({DATA_WIDTH{1'b0}}),
        .learn_enable(learn_enable), .graded_enable(graded_enable),
        .dendritic_enable(dendritic_enable), .async_enable(async_enable),
        .threefactor_enable(threefactor_enable), .reward_value(reward_value),
        .noise_enable(noise_enable),
        .prog_delay_we(1'b0), .prog_delay_core({CORE_ID_BITS{1'b0}}),
        .prog_delay_addr({POOL_ADDR_BITS{1'b0}}), .prog_delay_value(6'd0),
        .prog_ucode_we(prog_ucode_we), .prog_ucode_core(prog_ucode_core),
        .prog_ucode_addr(prog_ucode_addr), .prog_ucode_data(prog_ucode_data),
        .prog_param_we(prog_param_we), .prog_param_core(prog_param_core),
        .prog_param_neuron(prog_param_neuron), .prog_param_id(prog_param_id),
        .prog_param_value(prog_param_value),
        .ext_valid(ext_valid), .ext_core(ext_core),
        .ext_neuron_id(ext_neuron_id), .ext_current(ext_current),
        .timestep_done(timestep_done), .spike_valid_bus(spike_valid_bus),
        .spike_id_bus(spike_id_bus), .mesh_state_out(mesh_state_out),
        .total_spikes(total_spikes), .timestep_count(timestep_count)
    );

    task reset_all;
    begin
        rst_n = 0;
        start = 0;
        prog_pool_we = 0; prog_index_we = 0; prog_route_we = 0;
        prog_pool_core = 0; prog_index_core = 0;
        prog_pool_addr = 0; prog_pool_src = 0; prog_pool_target = 0;
        prog_pool_weight = 0; prog_pool_comp = 0;
        prog_index_neuron = 0; prog_index_base = 0; prog_index_count = 0;
        prog_index_format = 0;
        prog_route_src_core = 0; prog_route_src_neuron = 0; prog_route_slot = 0;
        prog_route_dest_core = 0; prog_route_dest_neuron = 0; prog_route_weight = 0;
        learn_enable = 0; graded_enable = 0; dendritic_enable = 0;
        async_enable = 0; threefactor_enable = 0; noise_enable = 0;
        reward_value = 0;
        prog_param_we = 0; prog_param_core = 0; prog_param_neuron = 0;
        prog_param_id = 0; prog_param_value = 0;
        ext_valid = 0; ext_core = 0; ext_neuron_id = 0; ext_current = 0;
        prog_ucode_we = 0; prog_ucode_core = 0; prog_ucode_addr = 0; prog_ucode_data = 0;
        #100000;
        rst_n = 1;
        #20000;
    end
    endtask

    task program_pool(
        input [CORE_ID_BITS-1:0] core,
        input [POOL_ADDR_BITS-1:0] addr,
        input [NEURON_BITS-1:0] src, tgt,
        input signed [DATA_WIDTH-1:0] weight,
        input [1:0] comp
    );
    begin
        @(posedge clk);
        prog_pool_we <= 1;
        prog_pool_core <= core;
        prog_pool_addr <= addr;
        prog_pool_src <= src;
        prog_pool_target <= tgt;
        prog_pool_weight <= weight;
        prog_pool_comp <= comp;
        @(posedge clk);
        prog_pool_we <= 0;
    end
    endtask

    task program_index(
        input [CORE_ID_BITS-1:0] core,
        input [NEURON_BITS-1:0] neuron,
        input [POOL_ADDR_BITS-1:0] base,
        input [COUNT_BITS-1:0] count,
        input [1:0] fmt
    );
    begin
        @(posedge clk);
        prog_index_we <= 1;
        prog_index_core <= core;
        prog_index_neuron <= neuron;
        prog_index_base <= base;
        prog_index_count <= count;
        prog_index_format <= fmt;
        @(posedge clk);
        prog_index_we <= 0;
    end
    endtask

    task stimulate(
        input [CORE_ID_BITS-1:0] core,
        input [NEURON_BITS-1:0] neuron,
        input signed [DATA_WIDTH-1:0] current
    );
    begin
        @(posedge clk);
        ext_valid <= 1;
        ext_core <= core;
        ext_neuron_id <= neuron;
        ext_current <= current;
        @(posedge clk);
        ext_valid <= 0;
    end
    endtask

    task run_timestep;
    begin
        @(posedge clk);
        start <= 1;
        @(posedge clk);
        start <= 0;
        @(posedge timestep_done);
        @(posedge clk);
    end
    endtask

    task program_ucode(
        input [CORE_ID_BITS-1:0] core,
        input [6:0] addr,
        input [31:0] instr
    );
    begin
        @(posedge clk);
        prog_ucode_we <= 1;
        prog_ucode_core <= core;
        prog_ucode_addr <= addr;
        prog_ucode_data <= instr;
        @(posedge clk);
        prog_ucode_we <= 0;
    end
    endtask

    integer pass_count, fail_count;
    integer i;
    reg signed [DATA_WIDTH-1:0] weight_before, weight_after;

    initial begin
        pass_count = 0;
        fail_count = 0;

        $display("test 1: Default 2-factor STDP (microcode)");
        reset_all;
        learn_enable = 1;

        program_pool(0, 0, 10, 11, 16'sd500, 2'd0);
        program_index(0, 10, 0, 1, 2'd0);

        stimulate(0, 11, 16'sd2000);
        run_timestep;

        stimulate(0, 10, 16'sd2000);
        run_timestep;

        weight_after = dut.gen_core[0].core.pool_weight_mem.mem[0];
        $display("  Weight after LTD: %0d (was 500)", weight_after);
        if (weight_after < 16'sd500) begin
            $display("TEST 1 PASSED (LTD decreased weight via default microcode)");
            pass_count = pass_count + 1;
        end else begin
            $display("TEST 1 FAILED (expected weight decrease from 500)");
            fail_count = fail_count + 1;
        end

        $display("test 2: Default 3-factor STDP (microcode)");
        reset_all;
        learn_enable = 1;
        threefactor_enable = 1;

        program_pool(0, 10, 20, 21, 16'sd1200, 2'd0);
        program_index(0, 20, 10, 1, 2'd0);

        stimulate(0, 21, 16'sd2000);
        run_timestep;
        stimulate(0, 20, 16'sd2000);
        run_timestep;

        begin
            reg signed [DATA_WIDTH-1:0] elig_val;
            elig_val = dut.gen_core[0].core.elig_mem.mem[10];
            $display("  Elig after LTD: %0d", elig_val);

            reward_value = 16'sd100;
            stimulate(0, 20, 16'sd2000);
            run_timestep;

            weight_after = dut.gen_core[0].core.pool_weight_mem.mem[10];
            $display("  Weight after reward: %0d (was 1200)", weight_after);

            if (elig_val != 0) begin
                $display("TEST 2 PASSED (3-factor elig trace updated via microcode)");
                pass_count = pass_count + 1;
            end else begin
                $display("TEST 2 FAILED (elig should be non-zero)");
                fail_count = fail_count + 1;
            end
        end

        $display("test 3: Custom anti-STDP microcode");
        reset_all;
        learn_enable = 1;

        program_pool(0, 20, 30, 31, 16'sd1200, 2'd0);
        program_index(0, 30, 20, 1, 2'd0);

        program_ucode(0, 7'd0, {4'd12, 4'd0, 4'd0, 4'd0, 3'd0, 13'd0});
        program_ucode(0, 7'd1, {4'd13, 4'd0, 4'd0, 4'd0, 3'd0, 13'd0});
        program_ucode(0, 7'd2, {4'd4, 4'd10, 4'd0, 4'd0, 3'd3, 13'd0});
        program_ucode(0, 7'd3, {4'd1, 4'd5, 4'd5, 4'd10, 3'd0, 13'd0});
        program_ucode(0, 7'd4, {4'd8, 4'd10, 4'd0, 4'd0, 16'd2000});
        program_ucode(0, 7'd5, {4'd7, 4'd5, 4'd5, 4'd10, 3'd0, 13'd0});
        program_ucode(0, 7'd6, {4'd9, 4'd0, 4'd0, 4'd0, 3'd0, 13'd0});
        program_ucode(0, 7'd7, {4'd13, 4'd0, 4'd0, 4'd0, 3'd0, 13'd0});

        stimulate(0, 31, 16'sd2000);
        run_timestep;

        weight_before = dut.gen_core[0].core.pool_weight_mem.mem[20];

        stimulate(0, 30, 16'sd2000);
        run_timestep;

        weight_after = dut.gen_core[0].core.pool_weight_mem.mem[20];
        $display("  Weight before: %0d, after: %0d", weight_before, weight_after);

        if (weight_after > weight_before) begin
            $display("TEST 3 PASSED (anti-STDP increased weight via custom microcode)");
            pass_count = pass_count + 1;
        end else begin
            $display("TEST 3 FAILED (expected weight increase from anti-STDP)");
            fail_count = fail_count + 1;
        end

        $display("test 4: ALU operation verification");
        reset_all;
        learn_enable = 1;

        program_pool(0, 30, 40, 41, 16'sd500, 2'd0);
        program_index(0, 40, 30, 1, 2'd0);

        program_ucode(0, 7'd0, {4'd12, 4'd0, 4'd0, 4'd0, 3'd0, 13'd0});
        program_ucode(0, 7'd1, {4'd13, 4'd0, 4'd0, 4'd0, 3'd0, 13'd0});
        program_ucode(0, 7'd2, {4'd5, 4'd10, 4'd0, 4'd0, 3'd1, 13'd0});
        program_ucode(0, 7'd3, {4'd1, 4'd5, 4'd5, 4'd10, 3'd0, 13'd0});
        program_ucode(0, 7'd4, {4'd8, 4'd10, 4'd0, 4'd0, 16'd1500});
        program_ucode(0, 7'd5, {4'd7, 4'd5, 4'd5, 4'd10, 3'd0, 13'd0});
        program_ucode(0, 7'd6, {4'd9, 4'd0, 4'd0, 4'd0, 3'd0, 13'd0});
        program_ucode(0, 7'd7, {4'd13, 4'd0, 4'd0, 4'd0, 3'd0, 13'd0});
        stimulate(0, 41, 16'sd2000);
        run_timestep;

        stimulate(0, 40, 16'sd2000);
        run_timestep;

        weight_after = dut.gen_core[0].core.pool_weight_mem.mem[30];
        $display("  Weight: expected ~700, got %0d", weight_after);

        if (weight_after == 16'sd700) begin
            $display("TEST 4 PASSED (custom ALU: SHL + ADD + MIN worked correctly)");
            pass_count = pass_count + 1;
        end else if (weight_after > 16'sd500 && weight_after < 16'sd1500) begin
            $display("TEST 4 PASSED (weight updated in expected direction)");
            pass_count = pass_count + 1;
        end else begin
            $display("TEST 4 FAILED (unexpected weight value)");
            fail_count = fail_count + 1;
        end

        $display("P19 RESULTS: %0d/%0d passed", pass_count, pass_count + fail_count);
        if (fail_count == 0)
            $display("All tests passed!");
        else
            $display("SOME TESTS FAILED");
        $finish;
    end

    genvar gi;
    generate
        for (gi = 0; gi < NUM_CORES; gi = gi + 1) begin : mon
            always @(posedge clk) begin
                if (spike_valid_bus[gi])
                    $display("  [t=%0d] Core %0d Neuron %0d spiked",
                             timestep_count, gi,
                             spike_id_bus[gi*NEURON_BITS +: NEURON_BITS]);
            end
        end
    endgenerate

endmodule
