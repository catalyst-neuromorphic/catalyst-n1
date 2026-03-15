`timescale 1ns/1ps

module tb_p22c_learning;

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

    reg                         prog_delay_we;
    reg  [CORE_ID_BITS-1:0]    prog_delay_core;
    reg  [POOL_ADDR_BITS-1:0]  prog_delay_addr;
    reg  [5:0]                  prog_delay_value;

    reg                         prog_ucode_we;
    reg  [CORE_ID_BITS-1:0]    prog_ucode_core;
    reg  [6:0]                  prog_ucode_addr;
    reg  [31:0]                 prog_ucode_data;

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
        .prog_delay_we     (prog_delay_we),
        .prog_delay_core   (prog_delay_core),
        .prog_delay_addr   (prog_delay_addr),
        .prog_delay_value  (prog_delay_value),
        .prog_ucode_we     (prog_ucode_we),
        .prog_ucode_core   (prog_ucode_core),
        .prog_ucode_addr   (prog_ucode_addr),
        .prog_ucode_data   (prog_ucode_data),
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

    always @(posedge clk) begin : spike_monitor
        integer c;
        for (c = 0; c < NUM_CORES; c = c + 1) begin
            if (spike_valid_bus[c]) begin
                $display("  [t=%0d] Core %0d Neuron %0d spiked",
                    timestep_count, c, spike_id_bus[c*NEURON_BITS +: NEURON_BITS]);
            end
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
        prog_delay_we = 0; prog_delay_core = 0; prog_delay_addr = 0; prog_delay_value = 0;
        prog_ucode_we = 0; prog_ucode_core = 0; prog_ucode_addr = 0; prog_ucode_data = 0;
        ext_valid = 0; ext_core = 0; ext_neuron_id = 0; ext_current = 0;
        probe_read = 0; probe_core = 0; probe_neuron = 0; probe_state_id = 0; probe_pool_addr = 0;
        #100;
        rst_n = 1;
        #20;
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

    task program_ucode;
        input [CORE_ID_BITS-1:0] core;
        input [6:0]               addr;
        input [31:0]              instr;
    begin
        @(posedge clk);
        prog_ucode_we   <= 1;
        prog_ucode_core <= core;
        prog_ucode_addr <= addr;
        prog_ucode_data <= instr;
        @(posedge clk);
        prog_ucode_we <= 0;
    end
    endtask

    task program_delay;
        input [CORE_ID_BITS-1:0]   core;
        input [POOL_ADDR_BITS-1:0] addr;
        input [5:0]                value;
    begin
        @(posedge clk);
        prog_delay_we    <= 1;
        prog_delay_core  <= core;
        prog_delay_addr  <= addr;
        prog_delay_value <= value;
        @(posedge clk);
        prog_delay_we <= 0;
    end
    endtask

    task stimulate;
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
    integer i;
    reg [7:0] trace_val;
    reg signed [DATA_WIDTH-1:0] weight_val;

    initial begin
        pass_count = 0;
        fail_count = 0;

        $display("test 1: 5-trace system readback");
        reset_all;

        set_param(0, 10'd10, 5'd6,  16'd3);
        set_param(0, 10'd10, 5'd7,  16'd4);
        set_param(0, 10'd10, 5'd19, 16'd2);
        set_param(0, 10'd10, 5'd20, 16'd5);
        set_param(0, 10'd10, 5'd21, 16'd1);

        run_timestep(0, 10'd10, 16'sd2000);

        begin
            reg [7:0] x1_val, x2_val, y1_val, y2_val, y3_val;
            x1_val = dut.gen_core[0].core.trace_mem.mem[10];
            x2_val = dut.gen_core[0].core.x2_trace_mem.mem[10];
            y1_val = dut.gen_core[0].core.trace2_mem.mem[10];
            y2_val = dut.gen_core[0].core.y2_trace_mem.mem[10];
            y3_val = dut.gen_core[0].core.y3_trace_mem.mem[10];
            $display("  After spike: x1=%0d x2=%0d y1=%0d y2=%0d y3=%0d",
                     x1_val, x2_val, y1_val, y2_val, y3_val);
        end

        run_empty;

        begin
            reg [7:0] x1_val, x2_val, y1_val, y2_val, y3_val;
            x1_val = dut.gen_core[0].core.trace_mem.mem[10];
            x2_val = dut.gen_core[0].core.x2_trace_mem.mem[10];
            y1_val = dut.gen_core[0].core.trace2_mem.mem[10];
            y2_val = dut.gen_core[0].core.y2_trace_mem.mem[10];
            y3_val = dut.gen_core[0].core.y3_trace_mem.mem[10];
            $display("  After decay: x1=%0d x2=%0d y1=%0d y2=%0d y3=%0d",
                     x1_val, x2_val, y1_val, y2_val, y3_val);

            if (x1_val == 8'd88 && x2_val == 8'd75 && y1_val == 8'd94 &&
                y2_val == 8'd97 && y3_val == 8'd50) begin
                $display("TEST 1 PASSED (all 5 traces decay correctly with distinct tau)");
                pass_count = pass_count + 1;
            end else begin
                $display("TEST 1 FAILED (expected x1=88 x2=75 y1=94 y2=97 y3=50)");
                fail_count = fail_count + 1;
            end
        end

        $display("test 2: Delay learning (STORE_D)");
        reset_all;
        learn_enable = 1;

        add_pool(0, 10'd0, 10'd20, 10'd21, 16'sd500);
        set_index(0, 10'd20, 10'd0, 10'd1);
        program_delay(0, 10'd0, 6'd5);

        program_ucode(0, 7'd0, {4'd12, 4'd0,  4'd0, 4'd0, 3'd0, 13'd0});
        program_ucode(0, 7'd1, {4'd13, 4'd0,  4'd0, 4'd0, 3'd0, 13'd0});
        program_ucode(0, 7'd2, {4'd8,  4'd6,  4'd0, 4'd0, 16'd10});
        program_ucode(0, 7'd3, {4'd14, 4'd0,  4'd0, 4'd0, 3'd0, 13'd0});
        program_ucode(0, 7'd4, {4'd13, 4'd0,  4'd0, 4'd0, 3'd0, 13'd0});

        program_ucode(0, 7'd16, {4'd13, 4'd0, 4'd0, 4'd0, 3'd0, 13'd0});

        begin
            reg [5:0] delay_before;
            delay_before = dut.gen_core[0].core.pool_delay_mem.mem[0];
            $display("  Delay before: %0d", delay_before);
        end

        run_timestep(0, 10'd21, 16'sd2000);

        run_timestep(0, 10'd20, 16'sd2000);

        begin
            reg [5:0] delay_after;
            delay_after = dut.gen_core[0].core.pool_delay_mem.mem[0];
            $display("  Delay after: %0d (expected 10)", delay_after);
            if (delay_after == 6'd10) begin
                $display("TEST 2 PASSED (STORE_D changed delay from 5 to 10)");
                pass_count = pass_count + 1;
            end else begin
                $display("TEST 2 FAILED (expected delay=10, got %0d)", delay_after);
                fail_count = fail_count + 1;
            end
        end

        $display("test 3: Tag learning (STORE_T)");
        reset_all;
        learn_enable = 1;

        add_pool(0, 10'd0, 10'd30, 10'd31, 16'sd600);
        set_index(0, 10'd30, 10'd0, 10'd1);

        program_ucode(0, 7'd0, {4'd12, 4'd0,  4'd0, 4'd0,  3'd0, 13'd0});
        program_ucode(0, 7'd1, {4'd13, 4'd0,  4'd0, 4'd0,  3'd0, 13'd0});
        program_ucode(0, 7'd2, {4'd1,  4'd7,  4'd5, 4'd0,  3'd0, 13'd0});
        program_ucode(0, 7'd3, {4'd15, 4'd0,  4'd0, 4'd0,  3'd0, 13'd0});
        program_ucode(0, 7'd4, {4'd13, 4'd0,  4'd0, 4'd0,  3'd0, 13'd0});

        program_ucode(0, 7'd16, {4'd13, 4'd0, 4'd0, 4'd0, 3'd0, 13'd0});

        begin
            reg signed [DATA_WIDTH-1:0] tag_before;
            tag_before = dut.gen_core[0].core.pool_tag_mem.mem[0];
            $display("  Tag before: %0d", tag_before);
        end

        run_timestep(0, 10'd31, 16'sd2000);

        run_timestep(0, 10'd30, 16'sd2000);

        begin
            reg signed [DATA_WIDTH-1:0] tag_after;
            tag_after = dut.gen_core[0].core.pool_tag_mem.mem[0];
            $display("  Tag after: %0d (expected ~700)", tag_after);
            if (tag_after >= 16'sd680 && tag_after <= 16'sd710) begin
                $display("TEST 3 PASSED (STORE_T wrote tag = weight + trace)");
                pass_count = pass_count + 1;
            end else begin
                $display("TEST 3 FAILED (expected tag ~688-700, got %0d)", tag_after);
                fail_count = fail_count + 1;
            end
        end

        $display("test 4: Stochastic rounding drift");
        reset_all;
        learn_enable = 1;

        add_pool(0, 10'd0, 10'd40, 10'd41, 16'sd500);
        set_index(0, 10'd40, 10'd0, 10'd1);

        program_ucode(0, 7'd0, {4'd12, 4'd0,  4'd0, 4'd0,  3'd0, 13'd0});
        program_ucode(0, 7'd1, {4'd13, 4'd0,  4'd0, 4'd0,  3'd0, 13'd0});
        program_ucode(0, 7'd2, {4'd9,  4'd0,  4'd0, 4'd0,  3'd0, 13'd0});
        program_ucode(0, 7'd3, {4'd13, 4'd0,  4'd0, 4'd0,  3'd0, 13'd0});

        program_ucode(0, 7'd16, {4'd13, 4'd0, 4'd0, 4'd0, 3'd0, 13'd0});

        run_timestep(0, 10'd41, 16'sd2000);

        for (i = 0; i < 20; i = i + 1) begin
            run_timestep(0, 10'd40, 16'sd2000);
        end

        begin
            reg signed [DATA_WIDTH-1:0] weight_final;
            weight_final = dut.gen_core[0].core.pool_weight_mem.mem[0];
            $display("  Weight after 20 rounds: %0d (started at 500)", weight_final);
            if (weight_final > 16'sd500 && weight_final <= 16'sd520) begin
                $display("TEST 4 PASSED (stochastic rounding drifted weight to %0d)", weight_final);
                pass_count = pass_count + 1;
            end else if (weight_final == 16'sd500) begin
                $display("TEST 4 FAILED (no drift — stochastic rounding not working)");
                fail_count = fail_count + 1;
            end else begin
                $display("TEST 4 FAILED (unexpected weight %0d)", weight_final);
                fail_count = fail_count + 1;
            end
        end

        $display("P22C RESULTS: %0d/%0d passed", pass_count, pass_count + fail_count);
        if (fail_count == 0)
            $display("All tests passed!");
        else
            $display("SOME TESTS FAILED");
        $finish;
    end

endmodule
