// ============================================================================
// P22C Testbench: Enhanced Learning Engine (ISA v2)
// ============================================================================
//
// Copyright 2026 Henry Arthur Shulayev Barnes / Catalyst Neuromorphic Ltd
// Company No. 17054540 — UK Patent Application No. 2602902.6
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================

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

        // TEST 1: 5-trace system with distinct tau values
        // Spike N10, all 5 traces → TRACE_MAX (100), then decay with
        // different tau: x1=3, x2=2, y1=4, y2=5, y3=1
        // Expected after 1 decay: x1=88, x2=75, y1=94, y2=97, y3=50
        $display("\n========================================");
        $display("TEST 1: 5-trace system readback");
        $display("========================================");
        reset_all;

        // Set tau values for N10 on core 0
        set_param(0, 10'd10, 5'd6,  16'd3);  // tau1 (x1) = 3
        set_param(0, 10'd10, 5'd7,  16'd4);  // tau2 (y1) = 4
        set_param(0, 10'd10, 5'd19, 16'd2);  // tau_x2 = 2
        set_param(0, 10'd10, 5'd20, 16'd5);  // tau_y2 = 5
        set_param(0, 10'd10, 5'd21, 16'd1);  // tau_y3 = 1

        // Spike N10 to set all traces to TRACE_MAX (100)
        run_timestep(0, 10'd10, 16'sd2000);

        // Verify all traces are 100 after spike
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

        // Run empty timestep to let traces decay
        run_empty;

        // Read back all 5 traces after one decay step
        begin
            reg [7:0] x1_val, x2_val, y1_val, y2_val, y3_val;
            x1_val = dut.gen_core[0].core.trace_mem.mem[10];
            x2_val = dut.gen_core[0].core.x2_trace_mem.mem[10];
            y1_val = dut.gen_core[0].core.trace2_mem.mem[10];
            y2_val = dut.gen_core[0].core.y2_trace_mem.mem[10];
            y3_val = dut.gen_core[0].core.y3_trace_mem.mem[10];
            $display("  After decay: x1=%0d x2=%0d y1=%0d y2=%0d y3=%0d",
                     x1_val, x2_val, y1_val, y2_val, y3_val);

            // Verify: each trace decays at its own rate
            // x1: tau=3, 100 - (100>>3) = 100 - 12 = 88
            // x2: tau=2, 100 - (100>>2) = 100 - 25 = 75
            // y1: tau=4, 100 - (100>>4) = 100 - 6  = 94
            // y2: tau=5, 100 - (100>>5) = 100 - 3  = 97
            // y3: tau=1, 100 - (100>>1) = 100 - 50 = 50
            if (x1_val == 8'd88 && x2_val == 8'd75 && y1_val == 8'd94 &&
                y2_val == 8'd97 && y3_val == 8'd50) begin
                $display("TEST 1 PASSED (all 5 traces decay correctly with distinct tau)");
                pass_count = pass_count + 1;
            end else begin
                $display("TEST 1 FAILED (expected x1=88 x2=75 y1=94 y2=97 y3=50)");
                fail_count = fail_count + 1;
            end
        end

        // TEST 2: Delay learning via STORE_D
        // Custom LTD microcode: LOADI R6, 10 → STORE_D → HALT
        // Verify pool_delay_mem changes from 5 to 10
        $display("\n========================================");
        $display("TEST 2: Delay learning (STORE_D)");
        $display("========================================");
        reset_all;
        learn_enable = 1;

        // Connection: N20→N21, weight=500, initial delay=5
        add_pool(0, 10'd0, 10'd20, 10'd21, 16'sd500);
        set_index(0, 10'd20, 10'd0, 10'd1);
        program_delay(0, 10'd0, 6'd5);

        // Custom LTD microcode (PC 0-4):
        // ISA v2: {op[3:0], dst[3:0], src_a[3:0], src_b[3:0], shift[2:0], imm[12:0]}
        // R0=x1(trace), R6=delay, R10=temp
        program_ucode(0, 7'd0, {4'd12, 4'd0,  4'd0, 4'd0, 3'd0, 13'd0});  // SKIP_NZ R0
        program_ucode(0, 7'd1, {4'd13, 4'd0,  4'd0, 4'd0, 3'd0, 13'd0});  // HALT
        program_ucode(0, 7'd2, {4'd8,  4'd6,  4'd0, 4'd0, 16'd10});       // LOADI R6, 10
        program_ucode(0, 7'd3, {4'd14, 4'd0,  4'd0, 4'd0, 3'd0, 13'd0}); // STORE_D
        program_ucode(0, 7'd4, {4'd13, 4'd0,  4'd0, 4'd0, 3'd0, 13'd0}); // HALT

        // Override LTP to do nothing (prevent default weight modification)
        program_ucode(0, 7'd16, {4'd13, 4'd0, 4'd0, 4'd0, 3'd0, 13'd0}); // HALT immediately

        // Verify initial delay
        begin
            reg [5:0] delay_before;
            delay_before = dut.gen_core[0].core.pool_delay_mem.mem[0];
            $display("  Delay before: %0d", delay_before);
        end

        // Spike N21 first (build post trace for R0 in LTD)
        run_timestep(0, 10'd21, 16'sd2000);

        // Spike N20 (pre neuron) → LTD runs custom code
        run_timestep(0, 10'd20, 16'sd2000);

        // Verify delay changed
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

        // TEST 3: Tag learning via STORE_T
        // Custom LTD: R7 = R5 (weight) + R0 (trace) → STORE_T
        // Verify pool_tag_mem gets weight+trace value
        $display("\n========================================");
        $display("TEST 3: Tag learning (STORE_T)");
        $display("========================================");
        reset_all;
        learn_enable = 1;

        // Connection: N30→N31, weight=600
        add_pool(0, 10'd0, 10'd30, 10'd31, 16'sd600);
        set_index(0, 10'd30, 10'd0, 10'd1);

        // Custom LTD microcode: tag = weight + trace
        // R0=x1(trace), R5=weight, R7=tag, R10=temp
        program_ucode(0, 7'd0, {4'd12, 4'd0,  4'd0, 4'd0,  3'd0, 13'd0});  // SKIP_NZ R0
        program_ucode(0, 7'd1, {4'd13, 4'd0,  4'd0, 4'd0,  3'd0, 13'd0});  // HALT
        program_ucode(0, 7'd2, {4'd1,  4'd7,  4'd5, 4'd0,  3'd0, 13'd0});  // ADD R7, R5, R0
        program_ucode(0, 7'd3, {4'd15, 4'd0,  4'd0, 4'd0,  3'd0, 13'd0}); // STORE_T
        program_ucode(0, 7'd4, {4'd13, 4'd0,  4'd0, 4'd0,  3'd0, 13'd0}); // HALT

        // Override LTP to do nothing
        program_ucode(0, 7'd16, {4'd13, 4'd0, 4'd0, 4'd0, 3'd0, 13'd0}); // HALT

        // Verify initial tag
        begin
            reg signed [DATA_WIDTH-1:0] tag_before;
            tag_before = dut.gen_core[0].core.pool_tag_mem.mem[0];
            $display("  Tag before: %0d", tag_before);
        end

        // Spike N31 first (build post trace)
        run_timestep(0, 10'd31, 16'sd2000);

        // Spike N30 (pre) → LTD: R0=trace of N31=100, R5=weight=600, R7=600+100=700
        run_timestep(0, 10'd30, 16'sd2000);

        // Verify tag changed
        begin
            reg signed [DATA_WIDTH-1:0] tag_after;
            tag_after = dut.gen_core[0].core.pool_tag_mem.mem[0];
            $display("  Tag after: %0d (expected ~700)", tag_after);
            // trace1 of N31 = 100 after spike, may have decayed by 1 timestep
            // In LTD, trace_addr=pool_tgt=N31, R0=trace_mem[N31]
            // After spike timestep, trace is TRACE_MAX=100
            // Next timestep (N30 spike), decay applied first: 100 - (100>>tau1_default=3) = 88
            // So R0 = 88, R5 = 600, tag = 600 + 88 = 688
            if (tag_after >= 16'sd680 && tag_after <= 16'sd710) begin
                $display("TEST 3 PASSED (STORE_T wrote tag = weight + trace)");
                pass_count = pass_count + 1;
            end else begin
                $display("TEST 3 FAILED (expected tag ~688-700, got %0d)", tag_after);
                fail_count = fail_count + 1;
            end
        end

        // TEST 4: Stochastic rounding
        // Custom LTD: just STORE_W (no delta, stores R5 + lfsr[0])
        // Run 20 times, weight should drift upward from 500
        $display("\n========================================");
        $display("TEST 4: Stochastic rounding drift");
        $display("========================================");
        reset_all;
        learn_enable = 1;

        // Connection: N40→N41, weight=500
        add_pool(0, 10'd0, 10'd40, 10'd41, 16'sd500);
        set_index(0, 10'd40, 10'd0, 10'd1);

        // Custom LTD: just store weight (no computation) — lfsr[0] adds 0 or 1
        program_ucode(0, 7'd0, {4'd12, 4'd0,  4'd0, 4'd0,  3'd0, 13'd0});  // SKIP_NZ R0
        program_ucode(0, 7'd1, {4'd13, 4'd0,  4'd0, 4'd0,  3'd0, 13'd0});  // HALT
        program_ucode(0, 7'd2, {4'd9,  4'd0,  4'd0, 4'd0,  3'd0, 13'd0});  // STORE_W
        program_ucode(0, 7'd3, {4'd13, 4'd0,  4'd0, 4'd0,  3'd0, 13'd0});  // HALT

        // Override LTP to do nothing
        program_ucode(0, 7'd16, {4'd13, 4'd0, 4'd0, 4'd0, 3'd0, 13'd0}); // HALT

        // Spike N41 once to build post trace
        run_timestep(0, 10'd41, 16'sd2000);

        // Run 20 rounds: spike N40 each time → LTD → STORE_W with stochastic rounding
        for (i = 0; i < 20; i = i + 1) begin
            run_timestep(0, 10'd40, 16'sd2000);
        end

        // Check weight drift
        begin
            reg signed [DATA_WIDTH-1:0] weight_final;
            weight_final = dut.gen_core[0].core.pool_weight_mem.mem[0];
            $display("  Weight after 20 rounds: %0d (started at 500)", weight_final);
            // Each round adds 0 or 1 (LFSR-dependent). After 20 rounds, expect ~510 ± 5.
            // Statistical test: weight > 500 (extremely unlikely all 20 rounds add 0)
            // and weight <= 520 (can't add more than 20)
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

        $display("\n========================================");
        $display("P22C RESULTS: %0d/%0d passed", pass_count, pass_count + fail_count);
        $display("========================================");
        if (fail_count == 0)
            $display("All tests passed!");
        else
            $display("SOME TESTS FAILED!");
        $finish;
    end

endmodule
