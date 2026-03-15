// ============================================================================
// Testbench: Graded Spikes (Phase 8)
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

`timescale 1ns / 1ps

module tb_graded;

    parameter NUM_NEURONS   = 256;
    parameter NEURON_BITS   = 8;
    parameter DATA_WIDTH    = 16;
    parameter MAX_FANOUT    = 32;
    parameter FANOUT_BITS   = 5;
    parameter CONN_ADDR_BITS = 13;
    parameter CLK_PERIOD    = 10;
    parameter GRADE_SHIFT   = 7;

    reg                    clk;
    reg                    rst_n;
    reg                    start;
    reg                    learn_enable;
    reg                    graded_enable;
    reg                    ext_valid;
    reg  [NEURON_BITS-1:0] ext_neuron_id;
    reg  signed [DATA_WIDTH-1:0] ext_current;
    reg                    conn_we;
    reg  [NEURON_BITS-1:0] conn_src;
    reg  [FANOUT_BITS-1:0] conn_slot;
    reg  [NEURON_BITS-1:0] conn_target;
    reg  signed [DATA_WIDTH-1:0] conn_weight;

    wire                   timestep_done;
    wire                   spike_out_valid;
    wire [NEURON_BITS-1:0] spike_out_id;
    wire [7:0]             spike_out_payload;
    wire [4:0]             state_out;
    wire [31:0]            total_spikes;
    wire [31:0]            timestep_count;

    scalable_core_v2 #(
        .NUM_NEURONS   (NUM_NEURONS),
        .NEURON_BITS   (NEURON_BITS),
        .DATA_WIDTH    (DATA_WIDTH),
        .MAX_FANOUT    (MAX_FANOUT),
        .FANOUT_BITS   (FANOUT_BITS),
        .CONN_ADDR_BITS(CONN_ADDR_BITS),
        .THRESHOLD     (16'sd1000),
        .LEAK_RATE     (16'sd3),
        .RESTING_POT   (16'sd0),
        .REFRAC_CYCLES (2),
        .TRACE_MAX     (8'd100),
        .TRACE_DECAY   (8'd10),
        .LEARN_SHIFT   (3),
        .GRADE_SHIFT   (GRADE_SHIFT),
        .WEIGHT_MAX    (16'sd2000),
        .WEIGHT_MIN    (16'sd0)
    ) dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .start          (start),
        .learn_enable   (learn_enable),
        .graded_enable  (graded_enable),
        .dendritic_enable(1'b0),
        .ext_valid      (ext_valid),
        .ext_neuron_id  (ext_neuron_id),
        .ext_current    (ext_current),
        .conn_we        (conn_we),
        .conn_src       (conn_src),
        .conn_slot      (conn_slot),
        .conn_target    (conn_target),
        .conn_weight    (conn_weight),
        .conn_comp      (2'd0),
        .prog_param_we  (1'b0),
        .prog_param_neuron(8'd0),
        .prog_param_id  (3'd0),
        .prog_param_value(16'sd0),
        .timestep_done  (timestep_done),
        .spike_out_valid(spike_out_valid),
        .spike_out_id   (spike_out_id),
        .spike_out_payload(spike_out_payload),
        .state_out      (state_out),
        .total_spikes   (total_spikes),
        .timestep_count (timestep_count)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    task program_conn;
        input [NEURON_BITS-1:0] src;
        input [FANOUT_BITS-1:0] slot;
        input [NEURON_BITS-1:0] target;
        input signed [DATA_WIDTH-1:0] weight;
    begin
        @(posedge clk);
        conn_we     <= 1;
        conn_src    <= src;
        conn_slot   <= slot;
        conn_target <= target;
        conn_weight <= weight;
        @(posedge clk);
        conn_we <= 0;
        @(posedge clk);
    end
    endtask

    task stimulate;
        input [NEURON_BITS-1:0] neuron;
        input signed [DATA_WIDTH-1:0] current;
    begin
        @(posedge clk);
        ext_valid     <= 1;
        ext_neuron_id <= neuron;
        ext_current   <= current;
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
        wait(timestep_done);
        @(posedge clk);
    end
    endtask

    // Read membrane potential
    function signed [DATA_WIDTH-1:0] read_potential;
        input [NEURON_BITS-1:0] neuron;
    begin
        read_potential = dut.neuron_mem.mem[neuron];
    end
    endfunction

    function signed [DATA_WIDTH-1:0] read_weight;
        input [NEURON_BITS-1:0] src;
        input [FANOUT_BITS-1:0] slot;
        reg [CONN_ADDR_BITS-1:0] addr;
    begin
        addr = {src, slot};
        read_weight = dut.weight_mem.mem[addr];
    end
    endfunction

    reg [7:0] last_payload;
    reg [7:0] last_spike_id;
    integer spike_count;

    always @(posedge clk) begin
        if (spike_out_valid) begin
            last_payload = spike_out_payload;
            last_spike_id = spike_out_id;
            spike_count = spike_count + 1;
        end
    end

    integer pass_count, fail_count;
    reg signed [DATA_WIDTH-1:0] pot_val, pot_binary, pot_graded;
    reg signed [31:0] expected32;
    reg signed [DATA_WIDTH-1:0] expected;

    initial begin
        rst_n         = 0;
        start         = 0;
        learn_enable  = 0;
        graded_enable = 0;
        ext_valid     = 0;
        conn_we       = 0;
        conn_src      = 0;
        conn_slot     = 0;
        conn_target   = 0;
        conn_weight   = 0;
        ext_neuron_id = 0;
        ext_current   = 0;
        spike_count   = 0;
        pass_count    = 0;
        fail_count    = 0;

        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 3);

        $display("");
        $display("================================================================");
        $display("  Graded Spikes Test (Phase 8)");
        $display("================================================================");

        // TEST 1: Binary mode (graded_enable=0)
        //   Neurons: N0 -> N2 (weight=500)
        $display("");
        $display("--- TEST 1: Binary Mode (graded_enable=0) ---");

        graded_enable = 0;
        learn_enable  = 0;
        program_conn(8'd0, 5'd0, 8'd2, 16'sd500);

        // N0 spikes: excess = 0+1200-3-1000 = 197, payload=197
        stimulate(8'd0, 16'sd1200);
        spike_count = 0;
        run_timestep;  // TS1: N0 spikes
        $display("  TS1: N0 spiked, payload=%0d, spikes=%0d", last_payload, spike_count);

        run_timestep;  // TS2: deliver N0->N2 with binary weight=500
        // N2: 0 + 500 - 3(leak) = 497
        pot_binary = read_potential(8'd2);
        $display("  N2 potential (binary) = %0d (expected 497)", pot_binary);

        if (pot_binary == 16'sd497) begin
            $display("  PASS: Binary delivery correct");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected 497, got %0d", pot_binary);
            fail_count = fail_count + 1;
        end

        // TEST 2: Graded mode - payload and delivery
        //   Neurons: N10 -> N12 (weight=500)
        //   No reset - fresh neurons, no stale SRAM state
        $display("");
        $display("--- TEST 2: Graded Mode (graded_enable=1) ---");

        graded_enable = 1;
        program_conn(8'd10, 5'd0, 8'd12, 16'sd500);

        // N10 spikes: excess = 0+1200-3-1000 = 197, payload=197
        stimulate(8'd10, 16'sd1200);
        spike_count = 0;
        run_timestep;  // TS3: N10 spikes

        $display("  TS3: spike_id=%0d, payload=%0d, spikes=%0d",
            last_spike_id, last_payload, spike_count);

        if (last_payload == 8'd197) begin
            $display("  PASS: Payload = 197");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected payload=197, got %0d", last_payload);
            fail_count = fail_count + 1;
        end

        run_timestep;  // TS4: deliver N10->N12 with graded
        // Graded: (500 * 197) >> 7 = 98500 >> 7 = 769
        // N12 potential: 0 + 769 - 3 = 766
        expected32 = (32'sd500 * 32'sd197) >>> GRADE_SHIFT;
        expected32 = expected32 - 32'sd3;
        expected = expected32[DATA_WIDTH-1:0];
        pot_graded = read_potential(8'd12);
        $display("  N12 potential (graded) = %0d (expected %0d)", pot_graded, expected);

        if (pot_graded == expected) begin
            $display("  PASS: Graded delivery correct");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected %0d, got %0d", expected, pot_graded);
            fail_count = fail_count + 1;
        end

        // TEST 3: Payload clamping at 255
        //   Neurons: N20 -> N22 (weight=400)
        $display("");
        $display("--- TEST 3: Payload Clamping at 255 ---");

        graded_enable = 1;
        program_conn(8'd20, 5'd0, 8'd22, 16'sd400);

        // N20 spikes: excess = 0+2000-3-1000 = 997 > 255, clamp to 255
        stimulate(8'd20, 16'sd2000);
        spike_count = 0;
        run_timestep;  // TS5: N20 spikes
        $display("  TS5: spike_id=%0d, payload=%0d", last_spike_id, last_payload);

        if (last_payload == 8'd255) begin
            $display("  PASS: Payload clamped to 255");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected payload=255, got %0d", last_payload);
            fail_count = fail_count + 1;
        end

        run_timestep;  // TS6: deliver N20->N22 with graded
        // (400 * 255) >> 7 = 102000 >> 7 = 796
        // N22: 0 + 796 - 3 = 793
        expected32 = (32'sd400 * 32'sd255) >>> GRADE_SHIFT;
        expected32 = expected32 - 32'sd3;
        expected = expected32[DATA_WIDTH-1:0];
        pot_val = read_potential(8'd22);
        $display("  N22 potential = %0d (expected %0d)", pot_val, expected);

        if (pot_val == expected) begin
            $display("  PASS: Clamped graded delivery correct");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected %0d, got %0d", expected, pot_val);
            fail_count = fail_count + 1;
        end

        // TEST 4: Graded > Binary comparison
        //   Compare TEST 1 (N2, binary=497) vs TEST 2 (N12, graded=766)
        //   Since payload=197 > 128 (unity), graded should deliver MORE
        $display("");
        $display("--- TEST 4: Graded > Binary Comparison ---");
        $display("  Binary N2 potential  = %0d", pot_binary);
        $display("  Graded N12 potential = %0d", pot_graded);

        if (pot_graded > pot_binary) begin
            $display("  PASS: Graded (payload=197>128) delivered more than binary (%0d > %0d)",
                pot_graded, pot_binary);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected graded > binary (%0d <= %0d)", pot_graded, pot_binary);
            fail_count = fail_count + 1;
        end

        // TEST 5: Graded + STDP coexistence
        //   Neurons: N30 -> N31, N31 -> N32
        //   Pre-before-post -> LTP should occur even with graded enabled
        $display("");
        $display("--- TEST 5: Graded + STDP Together ---");

        graded_enable = 1;
        learn_enable  = 1;

        program_conn(8'd30, 5'd0, 8'd31, 16'sd500);
        program_conn(8'd31, 5'd0, 8'd32, 16'sd100);

        stimulate(8'd30, 16'sd1200);
        run_timestep;

        // Post fires (N30's trace still active -> LTP)
        stimulate(8'd31, 16'sd1200);
        run_timestep;

        begin : test5_block
            reg signed [DATA_WIDTH-1:0] w_after;
            w_after = read_weight(8'd30, 5'd0);
            $display("  Weight N30->N31 after pre->post: %0d (was 500)", w_after);

            if (w_after > 16'sd500) begin
                $display("  PASS: LTP occurred with graded+STDP (%0d > 500)", w_after);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL: Expected weight > 500, got %0d", w_after);
                fail_count = fail_count + 1;
            end
        end

        $display("");
        $display("================================================================");
        $display("  GRADED SPIKE TEST RESULTS: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("================================================================");
        if (fail_count == 0)
            $display("  ALL TESTS PASSED");
        else
            $display("  SOME TESTS FAILED");
        $display("================================================================");

        #(CLK_PERIOD * 10);
        $finish;
    end

    initial begin
        #(CLK_PERIOD * 5_000_000);
        $display("TIMEOUT");
        $finish;
    end

endmodule
