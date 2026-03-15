// ============================================================================
// Testbench: STDP On-Chip Learning (Phase 7)
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

module tb_stdp;

    parameter NUM_NEURONS   = 256;
    parameter NEURON_BITS   = 8;
    parameter DATA_WIDTH    = 16;
    parameter MAX_FANOUT    = 32;
    parameter FANOUT_BITS   = 5;
    parameter CONN_ADDR_BITS = 13;
    parameter CLK_PERIOD    = 10;

    reg                    clk;
    reg                    rst_n;
    reg                    start;
    reg                    learn_enable;
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
        .WEIGHT_MAX    (16'sd2000),
        .WEIGHT_MIN    (16'sd0)
    ) dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .start          (start),
        .learn_enable   (learn_enable),
        .graded_enable  (1'b0),
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
        .spike_out_payload(),
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
        @(posedge clk); // extra cycle for reverse index to settle
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

    // Read weight from internal SRAM (hierarchical access for debug)
    function signed [DATA_WIDTH-1:0] read_weight;
        input [NEURON_BITS-1:0] src;
        input [FANOUT_BITS-1:0] slot;
        reg [CONN_ADDR_BITS-1:0] addr;
    begin
        addr = {src, slot};
        read_weight = dut.weight_mem.mem[addr];
    end
    endfunction

    reg [7:0] spike_log [0:255];
    integer spike_count;

    always @(posedge clk) begin
        if (spike_out_valid && spike_count < 256) begin
            spike_log[spike_count] = spike_out_id;
            spike_count = spike_count + 1;
        end
    end

    reg signed [DATA_WIDTH-1:0] w_before, w_after;
    integer i;
    integer pass_count, fail_count;

    initial begin
        rst_n         = 0;
        start         = 0;
        learn_enable  = 0;
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
        $display("  STDP On-Chip Learning Test (Phase 7)");
        $display("================================================================");

        // Setup: N0 → N1 (weight=500). Stimulate N0 to spike first,
        // then N1 spikes next timestep. N0's trace is still active
        // when N1 fires → LTP on the N0→N1 synapse.
        $display("");
        $display("--- TEST 1: Pre-before-Post → LTP ---");

        // Program: N0 → N1 with initial weight 500
        program_conn(8'd0, 5'd0, 8'd1, 16'sd500);
        // Program: N1 → N2 (dummy, so N1 spike has somewhere to go)
        program_conn(8'd1, 5'd0, 8'd2, 16'sd100);

        learn_enable = 1;

        // Timestep 1: Make N0 spike (strong stimulus)
        stimulate(8'd0, 16'sd1200);
        spike_count = 0;
        run_timestep;
        $display("  TS1: N0 stimulated with 1200, spikes=%0d", spike_count);

        w_before = read_weight(8'd0, 5'd0);
        $display("  Weight N0→N1 before LTP: %0d", w_before);

        // Timestep 2: Make N1 spike (N0's trace still active → LTP)
        stimulate(8'd1, 16'sd1200);
        spike_count = 0;
        run_timestep;
        $display("  TS2: N1 stimulated with 1200, spikes=%0d", spike_count);

        w_after = read_weight(8'd0, 5'd0);
        $display("  Weight N0→N1 after LTP:  %0d", w_after);

        if (w_after > w_before) begin
            $display("  PASS: Weight increased (%0d → %0d, +%0d)",
                w_before, w_after, w_after - w_before);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Weight did not increase (%0d → %0d)",
                w_before, w_after);
            fail_count = fail_count + 1;
        end

        // Setup: N10 → N11 (weight=500). Make N11 spike first,
        // then N10 spikes. N11's trace active when N10 fires → LTD.
        $display("");
        $display("--- TEST 2: Post-before-Pre → LTD ---");

        rst_n = 0;
        #(CLK_PERIOD * 3);
        rst_n = 1;
        #(CLK_PERIOD * 3);
        learn_enable = 1;

        // Program: N10 → N11 with initial weight 500
        program_conn(8'd10, 5'd0, 8'd11, 16'sd500);

        // Timestep 1: Make N11 (post) spike FIRST
        stimulate(8'd11, 16'sd1200);
        spike_count = 0;
        run_timestep;
        $display("  TS1: N11 (post) spiked first, spikes=%0d", spike_count);

        w_before = read_weight(8'd10, 5'd0);
        $display("  Weight N10→N11 before LTD: %0d", w_before);

        // Timestep 2: Make N10 (pre) spike — N11's trace still active → LTD
        stimulate(8'd10, 16'sd1200);
        spike_count = 0;
        run_timestep;
        $display("  TS2: N10 (pre) spiked second, spikes=%0d", spike_count);

        w_after = read_weight(8'd10, 5'd0);
        $display("  Weight N10→N11 after LTD:  %0d", w_after);

        if (w_after < w_before) begin
            $display("  PASS: Weight decreased (%0d → %0d, -%0d)",
                w_before, w_after, w_before - w_after);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Weight did not decrease (%0d → %0d)",
                w_before, w_after);
            fail_count = fail_count + 1;
        end

        // N20 → N21 with weight 500. Only N20 fires, N21 never fires.
        // No post trace → no LTD. No post spike → no LTP. Weight stable.
        $display("");
        $display("--- TEST 3: Uncorrelated → No Change ---");

        rst_n = 0;
        #(CLK_PERIOD * 3);
        rst_n = 1;
        #(CLK_PERIOD * 3);
        learn_enable = 1;

        program_conn(8'd20, 5'd0, 8'd21, 16'sd500);

        w_before = read_weight(8'd20, 5'd0);

        // Run 5 timesteps with only N20 spiking (N21 never reaches threshold)
        for (i = 0; i < 5; i = i + 1) begin
            stimulate(8'd20, 16'sd1200);
            run_timestep;
        end

        w_after = read_weight(8'd20, 5'd0);
        $display("  Weight N20→N21: %0d → %0d", w_before, w_after);

        if (w_after == w_before) begin
            $display("  PASS: Weight unchanged (no correlated post activity)");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Weight changed unexpectedly (%0d → %0d)",
                w_before, w_after);
            fail_count = fail_count + 1;
        end

        // Same as TEST 1 setup but with learn_enable=0.
        // Weight should NOT change.
        $display("");
        $display("--- TEST 4: Learning Disabled → No Change ---");

        rst_n = 0;
        #(CLK_PERIOD * 3);
        rst_n = 1;
        #(CLK_PERIOD * 3);
        learn_enable = 0;  // DISABLED

        program_conn(8'd0, 5'd0, 8'd1, 16'sd500);

        // Pre-before-post pattern (same as TEST 1)
        stimulate(8'd0, 16'sd1200);
        run_timestep;

        w_before = read_weight(8'd0, 5'd0);

        stimulate(8'd1, 16'sd1200);
        run_timestep;

        w_after = read_weight(8'd0, 5'd0);
        $display("  Weight N0→N1: %0d → %0d (learn_enable=0)", w_before, w_after);

        if (w_after == w_before) begin
            $display("  PASS: Weight unchanged with learning disabled");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Weight changed despite learning disabled");
            fail_count = fail_count + 1;
        end

        $display("");
        $display("--- TEST 5: Repeated Pre→Post Strengthens Over Time ---");

        rst_n = 0;
        #(CLK_PERIOD * 3);
        rst_n = 1;
        #(CLK_PERIOD * 3);
        learn_enable = 1;

        program_conn(8'd0, 5'd0, 8'd1, 16'sd200);

        w_before = read_weight(8'd0, 5'd0);
        $display("  Initial weight: %0d", w_before);

        for (i = 0; i < 10; i = i + 1) begin
            stimulate(8'd0, 16'sd1200);
            run_timestep;
            // Post fires (trace of pre still active → LTP)
            stimulate(8'd1, 16'sd1200);
            run_timestep;
            // Let traces decay
            run_timestep;
        end

        w_after = read_weight(8'd0, 5'd0);
        $display("  After 10 pre→post cycles: %0d", w_after);

        if (w_after > w_before + 50) begin
            $display("  PASS: Significant strengthening (%0d → %0d, +%0d)",
                w_before, w_after, w_after - w_before);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Insufficient strengthening (%0d → %0d)",
                w_before, w_after);
            fail_count = fail_count + 1;
        end

        $display("");
        $display("================================================================");
        $display("  STDP TEST RESULTS: %0d PASS, %0d FAIL", pass_count, fail_count);
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
