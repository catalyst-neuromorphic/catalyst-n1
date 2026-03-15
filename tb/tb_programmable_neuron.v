// ============================================================================
// Testbench: Programmable Neuron Parameters (Phase 9)
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

module tb_programmable_neuron;

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
    reg                    graded_enable;
    reg                    ext_valid;
    reg  [NEURON_BITS-1:0] ext_neuron_id;
    reg  signed [DATA_WIDTH-1:0] ext_current;
    reg                    conn_we;
    reg  [NEURON_BITS-1:0] conn_src;
    reg  [FANOUT_BITS-1:0] conn_slot;
    reg  [NEURON_BITS-1:0] conn_target;
    reg  signed [DATA_WIDTH-1:0] conn_weight;

    reg                    prog_param_we;
    reg  [NEURON_BITS-1:0] prog_param_neuron;
    reg  [2:0]             prog_param_id;
    reg  signed [DATA_WIDTH-1:0] prog_param_value;

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
        .prog_param_we    (prog_param_we),
        .prog_param_neuron(prog_param_neuron),
        .prog_param_id    (prog_param_id),
        .prog_param_value (prog_param_value),
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

    task set_param;
        input [NEURON_BITS-1:0] neuron;
        input [2:0] param_id;
        input signed [DATA_WIDTH-1:0] value;
    begin
        @(posedge clk);
        prog_param_we     <= 1;
        prog_param_neuron <= neuron;
        prog_param_id     <= param_id;
        prog_param_value  <= value;
        @(posedge clk);
        prog_param_we <= 0;
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

    // Read threshold parameter
    function signed [DATA_WIDTH-1:0] read_threshold;
        input [NEURON_BITS-1:0] neuron;
    begin
        read_threshold = dut.threshold_mem.mem[neuron];
    end
    endfunction

    integer spike_count_per_neuron [0:NUM_NEURONS-1];
    integer first_spike_ts [0:NUM_NEURONS-1];
    integer total_spike_count;
    integer i;

    always @(posedge clk) begin
        if (spike_out_valid) begin
            spike_count_per_neuron[spike_out_id] =
                spike_count_per_neuron[spike_out_id] + 1;
            if (first_spike_ts[spike_out_id] == -1)
                first_spike_ts[spike_out_id] = timestep_count;
            total_spike_count = total_spike_count + 1;
        end
    end

    task reset_spike_tracking;
    begin
        for (i = 0; i < NUM_NEURONS; i = i + 1) begin
            spike_count_per_neuron[i] = 0;
            first_spike_ts[i] = -1;
        end
        total_spike_count = 0;
    end
    endtask

    integer pass_count, fail_count;
    integer t;

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
        prog_param_we     = 0;
        prog_param_neuron = 0;
        prog_param_id     = 0;
        prog_param_value  = 0;
        ext_neuron_id = 0;
        ext_current   = 0;
        pass_count    = 0;
        fail_count    = 0;
        reset_spike_tracking();

        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 3);

        $display("");
        $display("================================================================");
        $display("  Programmable Neuron Parameters Test (Phase 9)");
        $display("================================================================");

        // TEST 1: Default Values (no programming)
        //   N0 with default threshold=1000, leak=3
        //   Stimulus=200/ts -> need ~6 timesteps to reach 1000
        //   (200-3)*5 = 985 < 1000, (200-3)*6 = 1182 >= 1000 -> spike at ts ~5-6
        $display("");
        $display("--- TEST 1: Default Values (backward compatibility) ---");

        reset_spike_tracking();

        for (t = 0; t < 10; t = t + 1) begin
            stimulate(8'd0, 16'sd200);
            run_timestep;
        end

        $display("  N0 spikes (default threshold=1000): %0d", spike_count_per_neuron[0]);
        $display("  N0 first spike at timestep: %0d", first_spike_ts[0]);

        // With stim=200, leak=3: net=197/ts. Threshold=1000.
        // Accumulation: 197, 394, 591, 788, 985, 1182 -> spike at ts 5 (0-indexed)
        if (spike_count_per_neuron[0] > 0 && first_spike_ts[0] >= 4 && first_spike_ts[0] <= 6) begin
            $display("  PASS: Default parameters work correctly");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected spike around ts 5, got first=%0d count=%0d",
                first_spike_ts[0], spike_count_per_neuron[0]);
            fail_count = fail_count + 1;
        end

        // Verify threshold SRAM was initialized to default
        begin : test1_verify
            reg signed [DATA_WIDTH-1:0] thr_val;
            thr_val = read_threshold(8'd0);
            $display("  Threshold SRAM N0 = %0d (expected 1000)", thr_val);
            if (thr_val == 16'sd1000) begin
                $display("  PASS: Threshold SRAM initialized correctly");
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL: Expected 1000, got %0d", thr_val);
                fail_count = fail_count + 1;
            end
        end

        // TEST 2: Per-Neuron Threshold Variation
        //   N10: threshold=500 (low), N11: threshold=1500 (high), N12: default=1000
        //   Same stimulus -> N10 fires first, N12 second, N11 last
        $display("");
        $display("--- TEST 2: Per-Neuron Threshold Variation ---");

        reset_spike_tracking();

        set_param(8'd10, 3'd0, 16'sd500);   // N10: low threshold
        set_param(8'd11, 3'd0, 16'sd1500);  // N11: high threshold
        // N12: keep default=1000

        // Verify SRAM write
        $display("  N10 threshold = %0d (programmed 500)", read_threshold(8'd10));
        $display("  N11 threshold = %0d (programmed 1500)", read_threshold(8'd11));
        $display("  N12 threshold = %0d (default 1000)", read_threshold(8'd12));

        // Stimulate all three with same current
        for (t = 0; t < 15; t = t + 1) begin
            stimulate(8'd10, 16'sd200);
            run_timestep;
            stimulate(8'd11, 16'sd200);
            run_timestep;
            stimulate(8'd12, 16'sd200);
            run_timestep;
        end

        $display("  N10 spikes: %0d (first at ts %0d) - threshold=500",
            spike_count_per_neuron[10], first_spike_ts[10]);
        $display("  N11 spikes: %0d (first at ts %0d) - threshold=1500",
            spike_count_per_neuron[11], first_spike_ts[11]);
        $display("  N12 spikes: %0d (first at ts %0d) - threshold=1000",
            spike_count_per_neuron[12], first_spike_ts[12]);

        // N10 (thr=500): 197, 394, 591 -> spikes at ts ~2
        // N12 (thr=1000): needs ~6 stimulations
        // N11 (thr=1500): needs ~8 stimulations
        // Since we stimulate each neuron every 3 timesteps:
        // N10 first spike should be earliest, N11 last
        if (first_spike_ts[10] < first_spike_ts[12] &&
            first_spike_ts[12] < first_spike_ts[11]) begin
            $display("  PASS: N10 < N12 < N11 (low thr fires first)");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected N10 < N12 < N11 ordering");
            fail_count = fail_count + 1;
        end

        if (spike_count_per_neuron[10] > spike_count_per_neuron[11]) begin
            $display("  PASS: Low threshold neuron fires more often (%0d > %0d)",
                spike_count_per_neuron[10], spike_count_per_neuron[11]);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected N10 > N11 spike count");
            fail_count = fail_count + 1;
        end

        // TEST 3: Per-Neuron Leak Rate Variation
        //   N20: leak=1 (slow decay), N21: leak=50 (fast decay)
        //   Give sub-threshold stimulus then check potential retention
        $display("");
        $display("--- TEST 3: Per-Neuron Leak Rate Variation ---");

        reset_spike_tracking();

        set_param(8'd20, 3'd1, 16'sd1);   // N20: very slow leak
        set_param(8'd21, 3'd1, 16'sd50);  // N21: very fast leak

        // Give both 3 stimulations of 200 each
        for (t = 0; t < 3; t = t + 1) begin
            stimulate(8'd20, 16'sd200);
            run_timestep;
            stimulate(8'd21, 16'sd200);
            run_timestep;
        end

        // Now run 5 empty timesteps (no stimulus) - let them leak
        for (t = 0; t < 5; t = t + 1) begin
            run_timestep;
        end

        begin : test3_block
            reg signed [DATA_WIDTH-1:0] pot20, pot21;
            pot20 = read_potential(8'd20);
            pot21 = read_potential(8'd21);
            $display("  N20 potential (leak=1):  %0d", pot20);
            $display("  N21 potential (leak=50): %0d", pot21);

            // N20 should retain much more potential than N21
            if (pot20 > pot21) begin
                $display("  PASS: Slow-leak neuron retains more potential (%0d > %0d)", pot20, pot21);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL: Expected N20 > N21 (%0d vs %0d)", pot20, pot21);
                fail_count = fail_count + 1;
            end
        end

        // TEST 4: Per-Neuron Refractory Period Variation
        //   N30: refrac=1 (fast recovery), N31: refrac=10 (slow recovery)
        //   Strong continuous stimulus -> N30 fires more often
        $display("");
        $display("--- TEST 4: Per-Neuron Refractory Period Variation ---");

        reset_spike_tracking();

        set_param(8'd30, 3'd3, 16'sd1);   // N30: refrac=1 (fast)
        set_param(8'd31, 3'd3, 16'sd10);  // N31: refrac=10 (slow)

        // Strong stimulus to both (above threshold in one shot)
        for (t = 0; t < 30; t = t + 1) begin
            stimulate(8'd30, 16'sd1200);
            run_timestep;
            stimulate(8'd31, 16'sd1200);
            run_timestep;
        end

        $display("  N30 spikes (refrac=1):  %0d", spike_count_per_neuron[30]);
        $display("  N31 spikes (refrac=10): %0d", spike_count_per_neuron[31]);

        // N30 should fire much more often (recovers in 1 cycle vs 10)
        if (spike_count_per_neuron[30] > spike_count_per_neuron[31]) begin
            $display("  PASS: Fast-recovery neuron fires more (%0d > %0d)",
                spike_count_per_neuron[30], spike_count_per_neuron[31]);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected N30 > N31 spike count");
            fail_count = fail_count + 1;
        end

        // TEST 5: Mixed Population Chain
        //   N40->N41: N40 threshold=500, N41 threshold=1500
        //   N50->N51: N50 threshold=1500, N51 threshold=500
        //   Same stimulus -> first chain propagates, second doesn't
        $display("");
        $display("--- TEST 5: Mixed Population Chain ---");

        reset_spike_tracking();

        // Chain 1: easy source -> hard target
        set_param(8'd40, 3'd0, 16'sd500);   // N40: low threshold
        set_param(8'd41, 3'd0, 16'sd1500);  // N41: high threshold
        program_conn(8'd40, 5'd0, 8'd41, 16'sd600);

        // Chain 2: hard source -> easy target
        set_param(8'd50, 3'd0, 16'sd1500);  // N50: high threshold
        set_param(8'd51, 3'd0, 16'sd500);   // N51: low threshold
        program_conn(8'd50, 5'd0, 8'd51, 16'sd600);

        // Moderate stimulus to both sources
        for (t = 0; t < 20; t = t + 1) begin
            stimulate(8'd40, 16'sd200);
            run_timestep;
            stimulate(8'd50, 16'sd200);
            run_timestep;
        end

        $display("  Chain 1: N40(thr=500) spikes=%0d, N41(thr=1500) spikes=%0d",
            spike_count_per_neuron[40], spike_count_per_neuron[41]);
        $display("  Chain 2: N50(thr=1500) spikes=%0d, N51(thr=500) spikes=%0d",
            spike_count_per_neuron[50], spike_count_per_neuron[51]);

        // N40 fires easily (low threshold), but N41 is hard to trigger
        // N50 fires rarely (high threshold), but when it does N51 triggers easily
        if (spike_count_per_neuron[40] > spike_count_per_neuron[50]) begin
            $display("  PASS: Low-threshold source fires more (%0d > %0d)",
                spike_count_per_neuron[40], spike_count_per_neuron[50]);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected N40 > N50");
            fail_count = fail_count + 1;
        end

        $display("");
        $display("================================================================");
        $display("  PROGRAMMABLE NEURON TEST RESULTS: %0d PASS, %0d FAIL",
            pass_count, fail_count);
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
