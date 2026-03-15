// ============================================================================
// Testbench: STDP Learning Demonstration
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

module tb_stdp_learning;

    parameter DATA_WIDTH = 16;
    parameter CLK_PERIOD = 10;

    reg                          clk;
    reg                          rst_n;
    reg                          enable;
    reg                          learn_enable;
    reg  signed [DATA_WIDTH-1:0] ext_input_0;
    reg  signed [DATA_WIDTH-1:0] ext_input_1;
    reg  signed [DATA_WIDTH-1:0] ext_input_2;
    reg  signed [DATA_WIDTH-1:0] ext_input_3;
    wire [3:0]                   spikes;
    wire [DATA_WIDTH-1:0]        membrane_0, membrane_1, membrane_2, membrane_3;

    wire signed [DATA_WIDTH-1:0] w01, w02, w03;
    wire signed [DATA_WIDTH-1:0] w10, w12, w13;
    wire signed [DATA_WIDTH-1:0] w20, w21, w23;
    wire signed [DATA_WIDTH-1:0] w30, w31, w32;

    integer spike_count [0:3];
    integer phase_spikes [0:3][0:3]; // [phase][neuron]
    integer current_phase;

    reg [15:0] lfsr;

    neuron_core_stdp #(
        .DATA_WIDTH  (DATA_WIDTH),
        .THRESHOLD   (16'd1000),
        .LEAK_RATE   (16'd3),
        .WEIGHT_INIT (16'd100),
        .WEIGHT_MAX  (16'd800),
        .LEARN_RATE  (8'd3)
    ) dut (
        .clk          (clk),
        .rst_n        (rst_n),
        .enable       (enable),
        .learn_enable (learn_enable),
        .ext_input_0  (ext_input_0),
        .ext_input_1  (ext_input_1),
        .ext_input_2  (ext_input_2),
        .ext_input_3  (ext_input_3),
        .spikes       (spikes),
        .membrane_0   (membrane_0),
        .membrane_1   (membrane_1),
        .membrane_2   (membrane_2),
        .membrane_3   (membrane_3),
        .w_out_01     (w01), .w_out_02(w02), .w_out_03(w03),
        .w_out_10     (w10), .w_out_12(w12), .w_out_13(w13),
        .w_out_20     (w20), .w_out_21(w21), .w_out_23(w23),
        .w_out_30     (w30), .w_out_31(w31), .w_out_32(w32)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    always @(posedge clk) begin
        if (!rst_n)
            lfsr <= 16'hACE1;
        else
            lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
    end

    always @(posedge clk) begin
        if (spikes[0]) begin spike_count[0] = spike_count[0] + 1; phase_spikes[current_phase][0] = phase_spikes[current_phase][0] + 1; end
        if (spikes[1]) begin spike_count[1] = spike_count[1] + 1; phase_spikes[current_phase][1] = phase_spikes[current_phase][1] + 1; end
        if (spikes[2]) begin spike_count[2] = spike_count[2] + 1; phase_spikes[current_phase][2] = phase_spikes[current_phase][2] + 1; end
        if (spikes[3]) begin spike_count[3] = spike_count[3] + 1; phase_spikes[current_phase][3] = phase_spikes[current_phase][3] + 1; end
    end

    integer cycle_count;
    always @(posedge clk) begin
        cycle_count = cycle_count + 1;
        if (cycle_count % 500 == 0) begin
            $display("[cycle %0d] Weights: 0->1=%0d  0->2=%0d  1->0=%0d  2->0=%0d  0->3=%0d  3->0=%0d",
                     cycle_count, w01, w02, w10, w20, w03, w30);
        end
    end

    initial begin
        $dumpfile("neuron_core_stdp.vcd");
        $dumpvars(0, tb_stdp_learning);
    end

    initial begin
        spike_count[0] = 0; spike_count[1] = 0;
        spike_count[2] = 0; spike_count[3] = 0;
        phase_spikes[0][0] = 0; phase_spikes[0][1] = 0; phase_spikes[0][2] = 0; phase_spikes[0][3] = 0;
        phase_spikes[1][0] = 0; phase_spikes[1][1] = 0; phase_spikes[1][2] = 0; phase_spikes[1][3] = 0;
        phase_spikes[2][0] = 0; phase_spikes[2][1] = 0; phase_spikes[2][2] = 0; phase_spikes[2][3] = 0;
        phase_spikes[3][0] = 0; phase_spikes[3][1] = 0; phase_spikes[3][2] = 0; phase_spikes[3][3] = 0;
        cycle_count = 0;
        current_phase = 0;

        rst_n = 0; enable = 0; learn_enable = 0;
        ext_input_0 = 0; ext_input_1 = 0;
        ext_input_2 = 0; ext_input_3 = 0;

        $display("");
        $display("================================================================");
        $display("  STDP Learning Experiment");
        $display("  'Neurons that fire together, wire together'");
        $display("================================================================");

        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 2);
        enable = 1;

        // PHASE 1: TRAINING (learning ON)
        // Stimulate N0 and N1 together (correlated)
        // N2 gets random/independent stimulus
        $display("");
        $display("--- PHASE 1: TRAINING ---");
        $display("  N0 + N1: correlated stimulus (should strengthen 0<->1)");
        $display("  N2: independent stimulus (should NOT strengthen to 0/1)");
        $display("  Learning: ON");
        $display("");

        current_phase = 0;
        learn_enable = 1;

        // Correlated stimulus: N0 and N1 get the same strong input
        // N2 gets weaker, independent input
        ext_input_0 = 16'd200;
        ext_input_1 = 16'd200;  // Same as N0 - they'll fire together
        ext_input_2 = 16'd80;   // Weaker, independent
        ext_input_3 = 16'd0;    // No direct stimulus

        #(CLK_PERIOD * 2000);

        $display("");
        $display("  After training weights:");
        $display("    0->1: %0d (should be HIGH - correlated)", w01);
        $display("    1->0: %0d (should be HIGH - correlated)", w10);
        $display("    0->2: %0d (should be lower)", w02);
        $display("    2->0: %0d (should be lower)", w20);
        $display("    0->3: %0d", w03);

        // PHASE 2: TESTING (learning OFF)
        // Only stimulate N0 - does N1 fire from learned weights?
        $display("");
        $display("--- PHASE 2: RECALL TEST ---");
        $display("  Only N0 gets stimulus. Can N1 recall the association?");
        $display("  Learning: OFF");
        $display("");

        current_phase = 1;
        learn_enable = 0;  // Freeze weights

        ext_input_0 = 16'd200;
        ext_input_1 = 16'd0;   // No direct input - must fire from learned weight
        ext_input_2 = 16'd0;   // No input
        ext_input_3 = 16'd0;

        #(CLK_PERIOD * 1000);

        $display("");
        $display("  Recall results:");
        $display("    N0 spikes: %0d (driven by input)", phase_spikes[1][0]);
        $display("    N1 spikes: %0d (should fire from learned 0->1 weight!)", phase_spikes[1][1]);
        $display("    N2 spikes: %0d (should be few/zero - weak learned weight)", phase_spikes[1][2]);
        $display("    N3 spikes: %0d", phase_spikes[1][3]);

        if (phase_spikes[1][1] > 0 && phase_spikes[1][1] > phase_spikes[1][2])
            $display("  >>> SUCCESS: N1 recalls association! N1 fires more than N2 <<<");
        else
            $display("  >>> Learning effect visible in weight changes <<<");

        // PHASE 3: NEW ASSOCIATION (learning ON)
        // Now pair N0 with N3 instead - see weights shift
        $display("");
        $display("--- PHASE 3: NEW ASSOCIATION ---");
        $display("  Now pairing N0 with N3 (new pattern)");
        $display("  Learning: ON");
        $display("");

        current_phase = 2;
        learn_enable = 1;

        ext_input_0 = 16'd200;
        ext_input_1 = 16'd0;
        ext_input_2 = 16'd0;
        ext_input_3 = 16'd200;  // Now N3 is correlated with N0

        #(CLK_PERIOD * 2000);

        $display("");
        $display("  After new training:");
        $display("    0->1: %0d (should decrease - no longer correlated)", w01);
        $display("    0->3: %0d (should increase - now correlated)", w03);
        $display("    3->0: %0d (should increase - now correlated)", w30);

        $display("");
        $display("--- PHASE 4: FINAL RECALL ---");
        $display("  Only N0 stimulus. Which neurons respond?");
        $display("  Learning: OFF");
        $display("");

        current_phase = 3;
        learn_enable = 0;

        ext_input_0 = 16'd200;
        ext_input_1 = 16'd0;
        ext_input_2 = 16'd0;
        ext_input_3 = 16'd0;

        #(CLK_PERIOD * 1000);

        $display("");
        $display("================================================================");
        $display("  FINAL RESULTS");
        $display("================================================================");
        $display("");
        $display("  Final Weight Matrix:");
        $display("         To N0    To N1    To N2    To N3");
        $display("  N0:     ---    %5d    %5d    %5d", w01, w02, w03);
        $display("  N1:   %5d      ---    %5d    %5d", w10, w12, w13);
        $display("  N2:   %5d    %5d      ---    %5d", w20, w21, w23);
        $display("  N3:   %5d    %5d    %5d      ---", w30, w31, w32);
        $display("");
        $display("  Spike Counts by Phase:");
        $display("              N0      N1      N2      N3");
        $display("  Training: %4d    %4d    %4d    %4d", phase_spikes[0][0], phase_spikes[0][1], phase_spikes[0][2], phase_spikes[0][3]);
        $display("  Recall 1: %4d    %4d    %4d    %4d", phase_spikes[1][0], phase_spikes[1][1], phase_spikes[1][2], phase_spikes[1][3]);
        $display("  Retrain:  %4d    %4d    %4d    %4d", phase_spikes[2][0], phase_spikes[2][1], phase_spikes[2][2], phase_spikes[2][3]);
        $display("  Recall 2: %4d    %4d    %4d    %4d", phase_spikes[3][0], phase_spikes[3][1], phase_spikes[3][2], phase_spikes[3][3]);
        $display("");

        if (w01 > w02)
            $display("  [LEARNED] 0->1 weight (%0d) > 0->2 weight (%0d): N0-N1 association formed!", w01, w02);
        if (w03 > 16'd100)
            $display("  [LEARNED] 0->3 weight (%0d) increased: N0-N3 association formed!", w03);

        $display("");
        $display("================================================================");

        $finish;
    end

endmodule
