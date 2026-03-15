// ============================================================================
// Testbench: Neuron Core
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

module tb_neuron_core;

    parameter DATA_WIDTH = 16;
    parameter CLK_PERIOD = 10; // 100 MHz clock

    reg                          clk;
    reg                          rst_n;
    reg                          enable;
    reg  signed [DATA_WIDTH-1:0] ext_input_0;
    reg  signed [DATA_WIDTH-1:0] ext_input_1;
    reg  signed [DATA_WIDTH-1:0] ext_input_2;
    reg  signed [DATA_WIDTH-1:0] ext_input_3;
    wire [3:0]                   spikes;
    wire [DATA_WIDTH-1:0]        membrane_0, membrane_1, membrane_2, membrane_3;

    reg signed [DATA_WIDTH-1:0] w00, w01, w02, w03;
    reg signed [DATA_WIDTH-1:0] w10, w11, w12, w13;
    reg signed [DATA_WIDTH-1:0] w20, w21, w22, w23;
    reg signed [DATA_WIDTH-1:0] w30, w31, w32, w33;

    integer spike_count_0 = 0;
    integer spike_count_1 = 0;
    integer spike_count_2 = 0;
    integer spike_count_3 = 0;

    neuron_core #(
        .DATA_WIDTH(DATA_WIDTH),
        .THRESHOLD(16'd1000),
        .LEAK_RATE(16'd2)
    ) dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .enable     (enable),
        .ext_input_0(ext_input_0),
        .ext_input_1(ext_input_1),
        .ext_input_2(ext_input_2),
        .ext_input_3(ext_input_3),
        .weight_00  (w00), .weight_01(w01), .weight_02(w02), .weight_03(w03),
        .weight_10  (w10), .weight_11(w11), .weight_12(w12), .weight_13(w13),
        .weight_20  (w20), .weight_21(w21), .weight_22(w22), .weight_23(w23),
        .weight_30  (w30), .weight_31(w31), .weight_32(w32), .weight_33(w33),
        .spikes     (spikes),
        .membrane_0 (membrane_0),
        .membrane_1 (membrane_1),
        .membrane_2 (membrane_2),
        .membrane_3 (membrane_3)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    always @(posedge clk) begin
        if (spikes[0]) spike_count_0 = spike_count_0 + 1;
        if (spikes[1]) spike_count_1 = spike_count_1 + 1;
        if (spikes[2]) spike_count_2 = spike_count_2 + 1;
        if (spikes[3]) spike_count_3 = spike_count_3 + 1;
    end

    always @(posedge clk) begin
        if (spikes[0]) $display("[%0t] SPIKE! Neuron 0 fired (membrane was %0d)", $time, membrane_0);
        if (spikes[1]) $display("[%0t] SPIKE! Neuron 1 fired (membrane was %0d)", $time, membrane_1);
        if (spikes[2]) $display("[%0t] SPIKE! Neuron 2 fired (membrane was %0d)", $time, membrane_2);
        if (spikes[3]) $display("[%0t] SPIKE! Neuron 3 fired (membrane was %0d)", $time, membrane_3);
    end

    initial begin
        $dumpfile("neuron_core.vcd");
        $dumpvars(0, tb_neuron_core);
    end

    initial begin
        $display("============================================");
        $display("  Neuromorphic Chip - Neuron Core Testbench");
        $display("============================================");
        $display("");

        rst_n   = 0;
        enable  = 0;
        ext_input_0 = 0;
        ext_input_1 = 0;
        ext_input_2 = 0;
        ext_input_3 = 0;

        // Setup weight matrix - our neural circuit
        // Neuron 0 -> Neuron 1 (excitatory, strong)
        // Neuron 0 -> Neuron 2 (excitatory, medium)
        // Neuron 2 -> Neuron 3 (excitatory, strong)
        // Neuron 3 -> Neuron 0 (inhibitory - negative feedback!)
        w00 = 16'd0;    w01 = 16'd500;  w02 = 16'd300;  w03 = 16'd0;
        w10 = 16'd0;    w11 = 16'd0;    w12 = 16'd0;    w13 = 16'd0;
        w20 = 16'd0;    w21 = 16'd0;    w22 = 16'd0;    w23 = 16'd500;
        w30 = -16'd400; w31 = 16'd0;    w32 = 16'd0;    w33 = 16'd0;

        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 2);
        enable = 1;

        $display("[%0t] --- Phase 1: Constant stimulus to Neuron 0 ---", $time);
        // Drive neuron 0 with constant excitatory input
        ext_input_0 = 16'd100;

        // Let it run for 200 cycles
        #(CLK_PERIOD * 200);

        $display("");
        $display("[%0t] --- Phase 2: Increased stimulus ---", $time);
        // Increase input - should fire faster
        ext_input_0 = 16'd200;
        #(CLK_PERIOD * 200);

        $display("");
        $display("[%0t] --- Phase 3: Dual stimulus (neurons 0 and 2) ---", $time);
        // Now also stimulate neuron 2 directly
        ext_input_2 = 16'd150;
        #(CLK_PERIOD * 200);

        $display("");
        $display("[%0t] --- Phase 4: Remove stimulus, observe decay ---", $time);
        // Remove all input - watch the network wind down
        ext_input_0 = 16'd0;
        ext_input_2 = 16'd0;
        #(CLK_PERIOD * 100);

        $display("");
        $display("============================================");
        $display("  Simulation Complete - Spike Statistics");
        $display("============================================");
        $display("  Neuron 0: %0d spikes", spike_count_0);
        $display("  Neuron 1: %0d spikes", spike_count_1);
        $display("  Neuron 2: %0d spikes", spike_count_2);
        $display("  Neuron 3: %0d spikes", spike_count_3);
        $display("  Total:    %0d spikes", spike_count_0 + spike_count_1 + spike_count_2 + spike_count_3);
        $display("============================================");

        $finish;
    end

endmodule
