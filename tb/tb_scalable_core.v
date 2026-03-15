// ============================================================================
// Testbench: Scalable Core (64 neurons, SRAM-backed)
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

module tb_scalable_core;

    parameter DATA_WIDTH  = 16;
    parameter NUM_NEURONS = 64;
    parameter NEURON_BITS = 6;
    parameter WEIGHT_BITS = 12;
    parameter CLK_PERIOD  = 10;

    reg                          clk, rst_n;
    reg                          start, learn_enable;
    reg                          ext_valid;
    reg  [NEURON_BITS-1:0]       ext_neuron_id;
    reg  signed [DATA_WIDTH-1:0] ext_current;
    reg                          inject_spike_valid;
    reg  [NEURON_BITS-1:0]       inject_spike_id;
    reg                          weight_we;
    reg  [WEIGHT_BITS-1:0]       weight_addr;
    reg  signed [DATA_WIDTH-1:0] weight_data;

    wire                         timestep_done;
    wire                         spike_out_valid;
    wire [NEURON_BITS-1:0]       spike_out_id;
    wire [3:0]                   state_out;
    wire [15:0]                  total_spikes;
    wire [15:0]                  timestep_count;

    integer spike_count [0:NUM_NEURONS-1];
    integer i;

    scalable_core #(
        .NUM_NEURONS  (NUM_NEURONS),
        .DATA_WIDTH   (DATA_WIDTH),
        .NEURON_BITS  (NEURON_BITS),
        .WEIGHT_BITS  (WEIGHT_BITS),
        .THRESHOLD    (16'sd1000),
        .LEAK_RATE    (16'sd3),
        .REFRAC_CYCLES(3),
        .TRACE_MAX    (8'd100),
        .TRACE_DECAY  (8'd3),
        .LEARN_SHIFT  (3)
    ) dut (
        .clk               (clk),
        .rst_n             (rst_n),
        .start             (start),
        .learn_enable      (learn_enable),
        .ext_valid         (ext_valid),
        .ext_neuron_id     (ext_neuron_id),
        .ext_current       (ext_current),
        .inject_spike_valid(inject_spike_valid),
        .inject_spike_id   (inject_spike_id),
        .weight_we         (weight_we),
        .weight_addr       (weight_addr),
        .weight_data       (weight_data),
        .timestep_done     (timestep_done),
        .spike_out_valid   (spike_out_valid),
        .spike_out_id      (spike_out_id),
        .state_out         (state_out),
        .total_spikes      (total_spikes),
        .timestep_count    (timestep_count)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    always @(posedge clk) begin
        if (spike_out_valid) begin
            spike_count[spike_out_id] = spike_count[spike_out_id] + 1;
            $display("  [t=%0d] Neuron %0d spiked!", timestep_count, spike_out_id);
        end
    end

    initial begin
        $dumpfile("scalable_core.vcd");
        $dumpvars(0, tb_scalable_core);
    end

    task set_weight;
        input [NEURON_BITS-1:0] src;
        input [NEURON_BITS-1:0] dst;
        input signed [DATA_WIDTH-1:0] w;
    begin
        @(posedge clk);
        weight_we   <= 1;
        weight_addr <= {src, dst};
        weight_data <= w;
        @(posedge clk);
        weight_we   <= 0;
    end
    endtask

    task run_timestep;
        input [NEURON_BITS-1:0] stim_neuron;
        input signed [DATA_WIDTH-1:0] stim_current;
    begin
        // Apply external current
        ext_valid     <= 1;
        ext_neuron_id <= stim_neuron;
        ext_current   <= stim_current;
        @(posedge clk);
        ext_valid     <= 0;

        @(posedge clk);
        start <= 1;
        @(posedge clk);
        start <= 0;

        // Wait for completion
        wait(timestep_done);
        @(posedge clk);
    end
    endtask

    task run_timestep_multi;
        input [NEURON_BITS-1:0] stim_n0;
        input signed [DATA_WIDTH-1:0] stim_c0;
        input [NEURON_BITS-1:0] stim_n1;
        input signed [DATA_WIDTH-1:0] stim_c1;
    begin
        ext_valid <= 1; ext_neuron_id <= stim_n0; ext_current <= stim_c0;
        @(posedge clk);
        ext_neuron_id <= stim_n1; ext_current <= stim_c1;
        @(posedge clk);
        ext_valid <= 0;

        start <= 1;
        @(posedge clk);
        start <= 0;

        wait(timestep_done);
        @(posedge clk);
    end
    endtask

    task run_timestep_empty;
    begin
        @(posedge clk);
        start <= 1;
        @(posedge clk);
        start <= 0;
        wait(timestep_done);
        @(posedge clk);
    end
    endtask

    integer t;
    initial begin
        for (i = 0; i < NUM_NEURONS; i = i + 1) spike_count[i] = 0;
        rst_n = 0; start = 0; learn_enable = 0;
        ext_valid = 0; ext_neuron_id = 0; ext_current = 0;
        inject_spike_valid = 0; inject_spike_id = 0;
        weight_we = 0; weight_addr = 0; weight_data = 0;

        $display("");
        $display("================================================================");
        $display("  Scalable Core Test - 64 Neurons, SRAM-backed");
        $display("================================================================");

        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 5);

        $display("");
        $display("--- TEST 1: Spike Chain (0->1->2->...->7) ---");
        $display("  Programming weights...");

        // Strong forward connections: each neuron excites the next
        set_weight(0, 1, 16'sd600);
        set_weight(1, 2, 16'sd600);
        set_weight(2, 3, 16'sd600);
        set_weight(3, 4, 16'sd600);
        set_weight(4, 5, 16'sd600);
        set_weight(5, 6, 16'sd600);
        set_weight(6, 7, 16'sd600);

        $display("  Running 30 timesteps with stimulus to N0...");

        // Run timesteps - stimulate neuron 0
        for (t = 0; t < 30; t = t + 1) begin
            run_timestep(0, 16'sd200);
        end

        $display("");
        $display("  Spike chain results:");
        for (i = 0; i < 8; i = i + 1) begin
            $display("    Neuron %0d: %0d spikes", i, spike_count[i]);
        end

        $display("");
        $display("--- TEST 2: Wide Activity (16 neurons with cross-connections) ---");

        // Reset spike counts
        for (i = 0; i < NUM_NEURONS; i = i + 1) spike_count[i] = 0;

        // Program some cross-connections in a ring: 10→11→12→...→25→10
        for (i = 10; i < 25; i = i + 1) begin
            set_weight(i[NEURON_BITS-1:0], (i+1), 16'sd500);
        end
        set_weight(25, 10, 16'sd500); // Close the ring

        $display("  Running 20 timesteps stimulating neurons 10-13...");

        for (t = 0; t < 20; t = t + 1) begin
            // Stimulate multiple neurons
            ext_valid <= 1; ext_neuron_id <= 10; ext_current <= 16'sd200;
            @(posedge clk);
            ext_neuron_id <= 11; ext_current <= 16'sd200;
            @(posedge clk);
            ext_neuron_id <= 12; ext_current <= 16'sd200;
            @(posedge clk);
            ext_neuron_id <= 13; ext_current <= 16'sd200;
            @(posedge clk);
            ext_valid <= 0;

            start <= 1; @(posedge clk); start <= 0;
            wait(timestep_done); @(posedge clk);
        end

        $display("");
        $display("  Ring activity results:");
        for (i = 10; i < 26; i = i + 1) begin
            if (spike_count[i] > 0)
                $display("    Neuron %0d: %0d spikes", i, spike_count[i]);
        end

        $display("");
        $display("--- TEST 3: STDP Learning ---");
        $display("  Stimulating N32 and N33 together (correlated)...");

        for (i = 0; i < NUM_NEURONS; i = i + 1) spike_count[i] = 0;

        // Start with no connections between 32-35
        learn_enable = 1;

        for (t = 0; t < 40; t = t + 1) begin
            // Correlated input to 32 and 33
            ext_valid <= 1; ext_neuron_id <= 32; ext_current <= 16'sd250;
            @(posedge clk);
            ext_neuron_id <= 33; ext_current <= 16'sd250;
            @(posedge clk);
            ext_valid <= 0;

            start <= 1; @(posedge clk); start <= 0;
            wait(timestep_done); @(posedge clk);
        end

        learn_enable = 0;

        $display("");
        $display("  After STDP training:");
        $display("    N32 spikes: %0d", spike_count[32]);
        $display("    N33 spikes: %0d", spike_count[33]);

        // Now test recall - only stimulate N32
        $display("");
        $display("  Recall test: only stimulating N32...");
        for (i = 0; i < NUM_NEURONS; i = i + 1) spike_count[i] = 0;

        for (t = 0; t < 20; t = t + 1) begin
            run_timestep(32, 16'sd250);
        end

        $display("    N32 spikes: %0d (stimulated)", spike_count[32]);
        $display("    N33 spikes: %0d (from learned weight)", spike_count[33]);
        $display("    N34 spikes: %0d (no connection, control)", spike_count[34]);

        $display("");
        $display("================================================================");
        $display("  FINAL REPORT");
        $display("================================================================");
        $display("  Total timesteps: %0d", timestep_count);
        $display("  Total spikes:    %0d", total_spikes);
        $display("  Architecture:    %0d neurons, SRAM-backed", NUM_NEURONS);
        $display("  Weight memory:   %0d x %0d = %0d entries",
                 NUM_NEURONS, NUM_NEURONS, NUM_NEURONS * NUM_NEURONS);
        $display("================================================================");

        #(CLK_PERIOD * 10);
        $finish;
    end

    reg [3:0] prev_state;
    always @(posedge clk) begin
        if (state_out != prev_state) begin
            $display("  [dbg] State: %0d -> %0d (cycle %0d)", prev_state, state_out, timestep_count);
            prev_state <= state_out;
        end
    end
    initial prev_state = 0;

    initial begin
        #(CLK_PERIOD * 50000);
        $display("TIMEOUT at state=%0d", state_out);
        $finish;
    end

endmodule
