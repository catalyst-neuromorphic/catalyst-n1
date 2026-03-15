// ============================================================================
// Testbench: Scalable Core V2 (256 neurons, sparse connectivity)
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

module tb_scalable_core_v2;

    parameter DATA_WIDTH     = 16;
    parameter NUM_NEURONS    = 256;
    parameter NEURON_BITS    = 8;
    parameter MAX_FANOUT     = 32;
    parameter FANOUT_BITS    = 5;
    parameter CONN_ADDR_BITS = 13;
    parameter CLK_PERIOD     = 10;

    reg                          clk, rst_n;
    reg                          start, learn_enable;
    reg                          ext_valid;
    reg  [NEURON_BITS-1:0]       ext_neuron_id;
    reg  signed [DATA_WIDTH-1:0] ext_current;
    reg                          conn_we;
    reg  [NEURON_BITS-1:0]       conn_src;
    reg  [FANOUT_BITS-1:0]       conn_slot;
    reg  [NEURON_BITS-1:0]       conn_target;
    reg  signed [DATA_WIDTH-1:0] conn_weight;

    wire                         timestep_done;
    wire                         spike_out_valid;
    wire [NEURON_BITS-1:0]       spike_out_id;
    wire [3:0]                   state_out;
    wire [31:0]                  total_spikes;
    wire [31:0]                  timestep_count;

    integer spike_count [0:NUM_NEURONS-1];
    integer i;

    scalable_core_v2 #(
        .NUM_NEURONS   (NUM_NEURONS),
        .DATA_WIDTH    (DATA_WIDTH),
        .NEURON_BITS   (NEURON_BITS),
        .MAX_FANOUT    (MAX_FANOUT),
        .FANOUT_BITS   (FANOUT_BITS),
        .CONN_ADDR_BITS(CONN_ADDR_BITS),
        .THRESHOLD     (16'sd1000),
        .LEAK_RATE     (16'sd3),
        .REFRAC_CYCLES (3),
        .TRACE_MAX     (8'd100),
        .TRACE_DECAY   (8'd3),
        .LEARN_SHIFT   (3)
    ) dut (
        .clk               (clk),
        .rst_n             (rst_n),
        .start             (start),
        .learn_enable      (learn_enable),
        .graded_enable     (1'b0),
        .dendritic_enable  (1'b0),
        .ext_valid         (ext_valid),
        .ext_neuron_id     (ext_neuron_id),
        .ext_current       (ext_current),
        .conn_we           (conn_we),
        .conn_src          (conn_src),
        .conn_slot         (conn_slot),
        .conn_target       (conn_target),
        .conn_weight       (conn_weight),
        .conn_comp         (2'd0),
        .prog_param_we     (1'b0),
        .prog_param_neuron (8'd0),
        .prog_param_id     (3'd0),
        .prog_param_value  (16'sd0),
        .timestep_done     (timestep_done),
        .spike_out_valid   (spike_out_valid),
        .spike_out_id      (spike_out_id),
        .spike_out_payload (),
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
        $dumpfile("scalable_core_v2.vcd");
        $dumpvars(0, tb_scalable_core_v2);
    end

    task add_connection;
        input [NEURON_BITS-1:0]      src;
        input [FANOUT_BITS-1:0]      slot;
        input [NEURON_BITS-1:0]      target;
        input signed [DATA_WIDTH-1:0] weight;
    begin
        @(posedge clk);
        conn_we     <= 1;
        conn_src    <= src;
        conn_slot   <= slot;
        conn_target <= target;
        conn_weight <= weight;
        @(posedge clk);
        conn_we     <= 0;
    end
    endtask

    task run_timestep;
        input [NEURON_BITS-1:0]      stim_neuron;
        input signed [DATA_WIDTH-1:0] stim_current;
    begin
        ext_valid     <= 1;
        ext_neuron_id <= stim_neuron;
        ext_current   <= stim_current;
        @(posedge clk);
        ext_valid     <= 0;

        @(posedge clk);
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
        conn_we = 0; conn_src = 0; conn_slot = 0;
        conn_target = 0; conn_weight = 0;

        $display("");
        $display("================================================================");
        $display("  Scalable Core V2 Test - 256 Neurons, Sparse Connectivity");
        $display("================================================================");

        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 5);

        $display("");
        $display("--- TEST 1: Spike Chain (0->1->2->...->7) ---");
        $display("  Programming sparse connections (1 per neuron, slot 0)...");

        add_connection(0, 0, 1, 16'sd600);
        add_connection(1, 0, 2, 16'sd600);
        add_connection(2, 0, 3, 16'sd600);
        add_connection(3, 0, 4, 16'sd600);
        add_connection(4, 0, 5, 16'sd600);
        add_connection(5, 0, 6, 16'sd600);
        add_connection(6, 0, 7, 16'sd600);

        $display("  Running 30 timesteps with stimulus to N0...");

        for (t = 0; t < 30; t = t + 1) begin
            run_timestep(0, 16'sd200);
        end

        $display("");
        $display("  Spike chain results:");
        for (i = 0; i < 8; i = i + 1) begin
            $display("    Neuron %0d: %0d spikes", i, spike_count[i]);
        end

        $display("");
        $display("--- TEST 2: Fan-out (N10 -> N11, N12, N13, N14) ---");

        for (i = 0; i < NUM_NEURONS; i = i + 1) spike_count[i] = 0;

        add_connection(10, 0, 11, 16'sd600);
        add_connection(10, 1, 12, 16'sd600);
        add_connection(10, 2, 13, 16'sd600);
        add_connection(10, 3, 14, 16'sd600);

        $display("  Running 20 timesteps with stimulus to N10...");

        for (t = 0; t < 20; t = t + 1) begin
            run_timestep(10, 16'sd200);
        end

        $display("");
        $display("  Fan-out results:");
        for (i = 10; i < 15; i = i + 1) begin
            $display("    Neuron %0d: %0d spikes", i, spike_count[i]);
        end
        $display("    Neuron 15: %0d spikes (no connection - control)", spike_count[15]);

        $display("");
        $display("--- TEST 3: High Neuron IDs (200->201->202->203) ---");

        for (i = 0; i < NUM_NEURONS; i = i + 1) spike_count[i] = 0;

        add_connection(200, 0, 201, 16'sd600);
        add_connection(201, 0, 202, 16'sd600);
        add_connection(202, 0, 203, 16'sd600);

        $display("  Running 20 timesteps with stimulus to N200...");

        for (t = 0; t < 20; t = t + 1) begin
            run_timestep(200, 16'sd200);
        end

        $display("");
        $display("  High-ID chain results:");
        for (i = 200; i < 204; i = i + 1) begin
            $display("    Neuron %0d: %0d spikes", i, spike_count[i]);
        end

        $display("");
        $display("--- TEST 4: Strong Chain (weight=1200 > threshold=1000) ---");

        for (i = 0; i < NUM_NEURONS; i = i + 1) spike_count[i] = 0;

        add_connection(100, 0, 101, 16'sd1200);
        add_connection(101, 0, 102, 16'sd1200);
        add_connection(102, 0, 103, 16'sd1200);
        add_connection(103, 0, 104, 16'sd1200);
        add_connection(104, 0, 105, 16'sd1200);
        add_connection(105, 0, 106, 16'sd1200);
        add_connection(106, 0, 107, 16'sd1200);

        $display("  Running 30 timesteps with stimulus to N100...");

        for (t = 0; t < 30; t = t + 1) begin
            run_timestep(100, 16'sd200);
        end

        $display("");
        $display("  Strong chain results:");
        for (i = 100; i < 108; i = i + 1) begin
            $display("    Neuron %0d: %0d spikes", i, spike_count[i]);
        end

        $display("");
        $display("================================================================");
        $display("  FINAL REPORT");
        $display("================================================================");
        $display("  Total timesteps: %0d", timestep_count);
        $display("  Total spikes:    %0d", total_spikes);
        $display("  Architecture:    %0d neurons, sparse (max %0d fanout)",
                 NUM_NEURONS, MAX_FANOUT);
        $display("  Connection table: %0d entries (vs %0d dense)",
                 NUM_NEURONS * MAX_FANOUT, NUM_NEURONS * NUM_NEURONS);
        $display("  Memory savings:  %0dx reduction",
                 NUM_NEURONS / MAX_FANOUT);
        $display("================================================================");

        #(CLK_PERIOD * 10);
        $finish;
    end

    reg [3:0] prev_state;
    always @(posedge clk) begin
        if (state_out != prev_state) begin
            if (timestep_count < 3)
                $display("  [dbg] State: %0d -> %0d (ts=%0d)", prev_state, state_out, timestep_count);
            prev_state <= state_out;
        end
    end
    initial prev_state = 0;

    initial begin
        #(CLK_PERIOD * 500000);
        $display("TIMEOUT at state=%0d, ts=%0d", state_out, timestep_count);
        $finish;
    end

endmodule
