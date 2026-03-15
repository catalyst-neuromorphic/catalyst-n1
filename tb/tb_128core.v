// ============================================================================
// Testbench: 128-Core Neuromorphic Mesh (Phase 11)
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

module tb_128core;

    parameter NUM_CORES      = 128;
    parameter CORE_ID_BITS   = 7;
    parameter NUM_NEURONS    = 256;
    parameter NEURON_BITS    = 8;
    parameter DATA_WIDTH     = 16;
    parameter MAX_FANOUT     = 32;
    parameter FANOUT_BITS    = 5;
    parameter CONN_ADDR_BITS = 13;
    parameter CLK_PERIOD     = 10;

    reg                          clk, rst_n;
    reg                          start;

    reg                          prog_conn_we;
    reg  [CORE_ID_BITS-1:0]     prog_conn_core;
    reg  [NEURON_BITS-1:0]      prog_conn_src;
    reg  [FANOUT_BITS-1:0]      prog_conn_slot;
    reg  [NEURON_BITS-1:0]      prog_conn_target;
    reg  signed [DATA_WIDTH-1:0] prog_conn_weight;

    reg                          prog_route_we;
    reg  [CORE_ID_BITS-1:0]     prog_route_src_core;
    reg  [NEURON_BITS-1:0]      prog_route_src_neuron;
    reg  [CORE_ID_BITS-1:0]     prog_route_dest_core;
    reg  [NEURON_BITS-1:0]      prog_route_dest_neuron;
    reg  signed [DATA_WIDTH-1:0] prog_route_weight;

    reg                          ext_valid;
    reg  [CORE_ID_BITS-1:0]     ext_core;
    reg  [NEURON_BITS-1:0]      ext_neuron_id;
    reg  signed [DATA_WIDTH-1:0] ext_current;

    wire                         timestep_done;
    wire [NUM_CORES-1:0]         spike_valid_bus;
    wire [NUM_CORES*NEURON_BITS-1:0] spike_id_bus;
    wire [4:0]                   mesh_state_out;
    wire [31:0]                  total_spikes;
    wire [31:0]                  timestep_count;

    integer spike_count;
    integer core_spiked [0:NUM_CORES-1];
    integer i;

    neuromorphic_mesh #(
        .NUM_CORES      (NUM_CORES),
        .CORE_ID_BITS   (CORE_ID_BITS),
        .NUM_NEURONS    (NUM_NEURONS),
        .NEURON_BITS    (NEURON_BITS),
        .DATA_WIDTH     (DATA_WIDTH),
        .MAX_FANOUT     (MAX_FANOUT),
        .FANOUT_BITS    (FANOUT_BITS),
        .CONN_ADDR_BITS (CONN_ADDR_BITS),
        .THRESHOLD      (16'sd1000),
        .LEAK_RATE      (16'sd3),
        .REFRAC_CYCLES  (3)
    ) dut (
        .clk               (clk),
        .rst_n             (rst_n),
        .start             (start),
        .prog_conn_we      (prog_conn_we),
        .prog_conn_core    (prog_conn_core),
        .prog_conn_src     (prog_conn_src),
        .prog_conn_slot    (prog_conn_slot),
        .prog_conn_target  (prog_conn_target),
        .prog_conn_weight  (prog_conn_weight),
        .prog_route_we     (prog_route_we),
        .prog_route_src_core   (prog_route_src_core),
        .prog_route_src_neuron (prog_route_src_neuron),
        .prog_route_dest_core  (prog_route_dest_core),
        .prog_route_dest_neuron(prog_route_dest_neuron),
        .prog_route_weight     (prog_route_weight),
        .learn_enable      (1'b0),
        .graded_enable     (1'b0),
        .dendritic_enable  (1'b0),
        .async_enable      (1'b0),
        .prog_conn_comp    (2'd0),
        .prog_param_we     (1'b0),
        .prog_param_core   ({CORE_ID_BITS{1'b0}}),
        .prog_param_neuron (8'd0),
        .prog_param_id     (3'd0),
        .prog_param_value  (16'sd0),
        .ext_valid         (ext_valid),
        .ext_core          (ext_core),
        .ext_neuron_id     (ext_neuron_id),
        .ext_current       (ext_current),
        .timestep_done     (timestep_done),
        .spike_valid_bus   (spike_valid_bus),
        .spike_id_bus      (spike_id_bus),
        .mesh_state_out    (mesh_state_out),
        .total_spikes      (total_spikes),
        .timestep_count    (timestep_count)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    always @(posedge clk) begin
        for (i = 0; i < NUM_CORES; i = i + 1) begin
            if (spike_valid_bus[i]) begin
                spike_count = spike_count + 1;
                core_spiked[i] = core_spiked[i] + 1;
            end
        end
    end

    task add_conn;
        input [CORE_ID_BITS-1:0]     core;
        input [NEURON_BITS-1:0]      src;
        input [FANOUT_BITS-1:0]      slot;
        input [NEURON_BITS-1:0]      target;
        input signed [DATA_WIDTH-1:0] weight;
    begin
        @(posedge clk);
        prog_conn_we     <= 1;
        prog_conn_core   <= core;
        prog_conn_src    <= src;
        prog_conn_slot   <= slot;
        prog_conn_target <= target;
        prog_conn_weight <= weight;
        @(posedge clk);
        prog_conn_we     <= 0;
    end
    endtask

    task add_route;
        input [CORE_ID_BITS-1:0]     src_core;
        input [NEURON_BITS-1:0]      src_neuron;
        input [CORE_ID_BITS-1:0]     dest_core;
        input [NEURON_BITS-1:0]      dest_neuron;
        input signed [DATA_WIDTH-1:0] weight;
    begin
        @(posedge clk);
        prog_route_we         <= 1;
        prog_route_src_core   <= src_core;
        prog_route_src_neuron <= src_neuron;
        prog_route_dest_core  <= dest_core;
        prog_route_dest_neuron<= dest_neuron;
        prog_route_weight     <= weight;
        @(posedge clk);
        prog_route_we         <= 0;
    end
    endtask

    task run_mesh_timestep;
        input [CORE_ID_BITS-1:0]     stim_core;
        input [NEURON_BITS-1:0]      stim_neuron;
        input signed [DATA_WIDTH-1:0] stim_current;
    begin
        ext_valid     <= 1;
        ext_core      <= stim_core;
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

    task run_mesh_empty;
    begin
        @(posedge clk);
        start <= 1;
        @(posedge clk);
        start <= 0;
        wait(timestep_done);
        @(posedge clk);
    end
    endtask

    task reset_counts;
    begin
        spike_count = 0;
        for (i = 0; i < NUM_CORES; i = i + 1)
            core_spiked[i] = 0;
    end
    endtask

    integer t, pass_count, fail_count;

    initial begin
        // Init all signals
        for (i = 0; i < NUM_CORES; i = i + 1)
            core_spiked[i] = 0;
        spike_count = 0;
        pass_count  = 0;
        fail_count  = 0;
        rst_n = 0; start = 0;
        ext_valid = 0; ext_core = 0; ext_neuron_id = 0; ext_current = 0;
        prog_conn_we = 0; prog_conn_core = 0; prog_conn_src = 0;
        prog_conn_slot = 0; prog_conn_target = 0; prog_conn_weight = 0;
        prog_route_we = 0; prog_route_src_core = 0; prog_route_src_neuron = 0;
        prog_route_dest_core = 0; prog_route_dest_neuron = 0; prog_route_weight = 0;

        $display("");
        $display("================================================================");
        $display("  128-Core Neuromorphic Mesh Test (Phase 11)");
        $display("  %0d cores x %0d neurons = %0d total neurons",
                 NUM_CORES, NUM_NEURONS, NUM_CORES * NUM_NEURONS);
        $display("================================================================");

        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 5);

        $display("");
        $display("--- TEST 1: All 128 Cores Start and Complete ---");

        // Stimulate core 0 N0 and core 127 N0
        ext_valid     <= 1;
        ext_core      <= 7'd0;
        ext_neuron_id <= 8'd0;
        ext_current   <= 16'sd1200;
        @(posedge clk);
        ext_core      <= 7'd127;
        @(posedge clk);
        ext_valid     <= 0;

        spike_count = 0;

        @(posedge clk);
        start <= 1;
        @(posedge clk);
        start <= 0;

        wait(timestep_done);
        @(posedge clk);

        $display("  Timestep completed: ts=%0d, total_spikes=%0d", timestep_count, total_spikes);
        $display("  Core 0 spiked: %0d, Core 127 spiked: %0d",
                 core_spiked[0], core_spiked[127]);

        if (timestep_count == 1 && total_spikes >= 2) begin
            $display("  PASS: All 128 cores completed timestep, both endpoints spiked");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected ts=1 with >=2 spikes, got ts=%0d spikes=%0d",
                     timestep_count, total_spikes);
            fail_count = fail_count + 1;
        end

        $display("");
        $display("--- TEST 2: Far-Core Route (Core 0 -> Core 127) ---");
        reset_counts();

        // Core 0: chain N0→N1→N2→N3 (strong weights)
        add_conn(7'd0, 8'd0, 5'd0, 8'd1, 16'sd1200);
        add_conn(7'd0, 8'd1, 5'd0, 8'd2, 16'sd1200);
        add_conn(7'd0, 8'd2, 5'd0, 8'd3, 16'sd1200);

        // Inter-core route: Core 0 N3 → Core 127 N0
        add_route(7'd0, 8'd3, 7'd127, 8'd0, 16'sd1200);

        // Core 127: chain N0→N1
        add_conn(7'd127, 8'd0, 5'd0, 8'd1, 16'sd1200);

        $display("  Running 20 timesteps with stimulus to Core 0 N0...");

        for (t = 0; t < 20; t = t + 1) begin
            run_mesh_timestep(7'd0, 8'd0, 16'sd200);
        end

        $display("  Core 0 spikes: %0d", core_spiked[0]);
        $display("  Core 127 spikes: %0d", core_spiked[127]);

        if (core_spiked[127] > 0) begin
            $display("  PASS: Spike propagated from Core 0 to Core 127!");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: No spikes reached Core 127");
            fail_count = fail_count + 1;
        end

        $display("");
        $display("--- TEST 3: Multi-Hop Chain (0 -> 42 -> 85 -> 127) ---");
        reset_counts();

        // Core 42: N0→N1→N2→N3
        add_conn(7'd42, 8'd0, 5'd0, 8'd1, 16'sd1200);
        add_conn(7'd42, 8'd1, 5'd0, 8'd2, 16'sd1200);
        add_conn(7'd42, 8'd2, 5'd0, 8'd3, 16'sd1200);

        // Route: Core 0 N3 → Core 42 N0 (already programmed in test 2? no, route table is keyed by {src_core, src_neuron})
        // Use N4-N7 chain on core 0 for this test to avoid conflicts.
        add_conn(7'd0, 8'd4, 5'd0, 8'd5, 16'sd1200);
        add_conn(7'd0, 8'd5, 5'd0, 8'd6, 16'sd1200);
        add_conn(7'd0, 8'd6, 5'd0, 8'd7, 16'sd1200);

        // Route: Core 0 N7 → Core 42 N0
        add_route(7'd0, 8'd7, 7'd42, 8'd0, 16'sd1200);

        // Route: Core 42 N3 → Core 85 N0
        add_route(7'd42, 8'd3, 7'd85, 8'd0, 16'sd1200);

        // Core 85: N0→N1→N2→N3
        add_conn(7'd85, 8'd0, 5'd0, 8'd1, 16'sd1200);
        add_conn(7'd85, 8'd1, 5'd0, 8'd2, 16'sd1200);
        add_conn(7'd85, 8'd2, 5'd0, 8'd3, 16'sd1200);

        // Route: Core 85 N3 → Core 127 N2 (use N2 to avoid conflict with test 2)
        add_route(7'd85, 8'd3, 7'd127, 8'd2, 16'sd1200);

        // Core 127: N2→N3
        add_conn(7'd127, 8'd2, 5'd0, 8'd3, 16'sd1200);

        $display("  Running 60 timesteps with stimulus to Core 0 N4...");

        for (t = 0; t < 60; t = t + 1) begin
            run_mesh_timestep(7'd0, 8'd4, 16'sd200);
        end

        $display("  Core 0 spikes:   %0d", core_spiked[0]);
        $display("  Core 42 spikes:  %0d", core_spiked[42]);
        $display("  Core 85 spikes:  %0d", core_spiked[85]);
        $display("  Core 127 spikes: %0d", core_spiked[127]);

        if (core_spiked[42] > 0 && core_spiked[85] > 0 && core_spiked[127] > 0) begin
            $display("  PASS: Spike traversed all 3 hops (0->42->85->127)!");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Chain incomplete (C42=%0d, C85=%0d, C127=%0d)",
                     core_spiked[42], core_spiked[85], core_spiked[127]);
            fail_count = fail_count + 1;
        end

        $display("");
        $display("================================================================");
        $display("  128-CORE TEST RESULTS: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("================================================================");
        $display("  Architecture: %0d cores x %0d neurons = %0d total",
                 NUM_CORES, NUM_NEURONS, NUM_CORES * NUM_NEURONS);
        $display("  Total timesteps: %0d", timestep_count);
        $display("  Total spikes:    %0d", total_spikes);
        if (fail_count == 0)
            $display("  ALL TESTS PASSED");
        else
            $display("  SOME TESTS FAILED");
        $display("================================================================");

        #(CLK_PERIOD * 10);
        $finish;
    end

    initial begin
        #(CLK_PERIOD * 50_000_000);
        $display("TIMEOUT at mesh_state=%0d, ts=%0d", mesh_state_out, timestep_count);
        $finish;
    end

endmodule
