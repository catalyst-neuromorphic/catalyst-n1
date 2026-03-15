// ============================================================================
// Testbench: Async Event-Driven Mode (Phase 12)
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

module tb_async;

    parameter NUM_CORES      = 4;
    parameter CORE_ID_BITS   = 2;
    parameter NUM_NEURONS    = 256;
    parameter NEURON_BITS    = 8;
    parameter DATA_WIDTH     = 16;
    parameter MAX_FANOUT     = 32;
    parameter FANOUT_BITS    = 5;
    parameter CONN_ADDR_BITS = 13;
    parameter CLK_PERIOD     = 10;

    reg                          clk, rst_n;
    reg                          start;
    reg                          async_enable;

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

    integer spike_count [0:NUM_CORES-1][0:NUM_NEURONS-1];
    integer core_spike_total [0:NUM_CORES-1];
    integer i, j;

    integer pass_count;
    integer fail_count;

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
        .async_enable      (async_enable),
        .prog_conn_comp    (2'd0),
        .prog_param_we     (1'b0),
        .prog_param_core   (2'd0),
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
                spike_count[i][spike_id_bus[i*NEURON_BITS +: NEURON_BITS]] =
                    spike_count[i][spike_id_bus[i*NEURON_BITS +: NEURON_BITS]] + 1;
                core_spike_total[i] = core_spike_total[i] + 1;
            end
        end
    end

    initial begin
        $dumpfile("async_mode.vcd");
        $dumpvars(0, tb_async);
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

    task apply_stimulus;
        input [CORE_ID_BITS-1:0]     stim_core;
        input [NEURON_BITS-1:0]      stim_neuron;
        input signed [DATA_WIDTH-1:0] stim_current;
    begin
        @(posedge clk);
        ext_valid     <= 1;
        ext_core      <= stim_core;
        ext_neuron_id <= stim_neuron;
        ext_current   <= stim_current;
        @(posedge clk);
        ext_valid     <= 0;
    end
    endtask

    task run_and_wait;
    begin
        @(posedge clk);
        start <= 1;
        @(posedge clk);
        start <= 0;
        wait(timestep_done);
        @(posedge clk);
    end
    endtask

    task run_sync_timestep;
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

    task reset_counts;
    begin
        for (i = 0; i < NUM_CORES; i = i + 1) begin
            core_spike_total[i] = 0;
            for (j = 0; j < NUM_NEURONS; j = j + 1)
                spike_count[i][j] = 0;
        end
    end
    endtask

    integer t;
    integer sync_spikes_total;
    integer async_spikes_total;
    integer cycle_start, cycle_end;
    initial begin
        pass_count = 0;
        fail_count = 0;
        for (i = 0; i < NUM_CORES; i = i + 1) begin
            core_spike_total[i] = 0;
            for (j = 0; j < NUM_NEURONS; j = j + 1)
                spike_count[i][j] = 0;
        end
        rst_n = 0; start = 0; async_enable = 0;
        ext_valid = 0; ext_core = 0; ext_neuron_id = 0; ext_current = 0;
        prog_conn_we = 0; prog_conn_core = 0; prog_conn_src = 0;
        prog_conn_slot = 0; prog_conn_target = 0; prog_conn_weight = 0;
        prog_route_we = 0; prog_route_src_core = 0; prog_route_src_neuron = 0;
        prog_route_dest_core = 0; prog_route_dest_neuron = 0; prog_route_weight = 0;

        $display("");
        $display("================================================================");
        $display("  Phase 12: Async Event-Driven Mode Test");
        $display("  %0d cores x %0d neurons, GALS architecture", NUM_CORES, NUM_NEURONS);
        $display("================================================================");

        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 5);

        $display("");
        $display("--- TEST 1: Basic Event Propagation (Async) ---");

        // Core 0: N0→N1 intra-core chain
        add_conn(0, 0, 0, 1, 16'sd1200);
        // Inter-core route: Core 0 N1 → Core 1 N0
        add_route(0, 1, 1, 0, 16'sd1200);
        // Core 1: N0→N1 intra-core chain
        add_conn(1, 0, 0, 1, 16'sd1200);

        // Enable async mode
        async_enable <= 1;
        @(posedge clk);

        // Apply stimulus to Core 0 N0 (goes to pcif[0])
        apply_stimulus(0, 0, 16'sd1200);

        // Run async and wait for quiescence
        run_and_wait;

        $display("  Core 0: N0=%0d spikes, N1=%0d spikes", spike_count[0][0], spike_count[0][1]);
        $display("  Core 1: N0=%0d spikes, N1=%0d spikes", spike_count[1][0], spike_count[1][1]);

        if (spike_count[0][0] >= 1 && spike_count[0][1] >= 1 &&
            spike_count[1][0] >= 1 && spike_count[1][1] >= 1) begin
            $display("  PASS: Spike propagated Core 0 -> Core 1 in async mode!");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected spikes on both cores");
            fail_count = fail_count + 1;
        end

        $display("");
        $display("--- TEST 2: Multi-Hop Async (0->1->2->3) ---");

        async_enable <= 0;
        @(posedge clk);
        rst_n <= 0;
        #(CLK_PERIOD * 3);
        rst_n <= 1;
        #(CLK_PERIOD * 5);
        reset_counts;

        // Build 4-core chain using N10-N12 (fresh neurons, no stale SRAM from Test 1)
        // Core 0: N10→N11→N12
        add_conn(0, 10, 0, 11, 16'sd1200);
        add_conn(0, 11, 0, 12, 16'sd1200);
        // Route: C0:N12 → C1:N10
        add_route(0, 12, 1, 10, 16'sd1200);
        // Core 1: N10→N11→N12
        add_conn(1, 10, 0, 11, 16'sd1200);
        add_conn(1, 11, 0, 12, 16'sd1200);
        // Route: C1:N12 → C2:N10
        add_route(1, 12, 2, 10, 16'sd1200);
        // Core 2: N10→N11→N12
        add_conn(2, 10, 0, 11, 16'sd1200);
        add_conn(2, 11, 0, 12, 16'sd1200);
        // Route: C2:N12 → C3:N10
        add_route(2, 12, 3, 10, 16'sd1200);
        // Core 3: N10→N11
        add_conn(3, 10, 0, 11, 16'sd1200);

        async_enable <= 1;
        @(posedge clk);

        // Stimulus to fresh neuron N10
        apply_stimulus(0, 10, 16'sd1200);

        run_and_wait;

        $display("  Core 0: total=%0d spikes", core_spike_total[0]);
        $display("  Core 1: total=%0d spikes", core_spike_total[1]);
        $display("  Core 2: total=%0d spikes", core_spike_total[2]);
        $display("  Core 3: total=%0d spikes", core_spike_total[3]);

        if (core_spike_total[0] >= 1 && core_spike_total[1] >= 1 &&
            core_spike_total[2] >= 1 && core_spike_total[3] >= 1) begin
            $display("  PASS: Multi-hop spike traversed all 4 cores!");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected spikes on all 4 cores");
            fail_count = fail_count + 1;
        end

        $display("");
        $display("--- TEST 3: Quiescence Detection ---");

        async_enable <= 0;
        @(posedge clk);
        rst_n <= 0;
        #(CLK_PERIOD * 3);
        rst_n <= 1;
        #(CLK_PERIOD * 5);
        reset_counts;

        // Simple: Core 0 N20 only (fresh neuron, no stale connections/routes)
        // No intra-core connections - just one neuron fires from stimulus

        async_enable <= 1;
        @(posedge clk);

        // Apply stimulus to fresh neuron N20
        apply_stimulus(0, 20, 16'sd1200);

        // Capture cycle count
        cycle_start = $time;

        run_and_wait;

        cycle_end = $time;

        $display("  Quiescence reached in %0d ns", cycle_end - cycle_start);
        $display("  Core 0 N20 spikes: %0d", spike_count[0][20]);
        $display("  Core 1 total: %0d (should be 0)", core_spike_total[1]);

        if (spike_count[0][20] >= 1 && core_spike_total[1] == 0 &&
            core_spike_total[2] == 0 && core_spike_total[3] == 0) begin
            $display("  PASS: Quiescence detected correctly (isolated stimulus)!");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Unexpected spike pattern");
            fail_count = fail_count + 1;
        end

        $display("");
        $display("--- TEST 4: Async vs Sync Equivalence ---");

        async_enable <= 0;
        @(posedge clk);
        rst_n <= 0;
        #(CLK_PERIOD * 3);
        rst_n <= 1;
        #(CLK_PERIOD * 5);
        reset_counts;

        // Build network: Core 0 N30→N31, route C0:N31→C1:N30, Core 1 N30→N31
        add_conn(0, 30, 0, 31, 16'sd1200);
        add_route(0, 31, 1, 30, 16'sd1200);
        add_conn(1, 30, 0, 31, 16'sd1200);

        $display("  Part A: Running in SYNC mode (10 timesteps, N30/N31)...");
        async_enable <= 0;
        @(posedge clk);

        for (t = 0; t < 10; t = t + 1) begin
            run_sync_timestep(0, 30, 16'sd200);
        end

        sync_spikes_total = 0;
        for (i = 0; i < NUM_CORES; i = i + 1)
            sync_spikes_total = sync_spikes_total + core_spike_total[i];

        $display("  Sync total spikes: %0d", sync_spikes_total);
        $display("    Core 0: N30=%0d, N31=%0d", spike_count[0][30], spike_count[0][31]);
        $display("    Core 1: N30=%0d, N31=%0d", spike_count[1][30], spike_count[1][31]);

        // Reset to clear FSMs/FIFOs (SRAMs retain, but N40/N41 are pristine)
        rst_n <= 0;
        #(CLK_PERIOD * 3);
        rst_n <= 1;
        #(CLK_PERIOD * 5);
        reset_counts;

        // Same topology but using N40/N41 (fresh neurons, identical initial state)
        add_conn(0, 40, 0, 41, 16'sd1200);
        add_route(0, 41, 1, 40, 16'sd1200);
        add_conn(1, 40, 0, 41, 16'sd1200);

        $display("  Part B: Running in ASYNC mode (10 async runs, N40/N41)...");
        async_enable <= 1;
        @(posedge clk);

        for (t = 0; t < 10; t = t + 1) begin
            apply_stimulus(0, 40, 16'sd200);
            run_and_wait;
        end

        async_spikes_total = 0;
        for (i = 0; i < NUM_CORES; i = i + 1)
            async_spikes_total = async_spikes_total + core_spike_total[i];

        $display("  Async total spikes: %0d", async_spikes_total);
        $display("    Core 0: N40=%0d, N41=%0d", spike_count[0][40], spike_count[0][41]);
        $display("    Core 1: N40=%0d, N41=%0d", spike_count[1][40], spike_count[1][41]);

        if (sync_spikes_total == async_spikes_total) begin
            $display("  PASS: Sync and async produced identical spike counts (%0d)!", sync_spikes_total);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Spike count mismatch (sync=%0d, async=%0d)", sync_spikes_total, async_spikes_total);
            fail_count = fail_count + 1;
        end

        $display("");
        $display("================================================================");
        $display("  RESULTS: %0d/%0d PASSED", pass_count, pass_count + fail_count);
        if (fail_count == 0)
            $display("  ALL TESTS PASSED!");
        else
            $display("  %0d TESTS FAILED!", fail_count);
        $display("================================================================");

        #(CLK_PERIOD * 10);
        $finish;
    end


    initial begin
        #(CLK_PERIOD * 5000000);
        $display("TIMEOUT at mesh_state=%0d, ts=%0d", mesh_state_out, timestep_count);
        $finish;
    end

endmodule
