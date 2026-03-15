// ============================================================================
// Testbench: Neuromorphic Mesh (4 cores × 256 neurons = 1024 neurons)
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

module tb_neuromorphic_mesh;

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
                $display("  [t=%0d] Core %0d Neuron %0d spiked!",
                    timestep_count, i, spike_id_bus[i*NEURON_BITS +: NEURON_BITS]);
            end
        end
    end

    initial begin
        $dumpfile("neuromorphic_mesh.vcd");
        $dumpvars(0, tb_neuromorphic_mesh);
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
        for (i = 0; i < NUM_CORES; i = i + 1) begin
            core_spike_total[i] = 0;
            for (j = 0; j < NUM_NEURONS; j = j + 1)
                spike_count[i][j] = 0;
        end
    end
    endtask

    integer t;
    initial begin
        // Init all signals
        for (i = 0; i < NUM_CORES; i = i + 1) begin
            core_spike_total[i] = 0;
            for (j = 0; j < NUM_NEURONS; j = j + 1)
                spike_count[i][j] = 0;
        end
        rst_n = 0; start = 0;
        ext_valid = 0; ext_core = 0; ext_neuron_id = 0; ext_current = 0;
        prog_conn_we = 0; prog_conn_core = 0; prog_conn_src = 0;
        prog_conn_slot = 0; prog_conn_target = 0; prog_conn_weight = 0;
        prog_route_we = 0; prog_route_src_core = 0; prog_route_src_neuron = 0;
        prog_route_dest_core = 0; prog_route_dest_neuron = 0; prog_route_weight = 0;

        $display("");
        $display("================================================================");
        $display("  Neuromorphic Mesh Test - 4 Cores x 256 Neurons = 1024 Total");
        $display("================================================================");

        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 5);

        $display("");
        $display("--- TEST 1: Cross-Core Chain (Core 0 -> Core 1) ---");
        $display("  Programming intra-core chains + inter-core route...");

        // Core 0: chain 0→1→2→3 (strong weights for instant propagation)
        add_conn(0, 0, 0, 1, 16'sd1200);
        add_conn(0, 1, 0, 2, 16'sd1200);
        add_conn(0, 2, 0, 3, 16'sd1200);

        // Inter-core route: Core 0 neuron 3 → Core 1 neuron 0
        add_route(0, 3, 1, 0, 16'sd1200);

        // Core 1: chain 0→1→2→3
        add_conn(1, 0, 0, 1, 16'sd1200);
        add_conn(1, 1, 0, 2, 16'sd1200);
        add_conn(1, 2, 0, 3, 16'sd1200);

        $display("  Running 30 timesteps with stimulus to Core 0 N0...");

        for (t = 0; t < 30; t = t + 1) begin
            run_mesh_timestep(0, 0, 16'sd200);
        end

        $display("");
        $display("  Cross-core chain results:");
        $display("  Core 0:");
        for (i = 0; i < 4; i = i + 1)
            $display("    N%0d: %0d spikes", i, spike_count[0][i]);
        $display("  Core 1:");
        for (i = 0; i < 4; i = i + 1)
            $display("    N%0d: %0d spikes", i, spike_count[1][i]);
        $display("  Core 2 total: %0d (should be 0)", core_spike_total[2]);
        $display("  Core 3 total: %0d (should be 0)", core_spike_total[3]);

        $display("");
        $display("--- TEST 2: Full 4-Core Chain (0->1->2->3) ---");
        $display("  Programming inter-core routes + intra-core chains...");
        reset_counts();

        // Route: Core 1 N3 → Core 2 N0
        add_route(1, 3, 2, 0, 16'sd1200);

        // Core 2: chain 0→1→2→3
        add_conn(2, 0, 0, 1, 16'sd1200);
        add_conn(2, 1, 0, 2, 16'sd1200);
        add_conn(2, 2, 0, 3, 16'sd1200);

        // Route: Core 2 N3 → Core 3 N0
        add_route(2, 3, 3, 0, 16'sd1200);

        // Core 3: chain 0→1→2→3
        add_conn(3, 0, 0, 1, 16'sd1200);
        add_conn(3, 1, 0, 2, 16'sd1200);
        add_conn(3, 2, 0, 3, 16'sd1200);

        $display("  Running 60 timesteps with stimulus to Core 0 N0...");

        for (t = 0; t < 60; t = t + 1) begin
            run_mesh_timestep(0, 0, 16'sd200);
        end

        $display("");
        $display("  Full 4-core chain results:");
        for (i = 0; i < NUM_CORES; i = i + 1) begin
            $display("  Core %0d:", i);
            for (j = 0; j < 4; j = j + 1)
                $display("    N%0d: %0d spikes", j, spike_count[i][j]);
        end

        $display("");
        $display("================================================================");
        $display("  FINAL REPORT");
        $display("================================================================");
        $display("  Total timesteps: %0d", timestep_count);
        $display("  Total spikes:    %0d", total_spikes);
        $display("  Architecture:    %0d cores x %0d neurons = %0d total",
                 NUM_CORES, NUM_NEURONS, NUM_CORES * NUM_NEURONS);
        $display("  Sparse intra-core: max %0d fanout per neuron", MAX_FANOUT);
        $display("  Inter-core NoC:    route table (%0d entries)",
                 NUM_CORES * NUM_NEURONS);
        $display("================================================================");

        #(CLK_PERIOD * 10);
        $finish;
    end

    reg [4:0] prev_mesh_state;
    always @(posedge clk) begin
        if (mesh_state_out != prev_mesh_state) begin
            if (timestep_count < 3)
                $display("  [dbg] Mesh: %0d -> %0d (ts=%0d)",
                    prev_mesh_state, mesh_state_out, timestep_count);
            prev_mesh_state <= mesh_state_out;
        end
    end
    initial prev_mesh_state = 0;

    initial begin
        #(CLK_PERIOD * 2000000);
        $display("TIMEOUT at mesh_state=%0d, ts=%0d", mesh_state_out, timestep_count);
        $finish;
    end

endmodule
