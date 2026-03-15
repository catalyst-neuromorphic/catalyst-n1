// ============================================================================
// Stress Test: Long-running stability and cross-core propagation
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

module tb_stress;

    parameter NUM_CORES      = 4;
    parameter CORE_ID_BITS   = 2;
    parameter NUM_NEURONS    = 256;
    parameter NEURON_BITS    = 8;
    parameter DATA_WIDTH     = 16;
    parameter POOL_DEPTH     = 256;
    parameter POOL_ADDR_BITS = 8;
    parameter COUNT_BITS     = 10;
    parameter REV_FANIN      = 32;
    parameter REV_SLOT_BITS  = 5;
    parameter ROUTE_FANOUT   = 8;
    parameter ROUTE_SLOT_BITS= 3;
    parameter CLK_PERIOD     = 10;

    reg clk, rst_n;
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;

    reg                         start;
    reg                         prog_pool_we;
    reg  [CORE_ID_BITS-1:0]    prog_pool_core;
    reg  [POOL_ADDR_BITS-1:0]  prog_pool_addr;
    reg  [NEURON_BITS-1:0]     prog_pool_src;
    reg  [NEURON_BITS-1:0]     prog_pool_target;
    reg  signed [DATA_WIDTH-1:0] prog_pool_weight;
    reg  [1:0]                  prog_pool_comp;
    reg                         prog_index_we;
    reg  [CORE_ID_BITS-1:0]    prog_index_core;
    reg  [NEURON_BITS-1:0]     prog_index_neuron;
    reg  [POOL_ADDR_BITS-1:0]  prog_index_base;
    reg  [COUNT_BITS-1:0]      prog_index_count;
    reg                         prog_route_we;
    reg  [CORE_ID_BITS-1:0]    prog_route_src_core;
    reg  [NEURON_BITS-1:0]     prog_route_src_neuron;
    reg  [ROUTE_SLOT_BITS-1:0] prog_route_slot;
    reg  [CORE_ID_BITS-1:0]    prog_route_dest_core;
    reg  [NEURON_BITS-1:0]     prog_route_dest_neuron;
    reg  signed [DATA_WIDTH-1:0] prog_route_weight;
    reg                         prog_param_we;
    reg  [CORE_ID_BITS-1:0]    prog_param_core;
    reg  [NEURON_BITS-1:0]     prog_param_neuron;
    reg  [4:0]                  prog_param_id;
    reg  signed [DATA_WIDTH-1:0] prog_param_value;
    reg                         ext_valid;
    reg  [CORE_ID_BITS-1:0]    ext_core;
    reg  [NEURON_BITS-1:0]     ext_neuron_id;
    reg  signed [DATA_WIDTH-1:0] ext_current;

    wire                        timestep_done;
    wire [31:0]                 total_spikes;
    wire [31:0]                 timestep_count;

    neuromorphic_mesh #(
        .NUM_CORES      (NUM_CORES),
        .CORE_ID_BITS   (CORE_ID_BITS),
        .NUM_NEURONS    (NUM_NEURONS),
        .NEURON_BITS    (NEURON_BITS),
        .DATA_WIDTH     (DATA_WIDTH),
        .POOL_DEPTH     (POOL_DEPTH),
        .POOL_ADDR_BITS (POOL_ADDR_BITS),
        .COUNT_BITS     (COUNT_BITS),
        .REV_FANIN      (REV_FANIN),
        .REV_SLOT_BITS  (REV_SLOT_BITS),
        .ROUTE_FANOUT   (ROUTE_FANOUT),
        .ROUTE_SLOT_BITS(ROUTE_SLOT_BITS),
        .THRESHOLD      (16'sd1000),
        .LEAK_RATE      (16'sd3),
        .REFRAC_CYCLES  (3)
    ) dut (
        .clk               (clk),
        .rst_n             (rst_n),
        .start             (start),
        .prog_pool_we      (prog_pool_we),
        .prog_pool_core    (prog_pool_core),
        .prog_pool_addr    (prog_pool_addr),
        .prog_pool_src     (prog_pool_src),
        .prog_pool_target  (prog_pool_target),
        .prog_pool_weight  (prog_pool_weight),
        .prog_pool_comp    (prog_pool_comp),
        .prog_index_we     (prog_index_we),
        .prog_index_core   (prog_index_core),
        .prog_index_neuron (prog_index_neuron),
        .prog_index_base   (prog_index_base),
        .prog_index_count  (prog_index_count),
        .prog_index_format (2'd0),
        .prog_route_we         (prog_route_we),
        .prog_route_src_core   (prog_route_src_core),
        .prog_route_src_neuron (prog_route_src_neuron),
        .prog_route_slot       (prog_route_slot),
        .prog_route_dest_core  (prog_route_dest_core),
        .prog_route_dest_neuron(prog_route_dest_neuron),
        .prog_route_weight     (prog_route_weight),
        .prog_global_route_we(1'b0),
        .prog_global_route_src_core({CORE_ID_BITS{1'b0}}),
        .prog_global_route_src_neuron({NEURON_BITS{1'b0}}),
        .prog_global_route_slot(2'b0),
        .prog_global_route_dest_core({CORE_ID_BITS{1'b0}}),
        .prog_global_route_dest_neuron({NEURON_BITS{1'b0}}),
        .prog_global_route_weight({DATA_WIDTH{1'b0}}),
        .learn_enable      (1'b0),
        .graded_enable     (1'b0),
        .dendritic_enable  (1'b0),
        .async_enable      (1'b0),
        .threefactor_enable(1'b0),
        .noise_enable      (1'b0),
        .skip_idle_enable  (1'b0),
        .scale_u_enable    (1'b0),
        .reward_value      (16'd0),
        .prog_delay_we     (1'b0),
        .prog_delay_core   ({CORE_ID_BITS{1'b0}}),
        .prog_delay_addr   ({POOL_ADDR_BITS{1'b0}}),
        .prog_delay_value  (6'd0),
        .prog_ucode_we     (1'b0),
        .prog_ucode_core   ({CORE_ID_BITS{1'b0}}),
        .prog_ucode_addr   (8'd0),
        .prog_ucode_data   (32'd0),
        .prog_param_we     (prog_param_we),
        .prog_param_core   (prog_param_core),
        .prog_param_neuron (prog_param_neuron),
        .prog_param_id     (prog_param_id),
        .prog_param_value  (prog_param_value),
        .ext_valid         (ext_valid),
        .ext_core          (ext_core),
        .ext_neuron_id     (ext_neuron_id),
        .ext_current       (ext_current),
        .probe_read        (1'b0),
        .probe_core        ({CORE_ID_BITS{1'b0}}),
        .probe_neuron      ({NEURON_BITS{1'b0}}),
        .probe_state_id    (5'd0),
        .probe_pool_addr   ({POOL_ADDR_BITS{1'b0}}),
        .probe_data        (),
        .probe_valid       (),
        .timestep_done     (timestep_done),
        .spike_valid_bus   (),
        .spike_id_bus      (),
        .mesh_state_out    (),
        .total_spikes      (total_spikes),
        .timestep_count    (timestep_count),
        .core_idle_bus     (),
        .dvfs_stall        (8'd0),
        .core_clock_en     (),
        .energy_counter    (),
        .power_idle_hint   (),
        .link_tx_push      (),
        .link_tx_core      (),
        .link_tx_neuron    (),
        .link_tx_payload   (),
        .link_tx_full      (1'b0),
        .link_rx_core      ({CORE_ID_BITS{1'b0}}),
        .link_rx_neuron    ({NEURON_BITS{1'b0}}),
        .link_rx_current   ({DATA_WIDTH{1'b0}}),
        .link_rx_pop       (),
        .link_rx_empty     (1'b1)
    );


    task set_param(input [CORE_ID_BITS-1:0] core, input [NEURON_BITS-1:0] neuron,
                   input [4:0] pid, input [DATA_WIDTH-1:0] value);
        begin
            @(posedge clk);
            prog_param_we     <= 1;
            prog_param_core   <= core;
            prog_param_neuron <= neuron;
            prog_param_id     <= pid;
            prog_param_value  <= value;
            @(posedge clk);
            prog_param_we     <= 0;
            @(posedge clk);
        end
    endtask

    task setup_neuron(input [CORE_ID_BITS-1:0] core, input [NEURON_BITS-1:0] neuron,
                      input [DATA_WIDTH-1:0] threshold);
        begin
            set_param(core, neuron, 5'd0, threshold);
            set_param(core, neuron, 5'd22, {NEURON_BITS{1'b1}});   // parent_ptr sentinel
            set_param(core, neuron, 5'd24, 16'd1);                 // is_root=1
        end
    endtask

    task inject_stim(input [CORE_ID_BITS-1:0] core, input [NEURON_BITS-1:0] neuron,
                     input signed [DATA_WIDTH-1:0] current);
        begin
            @(posedge clk);
            ext_valid     <= 1;
            ext_core      <= core;
            ext_neuron_id <= neuron;
            ext_current   <= current;
            @(posedge clk);
            ext_valid     <= 0;
            @(posedge clk);
        end
    endtask

    task run_one_ts;
        begin
            @(posedge clk);
            start <= 1;
            @(posedge clk);
            start <= 0;
            wait(timestep_done);
            @(posedge clk);
        end
    endtask

    task program_route(input [CORE_ID_BITS-1:0] sc, input [NEURON_BITS-1:0] sn,
                       input [ROUTE_SLOT_BITS-1:0] slot,
                       input [CORE_ID_BITS-1:0] dc, input [NEURON_BITS-1:0] dn,
                       input signed [DATA_WIDTH-1:0] w);
        begin
            @(posedge clk);
            prog_route_we          <= 1;
            prog_route_src_core    <= sc;
            prog_route_src_neuron  <= sn;
            prog_route_slot        <= slot;
            prog_route_dest_core   <= dc;
            prog_route_dest_neuron <= dn;
            prog_route_weight      <= w;
            @(posedge clk);
            prog_route_we          <= 0;
            @(posedge clk);
        end
    endtask

    integer ts;
    reg [31:0] saved_spikes;

    initial begin
        start = 0;
        prog_pool_we = 0; prog_index_we = 0; prog_route_we = 0; prog_param_we = 0;
        ext_valid = 0;
        rst_n = 0;
        #100;
        rst_n = 1;
        #50;

        $display("Test 1: Single neuron, 100 timestep stability");
        setup_neuron(0, 0, 16'd100);
        set_param(0, 0, 5'd1, 16'd0);   // leak=0
        set_param(0, 0, 5'd3, 16'd0);   // refrac=0

        for (ts = 0; ts < 100; ts = ts + 1) begin
            inject_stim(0, 0, 16'sd200);
            run_one_ts;
        end

        $display("  Spikes: %0d", total_spikes);
        if (total_spikes >= 90) begin
            $display("  PASSED: %0d spikes in 100 ts", total_spikes);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: expected >= 90, got %0d", total_spikes);
            fail_count = fail_count + 1;
        end

        $display("Test 2: 4-core chain propagation");
        rst_n = 0; #50; rst_n = 1; #50;

        setup_neuron(0, 0, 16'd100);
        setup_neuron(1, 0, 16'd100);
        setup_neuron(2, 0, 16'd100);
        setup_neuron(3, 0, 16'd100);

        set_param(0, 0, 5'd1, 0); set_param(0, 0, 5'd3, 0);
        set_param(1, 0, 5'd1, 0); set_param(1, 0, 5'd3, 0);
        set_param(2, 0, 5'd1, 0); set_param(2, 0, 5'd3, 0);
        set_param(3, 0, 5'd1, 0); set_param(3, 0, 5'd3, 0);

        // Route chain: core0->core1->core2->core3
        program_route(0, 0, 0,  1, 0, 16'sd200);
        program_route(1, 0, 0,  2, 0, 16'sd200);
        program_route(2, 0, 0,  3, 0, 16'sd200);

        // Inject stimulus to core 0 neuron 0 (enough to spike)
        inject_stim(0, 0, 16'sd200);

        // Run enough timesteps for chain propagation (need ~4 for 4 hops)
        for (ts = 0; ts < 10; ts = ts + 1) begin
            run_one_ts;
        end

        $display("  Spikes through 4-core chain: %0d", total_spikes);
        if (total_spikes >= 4) begin
            $display("  PASSED: chain propagated (%0d spikes)", total_spikes);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: expected >= 4, got %0d", total_spikes);
            fail_count = fail_count + 1;
        end

        $display("=== STRESS RESULTS: %0d passed, %0d failed out of %0d ===",
                 pass_count, fail_count, pass_count + fail_count);
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        $finish;
    end

    initial begin
        #500000000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
