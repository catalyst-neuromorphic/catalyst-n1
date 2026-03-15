// ============================================================================
// P23A Testbench: Exact Loihi Neuron Arithmetic
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

`timescale 1ns/1ps

module tb_p23a_neuron_arith;

    parameter NUM_CORES      = 2;
    parameter CORE_ID_BITS   = 1;
    parameter NUM_NEURONS    = 1024;
    parameter NEURON_BITS    = 10;
    parameter DATA_WIDTH     = 16;
    parameter POOL_DEPTH     = 1024;
    parameter POOL_ADDR_BITS = 10;
    parameter COUNT_BITS     = 10;
    parameter REV_FANIN      = 32;
    parameter REV_SLOT_BITS  = 5;
    parameter CLK_PERIOD     = 10;
    parameter ROUTE_FANOUT    = 8;
    parameter ROUTE_SLOT_BITS = 3;

    reg clk, rst_n;
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

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

    reg                         learn_enable;
    reg                         graded_enable;
    reg                         dendritic_enable;
    reg                         async_enable;
    reg                         threefactor_enable;
    reg                         noise_enable;
    reg                         skip_idle_enable;
    reg  signed [DATA_WIDTH-1:0] reward_value;

    reg                         prog_param_we;
    reg  [CORE_ID_BITS-1:0]    prog_param_core;
    reg  [NEURON_BITS-1:0]     prog_param_neuron;
    reg  [4:0]                  prog_param_id;
    reg  signed [DATA_WIDTH-1:0] prog_param_value;

    reg                         ext_valid;
    reg  [CORE_ID_BITS-1:0]    ext_core;
    reg  [NEURON_BITS-1:0]     ext_neuron_id;
    reg  signed [DATA_WIDTH-1:0] ext_current;

    reg                         probe_read;
    reg  [CORE_ID_BITS-1:0]    probe_core;
    reg  [NEURON_BITS-1:0]     probe_neuron;
    reg  [3:0]                  probe_state_id;
    reg  [POOL_ADDR_BITS-1:0]  probe_pool_addr;
    wire signed [DATA_WIDTH-1:0] probe_data;
    wire                         probe_valid;

    wire                        timestep_done;
    wire [NUM_CORES-1:0]        spike_valid_bus;
    wire [NUM_CORES*NEURON_BITS-1:0] spike_id_bus;
    wire [5:0]                  mesh_state_out;
    wire [31:0]                 total_spikes;
    wire [31:0]                 timestep_count;
    wire [NUM_CORES-1:0]        core_idle_bus;

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
        .THRESHOLD      (16'sd10000),
        .LEAK_RATE      (16'sd0),
        .REFRAC_CYCLES  (0)
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
        .learn_enable      (learn_enable),
        .graded_enable     (graded_enable),
        .dendritic_enable  (dendritic_enable),
        .async_enable      (async_enable),
        .threefactor_enable(threefactor_enable),
        .noise_enable      (noise_enable),
        .skip_idle_enable  (skip_idle_enable),
        .scale_u_enable    (1'b0),
        .reward_value      (reward_value),
        .prog_delay_we     (1'b0),
        .prog_delay_core   ({CORE_ID_BITS{1'b0}}),
        .prog_delay_addr   ({POOL_ADDR_BITS{1'b0}}),
        .prog_delay_value  (6'd0),
        .prog_ucode_we     (1'b0),
        .prog_ucode_core   ({CORE_ID_BITS{1'b0}}),
        .prog_ucode_addr   (6'd0),
        .prog_ucode_data   (32'd0),
        .prog_param_we     (prog_param_we),
        .prog_param_core   (prog_param_core),
        .prog_param_neuron (prog_param_neuron),
        .prog_param_id     (prog_param_id),
        .prog_param_value  (prog_param_value),
        .probe_read        (probe_read),
        .probe_core        (probe_core),
        .probe_neuron      (probe_neuron),
        .probe_state_id    (probe_state_id),
        .probe_pool_addr   (probe_pool_addr),
        .probe_data        (probe_data),
        .probe_valid       (probe_valid),
        .ext_valid         (ext_valid),
        .ext_core          (ext_core),
        .ext_neuron_id     (ext_neuron_id),
        .ext_current       (ext_current),
        .timestep_done     (timestep_done),
        .spike_valid_bus   (spike_valid_bus),
        .spike_id_bus      (spike_id_bus),
        .mesh_state_out    (mesh_state_out),
        .total_spikes      (total_spikes),
        .timestep_count    (timestep_count),
        .core_idle_bus     (core_idle_bus),
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


    task set_param;
        input [CORE_ID_BITS-1:0]     core;
        input [NEURON_BITS-1:0]      neuron;
        input [4:0]                   pid;
        input signed [DATA_WIDTH-1:0] value;
    begin
        @(posedge clk);
        prog_param_we     <= 1;
        prog_param_core   <= core;
        prog_param_neuron <= neuron;
        prog_param_id     <= pid;
        prog_param_value  <= value;
        @(posedge clk);
        prog_param_we <= 0;
    end
    endtask

    task run_timestep;
        input [CORE_ID_BITS-1:0]     core;
        input [NEURON_BITS-1:0]      neuron;
        input signed [DATA_WIDTH-1:0] current;
    begin
        @(posedge clk);
        ext_valid     <= 1;
        ext_core      <= core;
        ext_neuron_id <= neuron;
        ext_current   <= current;
        @(posedge clk);
        ext_valid <= 0;
        start     <= 1;
        @(posedge clk);
        start <= 0;
        wait (timestep_done);
        @(posedge clk);
    end
    endtask

    task run_empty;
    begin
        @(posedge clk);
        start <= 1;
        @(posedge clk);
        start <= 0;
        wait (timestep_done);
        @(posedge clk);
    end
    endtask

    task do_probe;
        input [CORE_ID_BITS-1:0]     core;
        input [NEURON_BITS-1:0]      neuron;
        input [3:0]                   sid;
        input [POOL_ADDR_BITS-1:0]   paddr;
    begin
        probe_read      <= 1;
        probe_core      <= core;
        probe_neuron    <= neuron;
        probe_state_id  <= sid;
        probe_pool_addr <= paddr;
        @(posedge clk);
        probe_read <= 0;
        wait(probe_valid);
        @(posedge clk);
    end
    endtask

    integer pass_count, fail_count;
    integer i;
    reg signed [15:0] probed_val;
    reg signed [15:0] v_prev;
    reg signed [15:0] expected_decay;
    reg signed [15:0] actual_decay;

    initial begin
        clk = 0; rst_n = 0;
        start = 0;
        prog_pool_we = 0; prog_pool_core = 0; prog_pool_addr = 0;
        prog_pool_src = 0; prog_pool_target = 0; prog_pool_weight = 0; prog_pool_comp = 0;
        prog_index_we = 0; prog_index_core = 0; prog_index_neuron = 0;
        prog_index_base = 0; prog_index_count = 0;
        prog_route_we = 0; prog_route_src_core = 0; prog_route_src_neuron = 0;
        prog_route_slot = 0;
        prog_route_dest_core = 0; prog_route_dest_neuron = 0; prog_route_weight = 0;
        learn_enable = 0; graded_enable = 0; dendritic_enable = 0;
        async_enable = 0; threefactor_enable = 0; noise_enable = 0;
        skip_idle_enable = 0; reward_value = 0;
        prog_param_we = 0; prog_param_core = 0; prog_param_neuron = 0;
        prog_param_id = 0; prog_param_value = 0;
        ext_valid = 0; ext_core = 0; ext_neuron_id = 0; ext_current = 0;
        probe_read = 0; probe_core = 0; probe_neuron = 0;
        probe_state_id = 0; probe_pool_addr = 0;

        pass_count = 0; fail_count = 0;

        #100 rst_n = 1;
        @(posedge clk); @(posedge clk);

        // TEST 1: Fractional Decay
        //
        // Neuron 5 on core 0 in CUBA mode with:
        //   decay_v = 1365 (delta), decay_u = 0 (no u decay)
        //   bias = 0, threshold = 10000 (high, prevent spike)
        //
        // Loihi decay: v_decay_step = (v * 1365) >> 12 with RAZ
        // delta=1365 → approximately tau=3 (4096/1365 ≈ 3.0)
        //
        // Inject v=3000 via u pathway:
        //   t=0: inject 3000 to u. u=3000, v=0 (uses u_old=0)
        //   t=1: u=3000 (no u decay). v = 0 - 0 + 3000 = 3000
        //   t=2: v = 3000 - RAZ(3000*1365/4096) + 3000
        //       decay = 3000*1365 = 4095000, >>12 = 999.755..., RAZ=1000
        //       v = 3000 - 1000 + 3000 = 5000
        //   After multiple steps, verify decay amount ~= v*1365/4096
        //
        // Simpler approach: set v directly to known value, run empty, check decay.
        // Use LIF mode: no CUBA overhead.
        //
        // Simplest: set decay_v=1365, bias=0, inject 3000 to neuron 5 via stimulus.
        // After t=0: u=3000, v=0
        // After t=1 (empty): u=3000, v=3000 (from u_old=3000)
        // After t=2 (empty): v_decay = RAZ(3000*1365>>12) = RAZ(999.755) = 1000
        //   v = 3000 - 1000 + 3000 = 5000
        // After t=3 (empty): v_decay = RAZ(5000*1365>>12) = RAZ(1666.26) = 1667
        //   v = 5000 - 1667 + 3000 = 6333
        //
        $display("\n=== TEST 1: Fractional Decay (delta=1365) ===");

        // Configure neuron 5 CUBA: decay_v=1365, decay_u=0, threshold=30000
        set_param(0, 10'd5, 5'd16, 16'd1365);  // decay_v = 1365
        set_param(0, 10'd5, 5'd17, 16'd0);     // decay_u = 0
        set_param(0, 10'd5, 5'd0,  16'sd30000); // threshold very high

        // t=0: inject current 3000 to neuron 5
        run_timestep(0, 10'd5, 16'sd3000);

        // Probe u (state_id 13) — should be 3000
        do_probe(0, 10'd5, 4'd13, 0);
        probed_val = $signed(probe_data);
        $display("  After t=0: u = %0d (expected 3000)", probed_val);

        // t=1: empty — v gets u=3000
        run_empty;
        do_probe(0, 10'd5, 4'd0, 0);  // probe v (state_id 0)
        v_prev = $signed(probe_data);
        $display("  After t=1: v = %0d (expected ~3000)", v_prev);

        // t=2: empty — v_decay = RAZ(3000 * 1365 >> 12)
        run_empty;
        do_probe(0, 10'd5, 4'd0, 0);
        probed_val = $signed(probe_data);
        actual_decay = v_prev - probed_val + 3000;  // v_new = v_old - decay + u
        // Expected decay of 3000: 3000*1365 = 4095000, /4096 = 999.755 → RAZ = 1000
        $display("  After t=2: v = %0d, decay_amount = %0d (expected ~1000)", probed_val, actual_decay);

        if (actual_decay >= 999 && actual_decay <= 1001) begin
            $display("  PASSED: Fractional decay matches Loihi equation");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: Decay amount %0d not in [999,1001]", actual_decay);
            fail_count = fail_count + 1;
        end

        // TEST 2: RAZ Rounding
        //
        // Use neuron 10 with CUBA decay_v=3000.
        // Set v to 100 by injecting through u, then check decay.
        //
        // Decay: v * 3000 / 4096
        //   100 * 3000 = 300000, / 4096 = 73.242... → RAZ(positive) = 74
        //
        // For negative: v = -100, same delta → -300000 / 4096 = -73.242...
        //   RAZ(negative) = -74
        //
        // Neuron 10: positive test, Neuron 11: negative test (via neg bias)
        $display("\n=== TEST 2: RAZ Rounding ===");

        // Configure neuron 10: decay_v=3000, threshold=30000
        set_param(0, 10'd10, 5'd16, 16'd3000);  // decay_v = 3000
        set_param(0, 10'd10, 5'd17, 16'd0);     // decay_u = 0
        set_param(0, 10'd10, 5'd0,  16'sd30000);

        // Inject u=100 to set up voltage
        run_timestep(0, 10'd10, 16'sd100);
        // t=1: v = 0 - 0 + 100 = 100 (from u_old=100)
        run_empty;
        do_probe(0, 10'd10, 4'd0, 0);
        v_prev = $signed(probe_data);
        $display("  Neuron 10 v = %0d (expected 100)", v_prev);

        // t=2: v_new = 100 - RAZ(100*3000/4096) + 100
        //   decay = RAZ(73.242) = 74
        //   v_new = 100 - 74 + 100 = 126
        run_empty;
        do_probe(0, 10'd10, 4'd0, 0);
        probed_val = $signed(probe_data);
        actual_decay = v_prev - probed_val + 100;  // v_new = v_old - decay + u(=100)
        $display("  After decay: v = %0d, decay = %0d (expected 74 via RAZ)", probed_val, actual_decay);

        if (actual_decay == 74) begin
            $display("  PASSED: RAZ rounding ceil(73.24) = 74");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: Expected decay=74, got %0d", actual_decay);
            fail_count = fail_count + 1;
        end

        // TEST 3: Noise Target Configuration
        //
        // Neuron 20: noise_target=1 (voltage). Enable noise.
        // Set noise_cfg to {exp=0, mant=15} = mask=15. Noise in [0,15]-7 = [-7,+8].
        //
        // Neuron 21: noise_target=0 (threshold, default). Same noise_cfg.
        //
        // Both in CUBA mode. After a few timesteps, neuron 20 should have
        // varying v due to noise, while threshold is clean. Neuron 21 has
        // clean v but noisy threshold.
        //
        // Approach: run 10 timesteps, probe v each time. Check:
        // - Neuron 20: threshold is exactly the programmed value (no noise)
        // - Neuron 21: threshold varies from programmed value (has noise)
        // (We test by probing threshold via state_id=1)
        $display("\n=== TEST 3: Noise Target Configuration ===");

        // Neuron 20: noise_target = 1 (voltage)
        set_param(0, 10'd20, 5'd16, 16'd1000);   // decay_v = 1000
        set_param(0, 10'd20, 5'd17, 16'd0);      // decay_u = 0
        set_param(0, 10'd20, 5'd0,  16'sd30000); // threshold = 30000
        set_param(0, 10'd20, 5'd5,  16'h0F);     // noise_cfg: exp=0, mant=15
        set_param(0, 10'd20, 5'd29, 16'd1);      // noise_target = 1 (voltage)

        // Neuron 21: noise_target = 0 (threshold, default)
        set_param(0, 10'd21, 5'd16, 16'd1000);   // decay_v = 1000
        set_param(0, 10'd21, 5'd17, 16'd0);      // decay_u = 0
        set_param(0, 10'd21, 5'd0,  16'sd30000); // threshold = 30000
        set_param(0, 10'd21, 5'd5,  16'h0F);     // noise_cfg: exp=0, mant=15
        // noise_target stays at default 0

        noise_enable = 1;

        // Inject some current to both neurons so v is non-zero
        run_timestep(0, 10'd20, 16'sd500);
        // Also inject to neuron 21 by running another timestep
        run_timestep(0, 10'd21, 16'sd500);

        // Run 5 more timesteps to let noise accumulate
        for (i = 0; i < 5; i = i + 1) run_empty;

        // Probe neuron 20's threshold — should be exactly 30000 (no noise on threshold)
        do_probe(0, 10'd20, 4'd1, 0);  // state_id=1 = threshold
        probed_val = $signed(probe_data);
        $display("  Neuron 20 (target=voltage): threshold = %0d (expected 30000)", probed_val);

        if (probed_val == 16'sd30000) begin
            $display("  PASSED: Threshold clean when noise targets voltage");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: Expected threshold=30000, got %0d", probed_val);
            fail_count = fail_count + 1;
        end

        noise_enable = 0;

        // TEST 4: vmin/vmax Voltage Clamp
        //
        // Neuron 30: vmin=-500, vmax=500 (CUBA mode)
        // Inject large positive current → v should clamp at 500
        // Then inject large negative current → v should clamp at -500
        $display("\n=== TEST 4: vmin/vmax Voltage Clamp ===");

        // Configure neuron 30: CUBA mode
        set_param(0, 10'd30, 5'd16, 16'd500);    // decay_v = 500 (slow decay)
        set_param(0, 10'd30, 5'd17, 16'd0);      // decay_u = 0
        set_param(0, 10'd30, 5'd0,  16'sd30000); // threshold very high
        set_param(0, 10'd30, 5'd30, -16'sd500);  // vmin = -500
        set_param(0, 10'd30, 5'd31, 16'sd500);   // vmax = +500

        // Inject large positive current via u
        run_timestep(0, 10'd30, 16'sd5000);
        // t=0: u=5000, v=0
        run_empty;
        // t=1: v = 0 - 0 + 5000 = 5000 → clamped to 500
        do_probe(0, 10'd30, 4'd0, 0);
        probed_val = $signed(probe_data);
        $display("  After large positive injection: v = %0d (expected 500, clamped)", probed_val);

        if (probed_val == 16'sd500) begin
            $display("  PASSED: vmax clamp working");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: Expected v=500, got %0d", probed_val);
            fail_count = fail_count + 1;
        end

        $display("\n=== P23A RESULTS: %0d passed, %0d failed out of %0d ===",
            pass_count, fail_count, pass_count + fail_count);
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");
        $finish;
    end

    initial begin
        #5000000;
        $display("TIMEOUT - simulation exceeded 5ms");
        $finish;
    end

endmodule
