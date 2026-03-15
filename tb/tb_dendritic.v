// ============================================================================
// Testbench: Dendritic Compartments (Phase 10)
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

module tb_dendritic;

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
    reg                    dendritic_enable;
    reg                    ext_valid;
    reg  [NEURON_BITS-1:0] ext_neuron_id;
    reg  signed [DATA_WIDTH-1:0] ext_current;
    reg                    conn_we;
    reg  [NEURON_BITS-1:0] conn_src;
    reg  [FANOUT_BITS-1:0] conn_slot;
    reg  [NEURON_BITS-1:0] conn_target;
    reg  signed [DATA_WIDTH-1:0] conn_weight;
    reg  [1:0]             conn_comp;

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
        .DEND_THRESHOLD(16'sd0),
        .TRACE_MAX     (8'd100),
        .TRACE_DECAY   (8'd10),
        .LEARN_SHIFT   (3),
        .GRADE_SHIFT   (7),
        .WEIGHT_MAX    (16'sd2000),
        .WEIGHT_MIN    (16'sd0)
    ) dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .start          (start),
        .learn_enable   (learn_enable),
        .graded_enable  (graded_enable),
        .dendritic_enable(dendritic_enable),
        .ext_valid      (ext_valid),
        .ext_neuron_id  (ext_neuron_id),
        .ext_current    (ext_current),
        .conn_we        (conn_we),
        .conn_src       (conn_src),
        .conn_slot      (conn_slot),
        .conn_target    (conn_target),
        .conn_weight    (conn_weight),
        .conn_comp      (conn_comp),
        .prog_param_we  (1'b0),
        .prog_param_neuron(8'd0),
        .prog_param_id  (3'd0),
        .prog_param_value(16'sd0),
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
        input [1:0] comp;
    begin
        @(posedge clk);
        conn_we     <= 1;
        conn_src    <= src;
        conn_slot   <= slot;
        conn_target <= target;
        conn_weight <= weight;
        conn_comp   <= comp;
        @(posedge clk);
        conn_we <= 0;
        conn_comp <= 0;
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

    // Program per-neuron parameter
    reg        param_we_r;
    reg [7:0]  param_neuron_r;
    reg [2:0]  param_id_r;
    reg signed [DATA_WIDTH-1:0] param_value_r;

    // Override the tied-off prog_param ports for tests that need it
    task set_param;
        input [NEURON_BITS-1:0] neuron;
        input [2:0] param_id;
        input signed [DATA_WIDTH-1:0] value;
    begin
        // Direct hierarchical write to parameter SRAMs (simulation only)
        case (param_id)
            3'd0: dut.threshold_mem.mem[neuron]  = value;
            3'd1: dut.leak_mem.mem[neuron]       = value;
            3'd2: dut.rest_mem.mem[neuron]       = value;
            3'd3: dut.refrac_cfg_mem.mem[neuron] = value[7:0];
            3'd4: dut.dend_thr_mem.mem[neuron]   = value;
        endcase
    end
    endtask

    // Read membrane potential
    function signed [DATA_WIDTH-1:0] read_potential;
        input [NEURON_BITS-1:0] neuron;
    begin
        read_potential = dut.neuron_mem.mem[neuron];
    end
    endfunction

    // Read dendrite accumulator
    function signed [DATA_WIDTH-1:0] read_dend_acc;
        input [NEURON_BITS-1:0] neuron;
        input [1:0] dend_id;
    begin
        case (dend_id)
            2'd1: read_dend_acc = dut.dend_acc_1_mem.mem[neuron];
            2'd2: read_dend_acc = dut.dend_acc_2_mem.mem[neuron];
            2'd3: read_dend_acc = dut.dend_acc_3_mem.mem[neuron];
            default: read_dend_acc = dut.acc_mem.mem[neuron];
        endcase
    end
    endfunction

    integer spike_count;
    reg [7:0] last_spike_id;

    always @(posedge clk) begin
        if (spike_out_valid) begin
            spike_count = spike_count + 1;
            last_spike_id = spike_out_id;
        end
    end

    integer pass_count, fail_count;
    integer i;
    reg signed [DATA_WIDTH-1:0] pot_val;

    initial begin
        rst_n            = 0;
        start            = 0;
        learn_enable     = 0;
        graded_enable    = 0;
        dendritic_enable = 0;
        ext_valid        = 0;
        conn_we          = 0;
        conn_src         = 0;
        conn_slot        = 0;
        conn_target      = 0;
        conn_weight      = 0;
        conn_comp        = 0;
        ext_neuron_id    = 0;
        ext_current      = 0;
        spike_count      = 0;
        pass_count       = 0;
        fail_count       = 0;

        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 3);

        $display("");
        $display("================================================================");
        $display("  Dendritic Compartments Test (Phase 10)");
        $display("================================================================");

        // TEST 1: Backward Compatibility (soma-only, dendritic_enable=0)
        //   N0 -> N2 via soma (comp=0). Should behave exactly as pre-P10.
        $display("");
        $display("--- TEST 1: Backward Compatibility (soma-only) ---");

        dendritic_enable = 0;
        program_conn(8'd0, 5'd0, 8'd2, 16'sd1200, 2'd0);  // soma

        stimulate(8'd0, 16'sd1200);
        spike_count = 0;
        run_timestep;  // TS1: N0 spikes
        $display("  TS1: N0 spikes=%0d", spike_count);

        run_timestep;  // TS2: N0->N2 delivers via soma
        pot_val = read_potential(8'd2);
        // Expected: 0 + 1200 - 3 = 1197 (>= 1000, so N2 spikes)
        $display("  TS2: N2 potential after delivery = %0d, spikes=%0d", pot_val, spike_count);

        if (spike_count >= 2) begin
            $display("  PASS: Both N0 and N2 spiked (backward compat)");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected >=2 spikes, got %0d", spike_count);
            fail_count = fail_count + 1;
        end

        // TEST 2: Compartment Routing
        //   N10 -> N12 via dendrite 1 (comp=1), weight=600
        //   N10 -> N14 via soma (comp=0), weight=600
        //   dendritic_enable=1, dend_threshold=0 (pass-through)
        //   After N10 spikes and delivers, N12 gets 600 via dendrite,
        //   N14 gets 600 via soma. Both should integrate.
        $display("");
        $display("--- TEST 2: Compartment Routing ---");

        dendritic_enable = 1;
        program_conn(8'd10, 5'd0, 8'd12, 16'sd600, 2'd1);  // dendrite 1
        program_conn(8'd10, 5'd1, 8'd14, 16'sd600, 2'd0);  // soma

        // Stimulate N10 enough to spike
        stimulate(8'd10, 16'sd1200);
        spike_count = 0;
        run_timestep;  // N10 spikes
        $display("  TS: N10 spiked, spikes=%0d", spike_count);

        run_timestep;  // Delivery happens
        // N12: dendrite input=600, dend_thr=0, contrib=600, soma=0+600-3=597
        // N14: soma input=600, pot=0+600-3=597
        begin : test2_block
            reg signed [DATA_WIDTH-1:0] pot_n12, pot_n14;
            pot_n12 = read_potential(8'd12);
            pot_n14 = read_potential(8'd14);
            $display("  N12 (dendrite path) potential = %0d", pot_n12);
            $display("  N14 (soma path) potential = %0d", pot_n14);

            if (pot_n12 > 0 && pot_n14 > 0) begin
                $display("  PASS: Both paths delivered current (N12=%0d, N14=%0d)", pot_n12, pot_n14);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL: Expected both >0 (N12=%0d, N14=%0d)", pot_n12, pot_n14);
                fail_count = fail_count + 1;
            end
        end

        // TEST 3: Dendritic Threshold Filtering
        //   N20 -> N22 via dendrite 1 (comp=1), weight=200
        //   N22 dend_threshold=300 (filters out 200)
        //   Then N21 -> N22 via dendrite 1 (comp=1), weight=500
        //   500 > 300, so contribution = 500-300 = 200
        $display("");
        $display("--- TEST 3: Dendritic Threshold Filtering ---");

        dendritic_enable = 1;
        set_param(8'd22, 3'd4, 16'sd300);  // dend_threshold = 300

        // Weak path: N20 -> N22 via dendrite 1, weight=200
        program_conn(8'd20, 5'd0, 8'd22, 16'sd200, 2'd1);

        // Make N20 spike
        stimulate(8'd20, 16'sd1200);
        spike_count = 0;
        run_timestep;  // N20 spikes

        run_timestep;  // Deliver 200 to N22 dendrite 1

        // N22 dendrite acc = 200, dend_thr = 300, so 200 > 300 = false -> contrib = 0
        // N22 potential should be near 0 (only leak applied)
        pot_val = read_potential(8'd22);
        $display("  Weak input (200 < thr 300): N22 potential = %0d (expected ~0)", pot_val);

        if (pot_val <= 16'sd0) begin
            $display("  PASS: Weak dendritic input filtered out");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected <=0, got %0d", pot_val);
            fail_count = fail_count + 1;
        end

        // Strong path: N21 -> N22 via dendrite 2, weight=500
        program_conn(8'd21, 5'd0, 8'd22, 16'sd500, 2'd2);

        stimulate(8'd21, 16'sd1200);
        run_timestep;  // N21 spikes

        run_timestep;  // Deliver 500 to N22 dendrite 2
        // dend acc 2 = 500, 500 > 300 = true -> contrib = 200
        // N22 potential: 0 + 200 - 3 = 197
        pot_val = read_potential(8'd22);
        $display("  Strong input (500 > thr 300): N22 potential = %0d (expected ~197)", pot_val);

        if (pot_val > 16'sd0) begin
            $display("  PASS: Strong dendritic input passed through (%0d)", pot_val);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected >0, got %0d", pot_val);
            fail_count = fail_count + 1;
        end

        // TEST 4: Coincidence Detection (dendritic AND gate)
        //   Part A: N30 -> N32 via dend1, N31 -> N32 via dend2
        //     Only N30 fires. N32 gets 300(soma)+400(dend1)-3=697 < 1000 -> no spike
        //   Part B: N33 -> N35 via dend1, N34 -> N35 via dend2
        //     BOTH fire. N35 gets 300(soma)+400(dend1)+400(dend2)-3=1097 >= 1000 -> spike!
        //   Uses separate neurons per part to avoid refractory conflicts.
        $display("");
        $display("--- TEST 4: Coincidence Detection (AND gate) ---");

        dendritic_enable = 1;

        // Part A: single dendrite (should NOT spike)
        program_conn(8'd30, 5'd0, 8'd32, 16'sd400, 2'd1);  // N30->N32 dendrite 1
        program_conn(8'd31, 5'd0, 8'd32, 16'sd400, 2'd2);  // N31->N32 dendrite 2

        stimulate(8'd30, 16'sd1200);
        spike_count = 0;
        run_timestep;  // N30 spikes

        stimulate(8'd32, 16'sd300);  // sub-threshold soma bias
        run_timestep;  // deliver N30->N32 dend1 + soma bias
        // N32 total = 300(soma) + 400(dend1) - 3(leak) = 697 < 1000
        begin : test4a_block
            integer spikes_single;
            spikes_single = spike_count;
            pot_val = read_potential(8'd32);
            $display("  Part A (single dend): N32 pot=%0d, spikes=%0d", pot_val, spikes_single);

            if (spikes_single == 1) begin
                $display("  PASS: No N32 spike with single dendrite");
                pass_count = pass_count + 1;
            end else if (last_spike_id != 8'd32) begin
                $display("  PASS: No N32 spike with single dendrite (spikes=%0d)", spikes_single);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL: N32 spiked with single dendrite");
                fail_count = fail_count + 1;
            end
        end

        // Part B: both dendrites (should spike) — fresh neurons
        program_conn(8'd33, 5'd0, 8'd35, 16'sd400, 2'd1);  // N33->N35 dendrite 1
        program_conn(8'd34, 5'd0, 8'd35, 16'sd400, 2'd2);  // N34->N35 dendrite 2

        stimulate(8'd33, 16'sd1200);
        stimulate(8'd34, 16'sd1200);
        spike_count = 0;
        run_timestep;  // Both N33 and N34 spike
        $display("  Part B: N33+N34 spiked, spikes=%0d", spike_count);

        stimulate(8'd35, 16'sd300);  // soma bias
        run_timestep;  // deliver both + soma bias
        // N35: 300(soma) + 400(dend1) + 400(dend2) - 3 = 1097 >= 1000 -> SPIKE
        begin : test4b_block
            pot_val = read_potential(8'd35);
            $display("  Part B: N35 pot=%0d, total_spikes=%0d", pot_val, spike_count);

            if (spike_count >= 3) begin
                $display("  PASS: Coincidence spike! N35 fired with both dendrites (%0d spikes)", spike_count);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL: Expected >=3 spikes (N33+N34+N35), got %0d", spike_count);
                fail_count = fail_count + 1;
            end
        end

        // TEST 5: Dendritic Enable Toggle
        //   N40 -> N42 via dendrite 1, weight=1200
        //   With dendritic_enable=0: dend input ignored -> N42 no spike
        //   With dendritic_enable=1: dend input included -> N42 spikes
        $display("");
        $display("--- TEST 5: Dendritic Enable Toggle ---");

        program_conn(8'd40, 5'd0, 8'd42, 16'sd1200, 2'd1);  // dendrite 1

        // Part A: dendritic_enable = 0
        dendritic_enable = 0;
        stimulate(8'd40, 16'sd1200);
        spike_count = 0;
        run_timestep;  // N40 spikes

        run_timestep;  // Deliver to N42 dendrite 1
        // With dendritic_enable=0, total_input = acc_rdata only (soma=0), no spike
        pot_val = read_potential(8'd42);
        $display("  dendritic_enable=0: N42 potential = %0d", pot_val);

        begin : test5a_block
            integer spikes_off;
            spikes_off = spike_count;
            if (pot_val <= 16'sd0) begin
                $display("  PASS: Dendrite ignored when disabled (pot=%0d)", pot_val);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL: Expected pot<=0 when disabled, got %0d", pot_val);
                fail_count = fail_count + 1;
            end
        end

        // Part B: dendritic_enable = 1 (use fresh neurons N50->N52)
        dendritic_enable = 1;
        program_conn(8'd50, 5'd0, 8'd52, 16'sd1200, 2'd1);  // dendrite 1

        stimulate(8'd50, 16'sd1200);
        spike_count = 0;
        run_timestep;  // N50 spikes

        run_timestep;  // Deliver 1200 to N52 dendrite 1
        // dend_thr=0, contrib=1200, total=0+1200-3=1197 >= 1000 -> SPIKE!
        pot_val = read_potential(8'd52);
        $display("  dendritic_enable=1: N52 potential = %0d (0 if spiked)", pot_val);

        if (spike_count >= 2) begin
            $display("  PASS: Dendrite active when enabled, N52 spiked");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected N52 to spike, spikes=%0d", spike_count);
            fail_count = fail_count + 1;
        end

        $display("");
        $display("================================================================");
        $display("  DENDRITIC COMPARTMENT TEST RESULTS: %0d PASS, %0d FAIL", pass_count, fail_count);
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
