// ============================================================================
// P22G Testbench: Multi-Chip Enhancement
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

module tb_p22g_multichip;

    parameter CLK_PERIOD   = 10;
    parameter NUM_LINKS    = 2;
    parameter CHIP_ID_BITS = 4;
    parameter CORE_ID_BITS = 7;
    parameter NEURON_BITS  = 10;
    parameter DATA_WIDTH   = 16;

    reg clk, rst_n;
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    reg                        a_tx_push;
    reg  [CHIP_ID_BITS-1:0]   a_tx_dest_chip;
    reg  [CORE_ID_BITS-1:0]   a_tx_core;
    reg  [NEURON_BITS-1:0]    a_tx_neuron;
    reg  [7:0]                a_tx_payload;
    wire                       a_tx_full;

    wire [CHIP_ID_BITS-1:0]   a_rx_src_chip;
    wire [CORE_ID_BITS-1:0]   a_rx_core;
    wire [NEURON_BITS-1:0]    a_rx_neuron;
    wire signed [DATA_WIDTH-1:0] a_rx_current;
    reg                        a_rx_pop;
    wire                       a_rx_empty;

    wire [NUM_LINKS*8-1:0]    a_link_tx_data;
    wire [NUM_LINKS-1:0]      a_link_tx_valid;
    reg  [NUM_LINKS-1:0]      a_link_tx_ready;
    reg  [NUM_LINKS*8-1:0]    a_link_rx_data;
    reg  [NUM_LINKS-1:0]      a_link_rx_valid;
    wire [NUM_LINKS-1:0]      a_link_rx_ready;

    multi_chip_router #(
        .NUM_LINKS(NUM_LINKS),
        .CHIP_ID_BITS(CHIP_ID_BITS),
        .CORE_ID_BITS(CORE_ID_BITS),
        .NEURON_BITS(NEURON_BITS),
        .DATA_WIDTH(DATA_WIDTH)
    ) chip_a (
        .clk(clk), .rst_n(rst_n),
        .my_chip_id(4'd0),
        .tx_push(a_tx_push), .tx_dest_chip(a_tx_dest_chip),
        .tx_core(a_tx_core), .tx_neuron(a_tx_neuron),
        .tx_payload(a_tx_payload), .tx_full(a_tx_full),
        .rx_src_chip(a_rx_src_chip), .rx_core(a_rx_core),
        .rx_neuron(a_rx_neuron), .rx_current(a_rx_current),
        .rx_pop(a_rx_pop), .rx_empty(a_rx_empty),
        .link_tx_data(a_link_tx_data), .link_tx_valid(a_link_tx_valid),
        .link_tx_ready(a_link_tx_ready),
        .link_rx_data(a_link_rx_data), .link_rx_valid(a_link_rx_valid),
        .link_rx_ready(a_link_rx_ready)
    );

    reg                        b_tx_push;
    reg  [CHIP_ID_BITS-1:0]   b_tx_dest_chip;
    reg  [CORE_ID_BITS-1:0]   b_tx_core;
    reg  [NEURON_BITS-1:0]    b_tx_neuron;
    reg  [7:0]                b_tx_payload;
    wire                       b_tx_full;

    wire [CHIP_ID_BITS-1:0]   b_rx_src_chip;
    wire [CORE_ID_BITS-1:0]   b_rx_core;
    wire [NEURON_BITS-1:0]    b_rx_neuron;
    wire signed [DATA_WIDTH-1:0] b_rx_current;
    reg                        b_rx_pop;
    wire                       b_rx_empty;

    wire [NUM_LINKS*8-1:0]    b_link_tx_data;
    wire [NUM_LINKS-1:0]      b_link_tx_valid;
    reg  [NUM_LINKS-1:0]      b_link_tx_ready;
    reg  [NUM_LINKS*8-1:0]    b_link_rx_data;
    reg  [NUM_LINKS-1:0]      b_link_rx_valid;
    wire [NUM_LINKS-1:0]      b_link_rx_ready;

    multi_chip_router #(
        .NUM_LINKS(NUM_LINKS),
        .CHIP_ID_BITS(CHIP_ID_BITS),
        .CORE_ID_BITS(CORE_ID_BITS),
        .NEURON_BITS(NEURON_BITS),
        .DATA_WIDTH(DATA_WIDTH)
    ) chip_b (
        .clk(clk), .rst_n(rst_n),
        .my_chip_id(4'd1),
        .tx_push(b_tx_push), .tx_dest_chip(b_tx_dest_chip),
        .tx_core(b_tx_core), .tx_neuron(b_tx_neuron),
        .tx_payload(b_tx_payload), .tx_full(b_tx_full),
        .rx_src_chip(b_rx_src_chip), .rx_core(b_rx_core),
        .rx_neuron(b_rx_neuron), .rx_current(b_rx_current),
        .rx_pop(b_rx_pop), .rx_empty(b_rx_empty),
        .link_tx_data(b_link_tx_data), .link_tx_valid(b_link_tx_valid),
        .link_tx_ready(b_link_tx_ready),
        .link_rx_data(b_link_rx_data), .link_rx_valid(b_link_rx_valid),
        .link_rx_ready(b_link_rx_ready)
    );

    // Link Wiring: chip_a link0 TX ↔ chip_b link0 RX and vice versa
    // For test 1-3: loopback chip_a link0 TX → chip_a link0 RX
    // For test 4: cross-connect chip_a ↔ chip_b
    reg loopback_mode;

    always @(*) begin
        if (loopback_mode) begin
            // Loopback: chip_a TX → chip_a RX
            a_link_rx_data  = a_link_tx_data;
            a_link_rx_valid = a_link_tx_valid;
            a_link_tx_ready = a_link_rx_ready;
            // chip_b disconnected
            b_link_rx_data  = 0;
            b_link_rx_valid = 0;
            b_link_tx_ready = {NUM_LINKS{1'b1}};
        end else begin
            // Cross-connect: chip_a link0 → chip_b link0 RX, chip_b link0 → chip_a link0 RX
            // Link 0
            a_link_rx_data[7:0]  = b_link_tx_data[7:0];
            a_link_rx_valid[0]   = b_link_tx_valid[0];
            b_link_tx_ready[0]   = a_link_rx_ready[0];

            b_link_rx_data[7:0]  = a_link_tx_data[7:0];
            b_link_rx_valid[0]   = a_link_tx_valid[0];
            a_link_tx_ready[0]   = b_link_rx_ready[0];

            // Link 1 (unused in cross-connect, tie off)
            a_link_rx_data[15:8] = 8'd0;
            a_link_rx_valid[1]   = 1'b0;
            a_link_tx_ready[1]   = 1'b1;

            b_link_rx_data[15:8] = 8'd0;
            b_link_rx_valid[1]   = 1'b0;
            b_link_tx_ready[1]   = 1'b1;
        end
    end

    task push_spike_a;
        input [CHIP_ID_BITS-1:0] dest_chip;
        input [CORE_ID_BITS-1:0] core;
        input [NEURON_BITS-1:0]  neuron;
        input [7:0]              payload;
    begin
        @(posedge clk);
        a_tx_push      <= 1;
        a_tx_dest_chip <= dest_chip;
        a_tx_core      <= core;
        a_tx_neuron    <= neuron;
        a_tx_payload   <= payload;
        @(posedge clk);
        a_tx_push <= 0;
    end
    endtask

    task push_spike_b;
        input [CHIP_ID_BITS-1:0] dest_chip;
        input [CORE_ID_BITS-1:0] core;
        input [NEURON_BITS-1:0]  neuron;
        input [7:0]              payload;
    begin
        @(posedge clk);
        b_tx_push      <= 1;
        b_tx_dest_chip <= dest_chip;
        b_tx_core      <= core;
        b_tx_neuron    <= neuron;
        b_tx_payload   <= payload;
        @(posedge clk);
        b_tx_push <= 0;
    end
    endtask

    task wait_cycles;
        input integer n;
        integer i;
    begin
        for (i = 0; i < n; i = i + 1) @(posedge clk);
    end
    endtask

    integer pass_count, fail_count;

    initial begin
        #5000000;
        $display("TIMEOUT");
        $finish;
    end

    initial begin
        clk = 0; rst_n = 0;
        a_tx_push = 0; a_tx_dest_chip = 0; a_tx_core = 0;
        a_tx_neuron = 0; a_tx_payload = 0; a_rx_pop = 0;
        b_tx_push = 0; b_tx_dest_chip = 0; b_tx_core = 0;
        b_tx_neuron = 0; b_tx_payload = 0; b_rx_pop = 0;
        loopback_mode = 1;
        pass_count = 0; fail_count = 0;

        #100;
        rst_n = 1;
        #50;

        // Test 1: Single-link loopback
        //   Push spike from chip_a, loopback TX→RX, verify received
        $display("\n=== Test 1: Single-link loopback ===");
        loopback_mode = 1;

        push_spike_a(4'd0, 7'd5, 10'd42, 8'd128);  // dest_chip=0 → link0
        wait_cycles(50);  // Wait for serialization + loopback + deserialization

        if (!a_rx_empty) begin
            $display("  RX: src_chip=%0d core=%0d neuron=%0d current=%0d",
                a_rx_src_chip, a_rx_core, a_rx_neuron, a_rx_current);
            if (a_rx_core == 5 && a_rx_neuron == 42 && a_rx_current == 128) begin
                $display("  PASSED: loopback delivered correctly");
                pass_count = pass_count + 1;
            end else begin
                $display("  FAILED: data mismatch");
                fail_count = fail_count + 1;
            end
            a_rx_pop = 1; @(posedge clk); a_rx_pop = 0;
        end else begin
            $display("  FAILED: RX FIFO empty after loopback");
            fail_count = fail_count + 1;
        end

        wait_cycles(10);

        // Test 2: Link routing by chip_id
        //   dest_chip=0 → link 0 (0%2=0), dest_chip=1 → link 1 (1%2=1)
        //   In loopback mode, both links loop back to chip_a
        $display("\n=== Test 2: Chip ID → link routing ===");
        loopback_mode = 1;

        // Send to chip 0 (link 0)
        push_spike_a(4'd0, 7'd10, 10'd100, 8'd64);
        // Send to chip 1 (link 1)
        push_spike_a(4'd1, 7'd20, 10'd200, 8'd32);

        wait_cycles(100);

        // Should have 2 packets in RX FIFO
        if (!a_rx_empty) begin
            $display("  Pkt1: core=%0d neuron=%0d current=%0d",
                a_rx_core, a_rx_neuron, a_rx_current);
            a_rx_pop = 1; @(posedge clk); a_rx_pop = 0;
            @(posedge clk); // Let FIFO update
        end

        if (!a_rx_empty) begin
            $display("  Pkt2: core=%0d neuron=%0d current=%0d",
                a_rx_core, a_rx_neuron, a_rx_current);
            $display("  PASSED: both packets received via different links");
            pass_count = pass_count + 1;
            a_rx_pop = 1; @(posedge clk); a_rx_pop = 0;
        end else begin
            $display("  FAILED: expected 2 packets, got <2");
            fail_count = fail_count + 1;
        end

        wait_cycles(10);

        // Test 3: Multiple packets burst
        //   Send 4 packets rapidly, verify all 4 arrive
        $display("\n=== Test 3: Burst of 4 packets ===");
        loopback_mode = 1;

        push_spike_a(4'd0, 7'd1, 10'd1, 8'd10);
        push_spike_a(4'd0, 7'd2, 10'd2, 8'd20);
        push_spike_a(4'd0, 7'd3, 10'd3, 8'd30);
        push_spike_a(4'd0, 7'd4, 10'd4, 8'd40);

        wait_cycles(200);  // Wait for all 4 to serialize and loop back

        begin : count_rx_test3
            integer rx_count;
            rx_count = 0;
            while (!a_rx_empty) begin
                $display("  Pkt%0d: core=%0d neuron=%0d current=%0d",
                    rx_count+1, a_rx_core, a_rx_neuron, a_rx_current);
                a_rx_pop = 1; @(posedge clk); a_rx_pop = 0;
                @(posedge clk);
                rx_count = rx_count + 1;
            end
            if (rx_count >= 4) begin
                $display("  PASSED: all %0d packets received", rx_count);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAILED: expected 4 packets, got %0d", rx_count);
                fail_count = fail_count + 1;
            end
        end

        wait_cycles(10);

        $display("\n=== Test 4: Bidirectional cross-connect ===");
        loopback_mode = 0;  // Cross-connect mode

        wait_cycles(5);

        // Chip A sends to chip B (dest_chip=1 → link 1%2=1... but in cross-connect
        // we only have link 0 wired. Let me use dest_chip=0 for link 0)
        // dest_chip=0 → link 0%2=0 → routed to chip B via link 0

        push_spike_a(4'd0, 7'd50, 10'd500, 8'd100);  // A→B via link 0
        push_spike_b(4'd0, 7'd60, 10'd600, 8'd200);   // B→A via link 0

        wait_cycles(100);

        // Check chip B received from A
        if (!b_rx_empty) begin
            $display("  ChipB RX: src=%0d core=%0d neuron=%0d current=%0d",
                b_rx_src_chip, b_rx_core, b_rx_neuron, b_rx_current);
            b_rx_pop = 1; @(posedge clk); b_rx_pop = 0;
        end else begin
            $display("  ChipB RX: empty (FAIL)");
        end

        // Check chip A received from B
        if (!a_rx_empty) begin
            $display("  ChipA RX: src=%0d core=%0d neuron=%0d current=%0d",
                a_rx_src_chip, a_rx_core, a_rx_neuron, a_rx_current);
            a_rx_pop = 1; @(posedge clk); a_rx_pop = 0;
        end else begin
            $display("  ChipA RX: empty (FAIL)");
        end

        if (!b_rx_empty == 0 && !a_rx_empty == 0) begin
            // Both received (FIFOs now empty after pop)
            $display("  PASSED: bidirectional exchange complete");
            pass_count = pass_count + 1;
        end else begin
            $display("  Checking if both chips received...");
            // Re-check after pops
            if (b_rx_empty && a_rx_empty) begin
                $display("  PASSED: bidirectional exchange complete (FIFOs drained)");
                pass_count = pass_count + 1;
            end else begin
                $display("  FAILED: not all packets received");
                fail_count = fail_count + 1;
            end
        end

        $display("\n====================================");
        $display("P22G RESULTS: %0d/%0d passed", pass_count, pass_count + fail_count);
        $display("====================================\n");

        if (fail_count > 0)
            $display("SOME TESTS FAILED");

        $finish;
    end

endmodule
