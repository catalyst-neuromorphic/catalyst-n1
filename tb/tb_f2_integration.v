// ============================================================================
// Testbench: F2 Integration — End-to-End AXI-Lite BFM to Neuromorphic Mesh
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

module tb_f2_integration;

    reg clk, rst_n;
    initial clk = 0;
    always #5 clk = ~clk;  // 100 MHz (sim speed; real = 250 MHz)

    reg  [31:0] axi_awaddr, axi_wdata, axi_araddr;
    reg  [3:0]  axi_wstrb;
    reg         axi_awvalid, axi_wvalid, axi_arvalid, axi_bready, axi_rready;
    wire        axi_awready, axi_wready, axi_arready, axi_bvalid, axi_rvalid;
    wire [1:0]  axi_bresp, axi_rresp;
    wire [31:0] axi_rdata;

    wire [31:0] cl_sh_id0, cl_sh_id1;
    wire [31:0] cl_sh_status0, cl_sh_status1;

    wire        flr_done;
    wire [15:0] irq_req;
    wire        ddr_stat_ack;
    wire [31:0] ddr_stat_rdata;
    wire [7:0]  ddr_stat_int;

    wire [63:0]  pcim_awaddr, pcim_araddr;
    wire [15:0]  pcim_awid, pcim_arid;
    wire [7:0]   pcim_awlen, pcim_arlen;
    wire [2:0]   pcim_awsize, pcim_arsize;
    wire         pcim_awvalid, pcim_arvalid;
    wire [511:0] pcim_wdata;
    wire [63:0]  pcim_wstrb;
    wire         pcim_wlast, pcim_wvalid;
    wire         pcim_bready, pcim_rready;

    wire         pcis_awready, pcis_wready;
    wire [1:0]   pcis_bresp;
    wire [15:0]  pcis_bid;
    wire         pcis_bvalid;
    wire         pcis_arready;
    wire [511:0] pcis_rdata;
    wire [15:0]  pcis_rid;
    wire [1:0]   pcis_rresp;
    wire         pcis_rlast, pcis_rvalid;

    wire         sda_awready, sda_wready;
    wire [1:0]   sda_bresp;
    wire         sda_bvalid;
    wire         sda_arready;
    wire [31:0]  sda_rdata;
    wire [1:0]   sda_rresp;
    wire         sda_rvalid;

    // instantiate bridge + neuromorphic_top directly with small params.
    // This tests the same wiring as cl_neuromorphic.v but at sim-friendly scale.

    wire [7:0] bridge_rx_data;
    wire       bridge_rx_valid;
    wire [7:0] bridge_tx_data;
    wire       bridge_tx_valid;
    wire       bridge_tx_ready;

    axi_uart_bridge #(
        .FIFO_DEPTH (32),
        .VERSION_ID (32'hF2_02_03_80),
        .NUM_CORES  (4)
    ) u_bridge (
        .clk            (clk),
        .rst_n          (rst_n),
        .s_axi_awaddr   (axi_awaddr),
        .s_axi_awvalid  (axi_awvalid),
        .s_axi_awready  (axi_awready),
        .s_axi_wdata    (axi_wdata),
        .s_axi_wstrb    (axi_wstrb),
        .s_axi_wvalid   (axi_wvalid),
        .s_axi_wready   (axi_wready),
        .s_axi_bresp    (axi_bresp),
        .s_axi_bvalid   (axi_bvalid),
        .s_axi_bready   (axi_bready),
        .s_axi_araddr   (axi_araddr),
        .s_axi_arvalid  (axi_arvalid),
        .s_axi_arready  (axi_arready),
        .s_axi_rdata    (axi_rdata),
        .s_axi_rresp    (axi_rresp),
        .s_axi_rvalid   (axi_rvalid),
        .s_axi_rready   (axi_rready),
        .hi_rx_data     (bridge_rx_data),
        .hi_rx_valid    (bridge_rx_valid),
        .hi_tx_data     (bridge_tx_data),
        .hi_tx_valid    (bridge_tx_valid),
        .hi_tx_ready    (bridge_tx_ready)
    );

    neuromorphic_top #(
        .CLK_FREQ       (100_000_000),
        .BAUD           (115200),
        .BYPASS_UART    (1),
        .NUM_CORES      (4),
        .CORE_ID_BITS   (2),
        .NUM_NEURONS    (256),
        .NEURON_BITS    (8),
        .DATA_WIDTH     (16),
        .POOL_DEPTH     (8192),
        .POOL_ADDR_BITS (13),
        .COUNT_BITS     (6),
        .REV_FANIN      (16),
        .REV_SLOT_BITS  (4),
        .THRESHOLD      (16'sd1000),
        .LEAK_RATE      (16'sd3),
        .REFRAC_CYCLES  (3),
        .ROUTE_FANOUT           (8),
        .ROUTE_SLOT_BITS        (3),
        .GLOBAL_ROUTE_SLOTS     (4),
        .GLOBAL_ROUTE_SLOT_BITS (2),
        .CHIP_LINK_EN   (0),
        .NOC_MODE       (0),
        .MESH_X         (2),
        .MESH_Y         (2)
    ) u_neuromorphic (
        .clk            (clk),
        .rst_n          (rst_n),
        .uart_rxd       (1'b1),
        .uart_txd       (),
        .rx_data_ext    (bridge_rx_data),
        .rx_valid_ext   (bridge_rx_valid),
        .tx_data_ext    (bridge_tx_data),
        .tx_valid_ext   (bridge_tx_valid),
        .tx_ready_ext   (bridge_tx_ready),
        .link_tx_data   (),
        .link_tx_valid  (),
        .link_tx_ready  (1'b0),
        .link_rx_data   (8'd0),
        .link_rx_valid  (1'b0),
        .link_rx_ready  ()
    );

    task axi_write;
        input [31:0] addr;
        input [31:0] data;
        begin
            @(posedge clk);
            axi_awaddr  <= addr;
            axi_awvalid <= 1'b1;
            axi_wdata   <= data;
            axi_wstrb   <= 4'hF;
            axi_wvalid  <= 1'b1;
            axi_bready  <= 1'b1;

            @(posedge clk);
            while (!(axi_awready || axi_wready))
                @(posedge clk);
            @(posedge clk);
            axi_awvalid <= 1'b0;
            axi_wvalid  <= 1'b0;

            while (!axi_bvalid)
                @(posedge clk);
            @(posedge clk);
            axi_bready <= 1'b0;
        end
    endtask

    task axi_read;
        input  [31:0] addr;
        output [31:0] data;
        begin
            @(posedge clk);
            axi_araddr  <= addr;
            axi_arvalid <= 1'b1;
            axi_rready  <= 1'b1;

            @(posedge clk);
            while (!axi_arready)
                @(posedge clk);
            @(posedge clk);
            axi_arvalid <= 1'b0;

            while (!axi_rvalid)
                @(posedge clk);
            data = axi_rdata;
            @(posedge clk);
            axi_rready <= 1'b0;
        end
    endtask

    task send_byte;
        input [7:0] b;
        reg [31:0] status;
        begin
            status = 0;
            while (!(status & 1)) begin
                axi_read(32'h004, status);
            end
            axi_write(32'h000, {24'd0, b});
        end
    endtask

    task recv_byte;
        output [7:0] b;
        reg [31:0] status, data;
        integer poll_count;
        begin
            status = 0;
            poll_count = 0;
            while (!(status & 1)) begin
                axi_read(32'h00C, status);
                poll_count = poll_count + 1;
                if (poll_count > 10000) begin
                    $display("  ERROR: recv_byte timeout (10000 polls)");
                    b = 8'hFF;
                    disable recv_byte;
                end
            end
            axi_read(32'h008, data);
            b = data[7:0];
        end
    endtask

    integer pass_count, fail_count;
    reg [31:0] rd_data;
    reg [7:0]  rx_byte;

    initial begin
        clk = 0;
        rst_n = 0;
        axi_awaddr = 0; axi_wdata = 0; axi_araddr = 0;
        axi_wstrb = 0;
        axi_awvalid = 0; axi_wvalid = 0; axi_arvalid = 0;
        axi_bready = 0; axi_rready = 0;
        pass_count = 0; fail_count = 0;

        repeat (20) @(posedge clk);
        rst_n = 1;
        repeat (10) @(posedge clk);

        $display("\n--- TEST 1: VERSION register ---");
        axi_read(32'h014, rd_data);
        if (rd_data == 32'hF2020380) begin
            $display("  PASSED: VERSION = 0x%08X", rd_data);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: VERSION = 0x%08X (expected 0xF2020380)", rd_data);
            fail_count = fail_count + 1;
        end

        $display("\n--- TEST 2: SCRATCH loopback ---");
        axi_write(32'h018, 32'hCAFEBABE);
        repeat (2) @(posedge clk);
        axi_read(32'h018, rd_data);
        if (rd_data == 32'hCAFEBABE) begin
            $display("  PASSED: SCRATCH = 0x%08X", rd_data);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: SCRATCH = 0x%08X (expected 0xCAFEBABE)", rd_data);
            fail_count = fail_count + 1;
        end

        $display("\n--- TEST 3: CORE_COUNT register ---");
        axi_read(32'h01C, rd_data);
        if (rd_data == 32'd4) begin
            $display("  PASSED: CORE_COUNT = %0d", rd_data);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: CORE_COUNT = %0d (expected 4)", rd_data);
            fail_count = fail_count + 1;
        end

        $display("\n--- TEST 4: STATUS command end-to-end ---");
        send_byte(8'h05);  // CMD_STATUS

        // Read 5-byte response: state(1) + timestep_count(4)
        begin : test4_block
            reg [7:0] state_byte, b1, b2, b3, b4;
            reg [31:0] ts_count;
            recv_byte(state_byte);
            recv_byte(b1);
            recv_byte(b2);
            recv_byte(b3);
            recv_byte(b4);
            ts_count = {b1, b2, b3, b4};
            $display("  State=0x%02X, ts_count=%0d", state_byte, ts_count);
            // Initial state: idle (0), timestep_count=0
            if (state_byte == 8'h00 && ts_count == 32'd0) begin
                $display("  PASSED: STATUS response correct");
                pass_count = pass_count + 1;
            end else begin
                $display("  FAILED: unexpected STATUS response");
                fail_count = fail_count + 1;
            end
        end

        // Program a 2-neuron chain: N0→N1 on core 0 (weight=1200 > threshold=1000)
        // Inject spike into N0, run 5 timesteps, expect spikes > 0
        $display("\n--- TEST 5: 2-neuron spike chain ---");

        // CMD_PROG_POOL = 0x01, 8 payload bytes
        send_byte(8'h01);  // opcode
        send_byte(8'h00);  // core=0
        send_byte(8'h00);  // addr_hi=0
        send_byte(8'h00);  // addr_lo=0
        send_byte(8'h00);  // flags/comp=0
        send_byte(8'h00);  // src=0
        send_byte(8'h01);  // tgt=1
        send_byte(8'h04);  // wt_hi (1200 >> 8 = 4)
        send_byte(8'hB0);  // wt_lo (1200 & 0xFF = 0xB0)
        recv_byte(rx_byte);
        $display("  PROG_POOL ACK: 0x%02X", rx_byte);

        // CMD_PROG_INDEX = 0x08, 7 payload bytes
        // [0]=core [1]=neuron_hi [2]=neuron_lo [3]=base_hi [4]=base_lo [5]=count_hi [6]=count_lo
        send_byte(8'h08);  // opcode
        send_byte(8'h00);  // core=0
        send_byte(8'h00);  // neuron_hi=0
        send_byte(8'h00);  // neuron_lo=0
        send_byte(8'h00);  // base_hi=0
        send_byte(8'h00);  // base_lo=0
        send_byte(8'h00);  // count_hi=0 (format[7:6]=0=SPARSE)
        send_byte(8'h01);  // count_lo=1
        recv_byte(rx_byte);
        $display("  PROG_INDEX ACK: 0x%02X", rx_byte);

        // CMD_STIMULUS = 0x03, 5 payload bytes
        // [0]=core [1]=neuron_hi [2]=neuron_lo [3]=current_hi [4]=current_lo
        send_byte(8'h03);  // opcode
        send_byte(8'h00);  // core=0
        send_byte(8'h00);  // neuron_hi=0
        send_byte(8'h00);  // neuron_lo=0
        send_byte(8'h05);  // current_hi (1500 >> 8 = 5)
        send_byte(8'hDC);  // current_lo (1500 & 0xFF = 0xDC)
        recv_byte(rx_byte);
        $display("  STIMULUS ACK: 0x%02X", rx_byte);

        // CMD_RUN = 0x04, 2 payload bytes
        send_byte(8'h04);  // opcode
        send_byte(8'h00);  // ts_hi=0
        send_byte(8'h05);  // ts_lo=5
        // RUN response: 0xDD + 4 bytes spike count
        begin : test5_block
            reg [7:0] done_marker, s1, s2, s3, s4;
            reg [31:0] spike_count;
            recv_byte(done_marker);
            recv_byte(s1);
            recv_byte(s2);
            recv_byte(s3);
            recv_byte(s4);
            spike_count = {s1, s2, s3, s4};
            $display("  RUN done=0x%02X, spikes=%0d", done_marker, spike_count);
            if (done_marker == 8'hDD && spike_count > 0) begin
                $display("  PASSED: Full spike chain via AXI bridge (spikes=%0d)", spike_count);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAILED: done=0x%02X spikes=%0d", done_marker, spike_count);
                fail_count = fail_count + 1;
            end
        end

        $display("\n=== F2 INTEGRATION RESULTS: %0d passed, %0d failed out of %0d ===",
                 pass_count, fail_count, pass_count + fail_count);
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");

        #100;
        $finish;
    end

    initial begin
        #10_000_000;  // 10 ms sim time — mesh needs many cycles
        $display("ERROR: Testbench timed out!");
        $finish;
    end

endmodule
