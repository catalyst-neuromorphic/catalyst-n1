// ============================================================================
// Testbench: AXI-UART Bridge
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

module tb_axi_uart_bridge;

    reg clk, rst_n;
    initial clk = 0;
    always #5 clk = ~clk;  // 100 MHz

    reg  [31:0] axi_awaddr, axi_wdata, axi_araddr;
    reg  [3:0]  axi_wstrb;
    reg         axi_awvalid, axi_wvalid, axi_arvalid, axi_bready, axi_rready;
    wire        axi_awready, axi_wready, axi_arready, axi_bvalid, axi_rvalid;
    wire [1:0]  axi_bresp, axi_rresp;
    wire [31:0] axi_rdata;

    wire [7:0]  hi_rx_data;
    wire        hi_rx_valid;
    wire [7:0]  hi_tx_data;
    wire        hi_tx_valid;
    wire        hi_tx_ready;

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
        .hi_rx_data     (hi_rx_data),
        .hi_rx_valid    (hi_rx_valid),
        .hi_tx_data     (hi_tx_data),
        .hi_tx_valid    (hi_tx_valid),
        .hi_tx_ready    (hi_tx_ready)
    );

    wire        mesh_start;
    wire        mesh_timestep_done;
    wire [5:0]  mesh_state;
    wire [31:0] mesh_total_spikes;
    wire [31:0] mesh_timestep_count;

    assign mesh_timestep_done = 1'b0;
    assign mesh_state         = 6'd0;
    assign mesh_total_spikes  = 32'd42;
    assign mesh_timestep_count = 32'd100;

    host_interface #(
        .NUM_CORES      (4),
        .CORE_ID_BITS   (2),
        .NUM_NEURONS    (256),
        .NEURON_BITS    (8),
        .DATA_WIDTH     (16),
        .POOL_ADDR_BITS (13),
        .COUNT_BITS     (6),
        .ROUTE_SLOT_BITS(3),
        .GLOBAL_ROUTE_SLOT_BITS(2)
    ) u_host_if (
        .clk                (clk),
        .rst_n              (rst_n),
        .rx_data            (hi_rx_data),
        .rx_valid           (hi_rx_valid),
        .tx_data            (hi_tx_data),
        .tx_valid           (hi_tx_valid),
        .tx_ready           (hi_tx_ready),
        .mesh_start         (mesh_start),
        .mesh_prog_pool_we  (),
        .mesh_prog_pool_core(),
        .mesh_prog_pool_addr(),
        .mesh_prog_pool_src (),
        .mesh_prog_pool_target(),
        .mesh_prog_pool_weight(),
        .mesh_prog_pool_comp  (),
        .mesh_prog_index_we   (),
        .mesh_prog_index_core  (),
        .mesh_prog_index_neuron(),
        .mesh_prog_index_base  (),
        .mesh_prog_index_count (),
        .mesh_prog_index_format(),
        .mesh_prog_route_we        (),
        .mesh_prog_route_src_core   (),
        .mesh_prog_route_src_neuron (),
        .mesh_prog_route_slot       (),
        .mesh_prog_route_dest_core  (),
        .mesh_prog_route_dest_neuron(),
        .mesh_prog_route_weight     (),
        .mesh_prog_global_route_we          (),
        .mesh_prog_global_route_src_core    (),
        .mesh_prog_global_route_src_neuron  (),
        .mesh_prog_global_route_slot        (),
        .mesh_prog_global_route_dest_core   (),
        .mesh_prog_global_route_dest_neuron (),
        .mesh_prog_global_route_weight      (),
        .mesh_ext_valid     (),
        .mesh_ext_core      (),
        .mesh_ext_neuron_id (),
        .mesh_ext_current   (),
        .mesh_learn_enable  (),
        .mesh_graded_enable (),
        .mesh_dendritic_enable(),
        .mesh_async_enable  (),
        .mesh_threefactor_enable(),
        .mesh_noise_enable  (),
        .mesh_skip_idle_enable(),
        .mesh_scale_u_enable(),
        .mesh_reward_value  (),
        .mesh_prog_delay_we (),
        .mesh_prog_delay_core(),
        .mesh_prog_delay_addr(),
        .mesh_prog_delay_value(),
        .mesh_prog_ucode_we (),
        .mesh_prog_ucode_core(),
        .mesh_prog_ucode_addr(),
        .mesh_prog_ucode_data(),
        .mesh_prog_param_we (),
        .mesh_prog_param_core(),
        .mesh_prog_param_neuron(),
        .mesh_prog_param_id (),
        .mesh_prog_param_value(),
        .mesh_probe_read    (),
        .mesh_probe_core    (),
        .mesh_probe_neuron  (),
        .mesh_probe_state_id(),
        .mesh_probe_pool_addr(),
        .mesh_probe_data    (16'sd0),
        .mesh_probe_valid   (1'b0),
        .mesh_dvfs_stall    (),
        .mesh_timestep_done (mesh_timestep_done),
        .mesh_state         (mesh_state),
        .mesh_total_spikes  (mesh_total_spikes),
        .mesh_timestep_count(mesh_timestep_count)
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

            // Wait for AW+W handshake
            @(posedge clk);
            while (!(axi_awready || axi_wready))
                @(posedge clk);
            @(posedge clk);
            axi_awvalid <= 1'b0;
            axi_wvalid  <= 1'b0;

            // Wait for B response
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

            // Wait for AR handshake
            @(posedge clk);
            while (!axi_arready)
                @(posedge clk);
            @(posedge clk);
            axi_arvalid <= 1'b0;

            // Wait for R response
            while (!axi_rvalid)
                @(posedge clk);
            data = axi_rdata;
            @(posedge clk);
            axi_rready <= 1'b0;
        end
    endtask

    // Send a byte to host_interface via bridge TX_DATA register
    task send_byte;
        input [7:0] b;
        reg [31:0] status;
        begin
            // Poll TX_STATUS until ready
            status = 0;
            while (!(status & 1)) begin
                axi_read(32'h004, status);
            end
            axi_write(32'h000, {24'd0, b});
        end
    endtask

    // Receive a byte from host_interface via bridge RX_DATA register
    task recv_byte;
        output [7:0] b;
        reg [31:0] status, data;
        begin
            // Poll RX_STATUS until not empty
            status = 0;
            while (!(status & 1)) begin
                axi_read(32'h00C, status);
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

        repeat (10) @(posedge clk);
        rst_n = 1;
        repeat (5) @(posedge clk);

        $display("\n--- TEST 1: SCRATCH register loopback ---");
        axi_write(32'h018, 32'hDEADBEEF);
        repeat (2) @(posedge clk);
        axi_read(32'h018, rd_data);
        if (rd_data == 32'hDEADBEEF) begin
            $display("  PASSED: SCRATCH = 0x%08X", rd_data);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: SCRATCH = 0x%08X (expected 0xDEADBEEF)", rd_data);
            fail_count = fail_count + 1;
        end

        $display("\n--- TEST 2: VERSION register read ---");
        axi_read(32'h014, rd_data);
        if (rd_data == 32'hF2020380) begin
            $display("  PASSED: VERSION = 0x%08X", rd_data);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: VERSION = 0x%08X (expected 0xF2020380)", rd_data);
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

        $display("\n--- TEST 4: TX_STATUS ready when empty ---");
        axi_read(32'h004, rd_data);
        if (rd_data[0] == 1'b1) begin
            $display("  PASSED: TX_STATUS ready = 1");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: TX_STATUS ready = 0 (expected 1)");
            fail_count = fail_count + 1;
        end

        $display("\n--- TEST 5: RX_STATUS empty initially ---");
        axi_read(32'h00C, rd_data);
        if (rd_data[0] == 1'b0) begin
            $display("  PASSED: RX_STATUS empty = 0 (not_empty bit)");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: RX_STATUS = 0x%08X (expected bit[0]=0)", rd_data);
            fail_count = fail_count + 1;
        end

        // Send CMD_STATUS (0x05, 0 payload) → expect 5-byte response
        $display("\n--- TEST 6: STATUS command via bridge ---");
        send_byte(8'h05);

        // Wait for host_interface to process and respond
        repeat (50) @(posedge clk);

        axi_read(32'h00C, rd_data);
        $display("  DEBUG: RX_STATUS after wait = 0x%08X (count=%0d, not_empty=%0d)",
                 rd_data, rd_data[5:1], rd_data[0]);

        // Read 5 response bytes: state(1) + timestep_count(4)
        recv_byte(rx_byte);
        $display("  Response byte 0 (state): 0x%02X", rx_byte);

        begin : status_block
            reg [31:0] ts_count;
            reg [7:0] b1, b2, b3, b4;
            recv_byte(b1);
            recv_byte(b2);
            recv_byte(b3);
            recv_byte(b4);
            ts_count = {b1, b2, b3, b4};
            $display("  Response bytes 1-4 (ts_count): %0d", ts_count);
            if (ts_count == 100) begin
                $display("  PASSED: STATUS response correct (ts_count=100)");
                pass_count = pass_count + 1;
            end else begin
                $display("  FAILED: ts_count=%0d (expected 100)", ts_count);
                fail_count = fail_count + 1;
            end
        end

        // CMD_PROG_POOL=0x01, 8 payload bytes
        $display("\n--- TEST 7: PROG_POOL command → ACK ---");
        send_byte(8'h01);  // opcode
        send_byte(8'h00);  // core=0
        send_byte(8'h00);  // addr_hi=0
        send_byte(8'h00);  // addr_lo=0
        send_byte(8'h00);  // flags=0
        send_byte(8'h00);  // src_lo=0
        send_byte(8'h01);  // tgt_lo=1
        send_byte(8'h04);  // wt_hi
        send_byte(8'hB0);  // wt_lo (weight=1200)

        repeat (30) @(posedge clk);
        recv_byte(rx_byte);
        if (rx_byte == 8'hAA) begin
            $display("  PASSED: Got ACK (0xAA)");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: Got 0x%02X (expected 0xAA)", rx_byte);
            fail_count = fail_count + 1;
        end

        $display("\n--- TEST 8: Soft reset ---");
        // Write some bytes into TX FIFO
        axi_write(32'h000, 32'hFF);
        axi_write(32'h000, 32'hFE);
        repeat (5) @(posedge clk);

        axi_write(32'h010, 32'h01);
        repeat (10) @(posedge clk);

        // Check RX FIFO is empty after reset
        axi_read(32'h00C, rd_data);
        if (rd_data[0] == 1'b0) begin
            $display("  PASSED: RX FIFO empty after soft reset");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: RX FIFO not empty after reset (0x%08X)", rd_data);
            fail_count = fail_count + 1;
        end

        $display("\n=== AXI-UART BRIDGE RESULTS: %0d passed, %0d failed out of %0d ===",
                 pass_count, fail_count, pass_count + fail_count);
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");

        #100;
        $finish;
    end

    initial begin
        #500000;
        $display("ERROR: Testbench timed out!");
        $finish;
    end

endmodule
