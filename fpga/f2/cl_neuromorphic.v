// ============================================================================
// CL Top-Level — AWS F2 Shell ↔ Neuromorphic Chip
// ============================================================================
//
// Wraps the 128-core neuromorphic system for the AWS F2 FPGA (VU47P).
//
// Active interfaces:
//   - OCL AXI-Lite (BAR0): Host MMIO → axi_uart_bridge → host_interface
//
// All other Shell interfaces (PCIM, PCIS/DMA, SDA, DDR, HBM, interrupts)
// are tied off as unused.
//
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

`include "cl_neuromorphic_defines.vh"

module cl_neuromorphic (
    input  wire        clk_main_a0,
    input  wire        rst_main_n,

    output wire [31:0] cl_sh_id0,
    output wire [31:0] cl_sh_id1,

    input  wire [31:0] sh_ocl_awaddr,
    input  wire        sh_ocl_awvalid,
    output wire        ocl_sh_awready,
    input  wire [31:0] sh_ocl_wdata,
    input  wire [3:0]  sh_ocl_wstrb,
    input  wire        sh_ocl_wvalid,
    output wire        ocl_sh_wready,
    output wire [1:0]  ocl_sh_bresp,
    output wire        ocl_sh_bvalid,
    input  wire        sh_ocl_bready,
    input  wire [31:0] sh_ocl_araddr,
    input  wire        sh_ocl_arvalid,
    output wire        ocl_sh_arready,
    output wire [31:0] ocl_sh_rdata,
    output wire [1:0]  ocl_sh_rresp,
    output wire        ocl_sh_rvalid,
    input  wire        sh_ocl_rready,

    input  wire [31:0] sh_sda_awaddr,
    input  wire        sh_sda_awvalid,
    output wire        sda_sh_awready,
    input  wire [31:0] sh_sda_wdata,
    input  wire [3:0]  sh_sda_wstrb,
    input  wire        sh_sda_wvalid,
    output wire        sda_sh_wready,
    output wire [1:0]  sda_sh_bresp,
    output wire        sda_sh_bvalid,
    input  wire        sh_sda_bready,
    input  wire [31:0] sh_sda_araddr,
    input  wire        sh_sda_arvalid,
    output wire        sda_sh_arready,
    output wire [31:0] sda_sh_rdata,
    output wire [1:0]  sda_sh_rresp,
    output wire        sda_sh_rvalid,
    input  wire        sh_sda_rready,

    output wire [63:0] cl_sh_pcim_awaddr,
    output wire [15:0] cl_sh_pcim_awid,
    output wire [7:0]  cl_sh_pcim_awlen,
    output wire [2:0]  cl_sh_pcim_awsize,
    output wire        cl_sh_pcim_awvalid,
    input  wire        sh_cl_pcim_awready,
    output wire [511:0] cl_sh_pcim_wdata,
    output wire [63:0] cl_sh_pcim_wstrb,
    output wire        cl_sh_pcim_wlast,
    output wire        cl_sh_pcim_wvalid,
    input  wire        sh_cl_pcim_wready,
    input  wire [1:0]  sh_cl_pcim_bresp,
    input  wire [15:0] sh_cl_pcim_bid,
    input  wire        sh_cl_pcim_bvalid,
    output wire        cl_sh_pcim_bready,
    output wire [63:0] cl_sh_pcim_araddr,
    output wire [15:0] cl_sh_pcim_arid,
    output wire [7:0]  cl_sh_pcim_arlen,
    output wire [2:0]  cl_sh_pcim_arsize,
    output wire        cl_sh_pcim_arvalid,
    input  wire        sh_cl_pcim_arready,
    input  wire [511:0] sh_cl_pcim_rdata,
    input  wire [15:0] sh_cl_pcim_rid,
    input  wire [1:0]  sh_cl_pcim_rresp,
    input  wire        sh_cl_pcim_rlast,
    input  wire        sh_cl_pcim_rvalid,
    output wire        cl_sh_pcim_rready,

    input  wire [63:0] sh_cl_dma_pcis_awaddr,
    input  wire [15:0] sh_cl_dma_pcis_awid,
    input  wire [7:0]  sh_cl_dma_pcis_awlen,
    input  wire [2:0]  sh_cl_dma_pcis_awsize,
    input  wire        sh_cl_dma_pcis_awvalid,
    output wire        cl_sh_dma_pcis_awready,
    input  wire [511:0] sh_cl_dma_pcis_wdata,
    input  wire [63:0] sh_cl_dma_pcis_wstrb,
    input  wire        sh_cl_dma_pcis_wlast,
    input  wire        sh_cl_dma_pcis_wvalid,
    output wire        cl_sh_dma_pcis_wready,
    output wire [1:0]  cl_sh_dma_pcis_bresp,
    output wire [15:0] cl_sh_dma_pcis_bid,
    output wire        cl_sh_dma_pcis_bvalid,
    input  wire        sh_cl_dma_pcis_bready,
    input  wire [63:0] sh_cl_dma_pcis_araddr,
    input  wire [15:0] sh_cl_dma_pcis_arid,
    input  wire [7:0]  sh_cl_dma_pcis_arlen,
    input  wire [2:0]  sh_cl_dma_pcis_arsize,
    input  wire        sh_cl_dma_pcis_arvalid,
    output wire        cl_sh_dma_pcis_arready,
    output wire [511:0] cl_sh_dma_pcis_rdata,
    output wire [15:0] cl_sh_dma_pcis_rid,
    output wire [1:0]  cl_sh_dma_pcis_rresp,
    output wire        cl_sh_dma_pcis_rlast,
    output wire        cl_sh_dma_pcis_rvalid,
    input  wire        sh_cl_dma_pcis_rready,

    input  wire        sh_cl_ddr_stat_wr,
    input  wire        sh_cl_ddr_stat_rd,
    input  wire [7:0]  sh_cl_ddr_stat_addr,
    input  wire [31:0] sh_cl_ddr_stat_wdata,
    output wire        cl_sh_ddr_stat_ack,
    output wire [31:0] cl_sh_ddr_stat_rdata,
    output wire [7:0]  cl_sh_ddr_stat_int,

    output wire [15:0] cl_sh_apppf_irq_req,
    input  wire [15:0] sh_cl_apppf_irq_ack,

    input  wire        sh_cl_flr_assert,
    output wire        cl_sh_flr_done,

    output wire [31:0] cl_sh_status0,
    output wire [31:0] cl_sh_status1
);

    assign cl_sh_id0 = `CL_SH_ID0;
    assign cl_sh_id1 = `CL_SH_ID1;

    assign cl_sh_status0 = 32'h0000_0001;  // bit 0 = CL alive
    assign cl_sh_status1 = 32'd128;         // core count

    // SDA — not used (management register space)
    assign sda_sh_awready  = 1'b0;
    assign sda_sh_wready   = 1'b0;
    assign sda_sh_bresp    = 2'b00;
    assign sda_sh_bvalid   = 1'b0;
    assign sda_sh_arready  = 1'b0;
    assign sda_sh_rdata    = 32'd0;
    assign sda_sh_rresp    = 2'b00;
    assign sda_sh_rvalid   = 1'b0;

    // PCIM — not used (no CL-initiated DMA)
    assign cl_sh_pcim_awaddr  = 64'd0;
    assign cl_sh_pcim_awid    = 16'd0;
    assign cl_sh_pcim_awlen   = 8'd0;
    assign cl_sh_pcim_awsize  = 3'd0;
    assign cl_sh_pcim_awvalid = 1'b0;
    assign cl_sh_pcim_wdata   = 512'd0;
    assign cl_sh_pcim_wstrb   = 64'd0;
    assign cl_sh_pcim_wlast   = 1'b0;
    assign cl_sh_pcim_wvalid  = 1'b0;
    assign cl_sh_pcim_bready  = 1'b1;  // Accept any write response
    assign cl_sh_pcim_araddr  = 64'd0;
    assign cl_sh_pcim_arid    = 16'd0;
    assign cl_sh_pcim_arlen   = 8'd0;
    assign cl_sh_pcim_arsize  = 3'd0;
    assign cl_sh_pcim_arvalid = 1'b0;
    assign cl_sh_pcim_rready  = 1'b1;  // Accept any read data

    // PCIS (DMA) — not used (no host DMA writes to CL)
    assign cl_sh_dma_pcis_awready = 1'b0;
    assign cl_sh_dma_pcis_wready  = 1'b0;
    assign cl_sh_dma_pcis_bresp   = 2'b00;
    assign cl_sh_dma_pcis_bid     = 16'd0;
    assign cl_sh_dma_pcis_bvalid  = 1'b0;
    assign cl_sh_dma_pcis_arready = 1'b0;
    assign cl_sh_dma_pcis_rdata   = 512'd0;
    assign cl_sh_dma_pcis_rid     = 16'd0;
    assign cl_sh_dma_pcis_rresp   = 2'b00;
    assign cl_sh_dma_pcis_rlast   = 1'b0;
    assign cl_sh_dma_pcis_rvalid  = 1'b0;

    // DDR stat — ack any request, return 0
    assign cl_sh_ddr_stat_ack   = sh_cl_ddr_stat_wr | sh_cl_ddr_stat_rd;
    assign cl_sh_ddr_stat_rdata = 32'd0;
    assign cl_sh_ddr_stat_int   = 8'd0;

    // Interrupts — none
    assign cl_sh_apppf_irq_req = 16'd0;

    // FLR — immediate acknowledge
    assign cl_sh_flr_done = sh_cl_flr_assert;

    wire [7:0] bridge_rx_data;
    wire       bridge_rx_valid;
    wire [7:0] bridge_tx_data;
    wire       bridge_tx_valid;
    wire       bridge_tx_ready;

    axi_uart_bridge #(
        .FIFO_DEPTH   (32),
        .VERSION_ID   (32'hF2_02_03_80),  // F2, v2.3, 128-core
        .NUM_CORES    (128)
    ) u_bridge (
        .clk           (clk_main_a0),
        .rst_n         (rst_main_n),

        // AXI-Lite slave ← Shell OCL master
        .s_axi_awaddr  (sh_ocl_awaddr),
        .s_axi_awvalid (sh_ocl_awvalid),
        .s_axi_awready (ocl_sh_awready),
        .s_axi_wdata   (sh_ocl_wdata),
        .s_axi_wstrb   (sh_ocl_wstrb),
        .s_axi_wvalid  (sh_ocl_wvalid),
        .s_axi_wready  (ocl_sh_wready),
        .s_axi_bresp   (ocl_sh_bresp),
        .s_axi_bvalid  (ocl_sh_bvalid),
        .s_axi_bready  (sh_ocl_bready),
        .s_axi_araddr  (sh_ocl_araddr),
        .s_axi_arvalid (sh_ocl_arvalid),
        .s_axi_arready (ocl_sh_arready),
        .s_axi_rdata   (ocl_sh_rdata),
        .s_axi_rresp   (ocl_sh_rresp),
        .s_axi_rvalid  (ocl_sh_rvalid),
        .s_axi_rready  (sh_ocl_rready),

        // Byte-stream to neuromorphic_top
        .hi_rx_data    (bridge_rx_data),
        .hi_rx_valid   (bridge_rx_valid),
        .hi_tx_data    (bridge_tx_data),
        .hi_tx_valid   (bridge_tx_valid),
        .hi_tx_ready   (bridge_tx_ready)
    );

    neuromorphic_top #(
        .CLK_FREQ       (250_000_000),  // F2 clk_main_a0 = 250 MHz
        .BAUD           (115200),       // Unused (BYPASS_UART=1)
        .BYPASS_UART    (1),
        .NUM_CORES      (128),
        .CORE_ID_BITS   (12),
        .NUM_NEURONS    (1024),
        .NEURON_BITS    (10),
        .DATA_WIDTH     (16),
        .POOL_DEPTH     (8192),         // 8K/core × 128 cores = 1M total
        .POOL_ADDR_BITS (13),
        .COUNT_BITS     (12),
        .REV_FANIN      (32),
        .REV_SLOT_BITS  (5),
        .THRESHOLD      (16'sd1000),
        .LEAK_RATE      (16'sd3),
        .REFRAC_CYCLES  (3),
        .ROUTE_FANOUT           (8),
        .ROUTE_SLOT_BITS        (3),
        .GLOBAL_ROUTE_SLOTS     (4),
        .GLOBAL_ROUTE_SLOT_BITS (2),
        .CHIP_LINK_EN   (0),
        .NOC_MODE       (0),           // Barrier mesh (deterministic)
        .MESH_X         (16),          // 16×8 = 128 cores
        .MESH_Y         (8)
    ) u_neuromorphic (
        .clk            (clk_main_a0),
        .rst_n          (rst_main_n),

        // UART — unused (BYPASS_UART=1)
        .uart_rxd       (1'b1),
        .uart_txd       (),

        // Byte-stream from AXI bridge
        .rx_data_ext    (bridge_rx_data),
        .rx_valid_ext   (bridge_rx_valid),
        .tx_data_ext    (bridge_tx_data),
        .tx_valid_ext   (bridge_tx_valid),
        .tx_ready_ext   (bridge_tx_ready),

        // Chip link — disabled
        .link_tx_data   (),
        .link_tx_valid  (),
        .link_tx_ready  (1'b0),
        .link_rx_data   (8'd0),
        .link_rx_valid  (1'b0),
        .link_rx_ready  ()
    );

endmodule
