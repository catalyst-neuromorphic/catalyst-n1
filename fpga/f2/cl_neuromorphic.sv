module cl_neuromorphic
    #(
      parameter EN_DDR = 0,
      parameter EN_HBM = 0
    )
    (
      `include "cl_ports.vh"
    );

`include "cl_neuromorphic_defines.vh"

  logic rst_main_n_sync;
  always_ff @(negedge rst_main_n or posedge clk_main_a0)
    if (!rst_main_n) rst_main_n_sync <= 1'b0;
    else             rst_main_n_sync <= 1'b1;

  wire clk_neuro_unbuf;
  wire clk_neuro;
  wire mmcm_fb;
  wire mmcm_locked;

  MMCME4_BASE #(
      .CLKIN1_PERIOD   (4.000),
      .CLKFBOUT_MULT_F (4.000),
      .CLKOUT0_DIVIDE_F(16.000),
      .CLKOUT0_PHASE   (0.000),
      .DIVCLK_DIVIDE   (1)
  ) u_mmcm (
      .CLKIN1   (clk_main_a0),
      .CLKFBOUT (mmcm_fb),
      .CLKFBIN  (mmcm_fb),
      .CLKOUT0  (clk_neuro_unbuf),
      .CLKOUT0B (),
      .CLKOUT1  (),
      .CLKOUT1B (),
      .CLKOUT2  (),
      .CLKOUT2B (),
      .CLKOUT3  (),
      .CLKOUT3B (),
      .CLKOUT4  (),
      .CLKOUT5  (),
      .CLKOUT6  (),
      .LOCKED   (mmcm_locked),
      .PWRDWN   (1'b0),
      .RST      (~rst_main_n)
  );

  BUFG u_bufg_neuro (.I(clk_neuro_unbuf), .O(clk_neuro));

  logic rst_neuro_n_sync;
  logic rst_neuro_n_pipe;
  always_ff @(negedge mmcm_locked or posedge clk_neuro)
    if (!mmcm_locked) begin
      rst_neuro_n_pipe <= 1'b0;
      rst_neuro_n_sync <= 1'b0;
    end else begin
      rst_neuro_n_pipe <= rst_main_n;
      rst_neuro_n_sync <= rst_neuro_n_pipe;
    end

  assign cl_sh_flr_done    = 1'b1;
  assign cl_sh_status0     = {31'b0, mmcm_locked};
  assign cl_sh_status1     = 32'b0;
  assign cl_sh_status2     = 32'b0;
  assign cl_sh_id0         = `CL_SH_ID0;
  assign cl_sh_id1         = `CL_SH_ID1;
  assign cl_sh_status_vled = {15'b0, mmcm_locked};

  `include "unused_pcim_template.inc"

  `include "unused_dma_pcis_template.inc"

  `include "unused_cl_sda_template.inc"

  `include "unused_ddr_template.inc"

  `include "unused_apppf_irq_template.inc"

  assign tdo = 1'b0;

  assign hbm_apb_paddr_1   = 22'b0;
  assign hbm_apb_pprot_1   = 3'b0;
  assign hbm_apb_psel_1    = 1'b0;
  assign hbm_apb_penable_1 = 1'b0;
  assign hbm_apb_pwrite_1  = 1'b0;
  assign hbm_apb_pwdata_1  = 32'b0;
  assign hbm_apb_pstrb_1   = 4'b0;
  assign hbm_apb_pready_1  = 1'b0;
  assign hbm_apb_prdata_1  = 32'b0;
  assign hbm_apb_pslverr_1 = 1'b0;

  assign hbm_apb_paddr_0   = 22'b0;
  assign hbm_apb_pprot_0   = 3'b0;
  assign hbm_apb_psel_0    = 1'b0;
  assign hbm_apb_penable_0 = 1'b0;
  assign hbm_apb_pwrite_0  = 1'b0;
  assign hbm_apb_pwdata_0  = 32'b0;
  assign hbm_apb_pstrb_0   = 4'b0;
  assign hbm_apb_pready_0  = 1'b0;
  assign hbm_apb_prdata_0  = 32'b0;
  assign hbm_apb_pslverr_0 = 1'b0;

  assign PCIE_EP_TXP    = 8'b0;
  assign PCIE_EP_TXN    = 8'b0;
  assign PCIE_RP_PERSTN = 1'b0;
  assign PCIE_RP_TXP    = 8'b0;
  assign PCIE_RP_TXN    = 8'b0;

  wire [7:0]  bridge_rx_data;
  wire        bridge_rx_valid;
  wire [7:0]  bridge_tx_data;
  wire        bridge_tx_valid;
  wire        bridge_tx_ready;

  axi_uart_bridge #(
      .VERSION_ID (32'hF2_02_03_10),
      .NUM_CORES  (16)
  ) u_bridge (
      .clk          (clk_main_a0),
      .rst_n        (rst_main_n_sync),
      .clk_neuro    (clk_neuro),
      .rst_neuro_n  (rst_neuro_n_sync),

      .s_axi_awaddr (ocl_cl_awaddr),
      .s_axi_awvalid(ocl_cl_awvalid),
      .s_axi_awready(cl_ocl_awready),
      .s_axi_wdata  (ocl_cl_wdata),
      .s_axi_wstrb  (ocl_cl_wstrb),
      .s_axi_wvalid (ocl_cl_wvalid),
      .s_axi_wready (cl_ocl_wready),
      .s_axi_bresp  (cl_ocl_bresp),
      .s_axi_bvalid (cl_ocl_bvalid),
      .s_axi_bready (ocl_cl_bready),
      .s_axi_araddr (ocl_cl_araddr),
      .s_axi_arvalid(ocl_cl_arvalid),
      .s_axi_arready(cl_ocl_arready),
      .s_axi_rdata  (cl_ocl_rdata),
      .s_axi_rresp  (cl_ocl_rresp),
      .s_axi_rvalid (cl_ocl_rvalid),
      .s_axi_rready (ocl_cl_rready),

      .hi_rx_data   (bridge_rx_data),
      .hi_rx_valid  (bridge_rx_valid),
      .hi_tx_data   (bridge_tx_data),
      .hi_tx_valid  (bridge_tx_valid),
      .hi_tx_ready  (bridge_tx_ready)
  );

  neuromorphic_top #(
      .CLK_FREQ       (62_500_000),
      .BAUD           (115200),
      .BYPASS_UART    (1),
      .NUM_CORES      (16),
      .CORE_ID_BITS   (4),
      .NUM_NEURONS    (1024),
      .NEURON_BITS    (10),
      .POOL_DEPTH     (4096),
      .POOL_ADDR_BITS (12),
      .COUNT_BITS     (12),
      .CHIP_LINK_EN   (0),
      .NOC_MODE       (0),
      .MESH_X         (4),
      .MESH_Y         (4)
  ) u_neuromorphic (
      .clk            (clk_neuro),
      .rst_n          (rst_neuro_n_sync),

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
      .link_rx_data   (8'b0),
      .link_rx_valid  (1'b0),
      .link_rx_ready  ()
  );

endmodule
