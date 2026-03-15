module kria_neuromorphic #(
    parameter NUM_CORES      = 2,
    parameter CORE_ID_BITS   = 1,
    parameter NUM_NEURONS    = 256,
    parameter NEURON_BITS    = 8,
    parameter POOL_DEPTH     = 4096,
    parameter POOL_ADDR_BITS = 12,
    parameter COUNT_BITS     = 8,
    parameter VERSION_ID     = 32'hA0_23_02_01
)(
    input  wire        s_axi_aclk,
    input  wire        s_axi_aresetn,
    input  wire [31:0] s_axi_awaddr,
    input  wire        s_axi_awvalid,
    output wire        s_axi_awready,
    input  wire [31:0] s_axi_wdata,
    input  wire [3:0]  s_axi_wstrb,
    input  wire        s_axi_wvalid,
    output wire        s_axi_wready,
    output wire [1:0]  s_axi_bresp,
    output wire        s_axi_bvalid,
    input  wire        s_axi_bready,
    input  wire [31:0] s_axi_araddr,
    input  wire        s_axi_arvalid,
    output wire        s_axi_arready,
    output wire [31:0] s_axi_rdata,
    output wire [1:0]  s_axi_rresp,
    output wire        s_axi_rvalid,
    input  wire        s_axi_rready
);

    wire clk   = s_axi_aclk;
    wire rst_n = s_axi_aresetn;

    wire [7:0] bridge_rx_data;
    wire       bridge_rx_valid;
    wire [7:0] bridge_tx_data;
    wire       bridge_tx_valid;
    wire       bridge_tx_ready;

    axi_uart_bridge #(
        .VERSION_ID (VERSION_ID),
        .NUM_CORES  (NUM_CORES)
    ) u_bridge (
        .clk          (clk),
        .rst_n        (rst_n),
        .clk_neuro    (clk),
        .rst_neuro_n  (rst_n),
        .s_axi_awaddr (s_axi_awaddr),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready),
        .s_axi_wdata  (s_axi_wdata),
        .s_axi_wstrb  (s_axi_wstrb),
        .s_axi_wvalid (s_axi_wvalid),
        .s_axi_wready (s_axi_wready),
        .s_axi_bresp  (s_axi_bresp),
        .s_axi_bvalid (s_axi_bvalid),
        .s_axi_bready (s_axi_bready),
        .s_axi_araddr (s_axi_araddr),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready),
        .s_axi_rdata  (s_axi_rdata),
        .s_axi_rresp  (s_axi_rresp),
        .s_axi_rvalid (s_axi_rvalid),
        .s_axi_rready (s_axi_rready),
        .hi_rx_data   (bridge_rx_data),
        .hi_rx_valid  (bridge_rx_valid),
        .hi_tx_data   (bridge_tx_data),
        .hi_tx_valid  (bridge_tx_valid),
        .hi_tx_ready  (bridge_tx_ready)
    );

    neuromorphic_top #(
        .CLK_FREQ       (100_000_000),
        .BAUD           (115200),
        .BYPASS_UART    (1),
        .NUM_CORES      (NUM_CORES),
        .CORE_ID_BITS   (CORE_ID_BITS),
        .NUM_NEURONS    (NUM_NEURONS),
        .NEURON_BITS    (NEURON_BITS),
        .DATA_WIDTH     (16),
        .POOL_DEPTH     (POOL_DEPTH),
        .POOL_ADDR_BITS (POOL_ADDR_BITS),
        .COUNT_BITS     (COUNT_BITS),
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
        .MESH_Y         (1)
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
        .link_rx_data   (8'b0),
        .link_rx_valid  (1'b0),
        .link_rx_ready  ()
    );

endmodule
