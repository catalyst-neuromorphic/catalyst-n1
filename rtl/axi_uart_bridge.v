module axi_uart_bridge #(
    parameter VERSION_ID = 32'hF2_02_03_10,
    parameter NUM_CORES  = 16
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        clk_neuro,
    input  wire        rst_neuro_n,

    input  wire [31:0] s_axi_awaddr,
    input  wire        s_axi_awvalid,
    output reg         s_axi_awready,
    input  wire [31:0] s_axi_wdata,
    input  wire [3:0]  s_axi_wstrb,
    input  wire        s_axi_wvalid,
    output reg         s_axi_wready,
    output reg  [1:0]  s_axi_bresp,
    output reg         s_axi_bvalid,
    input  wire        s_axi_bready,
    input  wire [31:0] s_axi_araddr,
    input  wire        s_axi_arvalid,
    output reg         s_axi_arready,
    output reg  [31:0] s_axi_rdata,
    output reg  [1:0]  s_axi_rresp,
    output reg         s_axi_rvalid,
    input  wire        s_axi_rready,

    output reg  [7:0]  hi_rx_data,
    output reg         hi_rx_valid,
    input  wire [7:0]  hi_tx_data,
    input  wire        hi_tx_valid,
    output wire        hi_tx_ready
);

    localparam REG_TX_DATA    = 3'd0;
    localparam REG_TX_STATUS  = 3'd1;
    localparam REG_RX_DATA    = 3'd2;
    localparam REG_RX_STATUS  = 3'd3;
    localparam REG_CONTROL    = 3'd4;
    localparam REG_VERSION    = 3'd5;
    localparam REG_SCRATCH    = 3'd6;
    localparam REG_CORE_COUNT = 3'd7;

    wire       tx_wr_full;
    wire       tx_rd_empty;
    wire [7:0] tx_rd_data;
    reg        tx_rd_en;
    reg        tx_wr_en;
    reg  [7:0] tx_wr_data;

    async_fifo #(.DATA_WIDTH(8), .ADDR_BITS(5)) u_tx_fifo (
        .wr_clk   (clk),
        .wr_rst_n (rst_n),
        .wr_data  (tx_wr_data),
        .wr_en    (tx_wr_en),
        .wr_full  (tx_wr_full),
        .rd_clk   (clk_neuro),
        .rd_rst_n (rst_neuro_n),
        .rd_en    (tx_rd_en),
        .rd_data  (tx_rd_data),
        .rd_empty (tx_rd_empty)
    );

    wire       rx_wr_full;
    wire       rx_rd_empty;
    wire [7:0] rx_rd_data;
    reg        rx_rd_en;
    reg        rx_wr_en;
    reg  [7:0] rx_wr_data;

    async_fifo #(.DATA_WIDTH(8), .ADDR_BITS(5)) u_rx_fifo (
        .wr_clk   (clk_neuro),
        .wr_rst_n (rst_neuro_n),
        .wr_data  (rx_wr_data),
        .wr_en    (rx_wr_en),
        .wr_full  (rx_wr_full),
        .rd_clk   (clk),
        .rd_rst_n (rst_n),
        .rd_en    (rx_rd_en),
        .rd_data  (rx_rd_data),
        .rd_empty (rx_rd_empty)
    );

    always @(posedge clk_neuro or negedge rst_neuro_n) begin
        if (!rst_neuro_n) begin
            hi_rx_data  <= 8'd0;
            hi_rx_valid <= 1'b0;
            tx_rd_en    <= 1'b0;
        end else begin
            hi_rx_valid <= 1'b0;
            tx_rd_en    <= 1'b0;
            if (!tx_rd_empty && !hi_rx_valid) begin
                hi_rx_data  <= tx_rd_data;
                hi_rx_valid <= 1'b1;
                tx_rd_en    <= 1'b1;
            end
        end
    end

    reg [1:0] rx_holdoff;
    reg       tx_ready_prev;

    wire internal_tx_ready = ~rx_wr_full & (rx_holdoff == 0);
    wire tx_ready_rising   = internal_tx_ready & ~tx_ready_prev;
    wire do_rx_capture     = hi_tx_valid & internal_tx_ready & ~tx_ready_rising;

    assign hi_tx_ready = internal_tx_ready;

    always @(posedge clk_neuro or negedge rst_neuro_n) begin
        if (!rst_neuro_n) begin
            rx_holdoff    <= 2'd0;
            tx_ready_prev <= 1'b1;
            rx_wr_en      <= 1'b0;
            rx_wr_data    <= 8'd0;
        end else begin
            tx_ready_prev <= internal_tx_ready;
            rx_wr_en      <= 1'b0;

            if (rx_holdoff != 0)
                rx_holdoff <= rx_holdoff - 1;

            if (do_rx_capture) begin
                rx_wr_data <= hi_tx_data;
                rx_wr_en   <= 1'b1;
                rx_holdoff <= 2'd2;
            end
        end
    end

    reg [31:0] scratch_reg;

    localparam S_IDLE       = 2'd0;
    localparam S_WRITE_RESP = 2'd1;
    localparam S_READ_RESP  = 2'd2;

    reg [1:0]  axi_state;
    reg [2:0]  wr_reg_addr;
    reg [31:0] wr_data_reg;
    reg [2:0]  rd_reg_addr;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            axi_state     <= S_IDLE;
            s_axi_awready <= 1'b0;
            s_axi_wready  <= 1'b0;
            s_axi_bvalid  <= 1'b0;
            s_axi_bresp   <= 2'b00;
            s_axi_arready <= 1'b0;
            s_axi_rvalid  <= 1'b0;
            s_axi_rdata   <= 32'd0;
            s_axi_rresp   <= 2'b00;
            scratch_reg   <= 32'd0;
            wr_reg_addr   <= 3'd0;
            wr_data_reg   <= 32'd0;
            rd_reg_addr   <= 3'd0;
            tx_wr_en      <= 1'b0;
            tx_wr_data    <= 8'd0;
            rx_rd_en      <= 1'b0;
        end else begin
            tx_wr_en <= 1'b0;
            rx_rd_en <= 1'b0;

            case (axi_state)
                S_IDLE: begin
                    s_axi_bvalid <= 1'b0;
                    s_axi_rvalid <= 1'b0;

                    if (s_axi_awvalid && s_axi_wvalid) begin
                        s_axi_awready <= 1'b1;
                        s_axi_wready  <= 1'b1;
                        wr_reg_addr   <= s_axi_awaddr[4:2];
                        wr_data_reg   <= s_axi_wdata;
                        axi_state     <= S_WRITE_RESP;
                    end else if (s_axi_arvalid) begin
                        s_axi_arready <= 1'b1;
                        rd_reg_addr   <= s_axi_araddr[4:2];
                        axi_state     <= S_READ_RESP;
                    end
                end

                S_WRITE_RESP: begin
                    s_axi_awready <= 1'b0;
                    s_axi_wready  <= 1'b0;

                    if (!s_axi_bvalid) begin
                        case (wr_reg_addr)
                            REG_TX_DATA: begin
                                if (!tx_wr_full) begin
                                    tx_wr_data <= wr_data_reg[7:0];
                                    tx_wr_en   <= 1'b1;
                                end
                            end
                            REG_SCRATCH: scratch_reg <= wr_data_reg;
                            default: ;
                        endcase
                        s_axi_bvalid <= 1'b1;
                        s_axi_bresp  <= 2'b00;
                    end

                    if (s_axi_bvalid && s_axi_bready)
                        axi_state <= S_IDLE;
                end

                S_READ_RESP: begin
                    s_axi_arready <= 1'b0;

                    if (!s_axi_rvalid) begin
                        case (rd_reg_addr)
                            REG_TX_DATA:    s_axi_rdata <= 32'd0;
                            REG_TX_STATUS:  s_axi_rdata <= {31'd0, ~tx_wr_full};
                            REG_RX_DATA: begin
                                if (!rx_rd_empty) begin
                                    s_axi_rdata <= {24'd0, rx_rd_data};
                                    rx_rd_en    <= 1'b1;
                                end else begin
                                    s_axi_rdata <= 32'd0;
                                end
                            end
                            REG_RX_STATUS:  s_axi_rdata <= {31'd0, ~rx_rd_empty};
                            REG_CONTROL:    s_axi_rdata <= 32'd0;
                            REG_VERSION:    s_axi_rdata <= VERSION_ID;
                            REG_SCRATCH:    s_axi_rdata <= scratch_reg;
                            REG_CORE_COUNT: s_axi_rdata <= NUM_CORES;
                        endcase
                        s_axi_rvalid <= 1'b1;
                        s_axi_rresp  <= 2'b00;
                    end

                    if (s_axi_rvalid && s_axi_rready)
                        axi_state <= S_IDLE;
                end

                default: axi_state <= S_IDLE;
            endcase
        end
    end

endmodule
