module chip_link #(
    parameter CORE_ID_BITS = 7,
    parameter NEURON_BITS  = 10,
    parameter DATA_WIDTH   = 16,
    parameter TX_DEPTH     = 256,
    parameter RX_DEPTH     = 256
)(
    input  wire clk,
    input  wire rst_n,

    input  wire                        tx_push,
    input  wire [CORE_ID_BITS-1:0]     tx_core,
    input  wire [NEURON_BITS-1:0]      tx_neuron,
    input  wire [7:0]                  tx_payload,
    output wire                        tx_full,

    output wire [CORE_ID_BITS-1:0]     rx_core,
    output wire [NEURON_BITS-1:0]      rx_neuron,
    output wire signed [DATA_WIDTH-1:0] rx_current,
    input  wire                        rx_pop,
    output wire                        rx_empty,

    output reg  [7:0]                  link_tx_data,
    output reg                         link_tx_valid,
    input  wire                        link_tx_ready,

    input  wire [7:0]                  link_rx_data,
    input  wire                        link_rx_valid,
    output wire                        link_rx_ready
);

    localparam TX_PKT_W = CORE_ID_BITS + NEURON_BITS + 8;

    reg  [TX_PKT_W-1:0] tx_fifo [0:TX_DEPTH-1];
    reg  [8:0] tx_wr_ptr, tx_rd_ptr;
    wire [8:0] tx_count = tx_wr_ptr - tx_rd_ptr;
    wire        tx_empty_i = (tx_wr_ptr == tx_rd_ptr);
    assign      tx_full    = (tx_count >= TX_DEPTH);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            tx_wr_ptr <= 0;
        else if (tx_push && !tx_full) begin
            tx_fifo[tx_wr_ptr[7:0]] <= {tx_core, tx_neuron, tx_payload};
            tx_wr_ptr <= tx_wr_ptr + 1;
        end
    end

    localparam TX_IDLE = 2'd0, TX_BYTE1 = 2'd1, TX_BYTE2 = 2'd2, TX_BYTE3 = 2'd3;
    reg [1:0] tx_state;
    reg [TX_PKT_W-1:0] tx_pkt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx_state     <= TX_IDLE;
            tx_rd_ptr    <= 0;
            link_tx_valid <= 0;
            link_tx_data  <= 0;
        end else begin
            link_tx_valid <= 0;

            case (tx_state)
                TX_IDLE: begin
                    if (!tx_empty_i && link_tx_ready) begin
                        tx_pkt    <= tx_fifo[tx_rd_ptr[7:0]];
                        tx_rd_ptr <= tx_rd_ptr + 1;
                        link_tx_data  <= 8'h80 | tx_fifo[tx_rd_ptr[7:0]][TX_PKT_W-1 -: CORE_ID_BITS];
                        link_tx_valid <= 1;
                        tx_state      <= TX_BYTE1;
                    end
                end

                TX_BYTE1: begin
                    if (link_tx_ready) begin
                        link_tx_data  <= tx_pkt[NEURON_BITS+7:10];
                        link_tx_valid <= 1;
                        tx_state      <= TX_BYTE2;
                    end
                end

                TX_BYTE2: begin
                    if (link_tx_ready) begin
                        link_tx_data  <= {tx_pkt[9:8], tx_pkt[7:2]};
                        link_tx_valid <= 1;
                        tx_state      <= TX_BYTE3;
                    end
                end

                TX_BYTE3: begin
                    if (link_tx_ready) begin
                        link_tx_data  <= {tx_pkt[1:0], 6'd0};
                        link_tx_valid <= 1;
                        tx_state      <= TX_IDLE;
                    end
                end
            endcase
        end
    end

    localparam RX_PKT_W = CORE_ID_BITS + NEURON_BITS + DATA_WIDTH;

    localparam RX_IDLE = 2'd0, RX_BYTE1 = 2'd1, RX_BYTE2 = 2'd2, RX_BYTE3 = 2'd3;
    reg [1:0] rx_state;
    reg [CORE_ID_BITS-1:0]  rx_pkt_core;
    reg [NEURON_BITS-1:0]   rx_pkt_neuron;
    reg [7:0]               rx_pkt_payload;
    reg                     rx_push;

    assign link_rx_ready = (rx_count < RX_DEPTH - 4);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_state <= RX_IDLE;
            rx_push  <= 0;
        end else begin
            rx_push <= 0;

            case (rx_state)
                RX_IDLE: begin
                    if (link_rx_valid && link_rx_data[7]) begin
                        rx_pkt_core <= link_rx_data[CORE_ID_BITS-1:0];
                        rx_state    <= RX_BYTE1;
                    end
                end

                RX_BYTE1: begin
                    if (link_rx_valid) begin
                        rx_pkt_neuron[NEURON_BITS-1:2] <= link_rx_data;
                        rx_state <= RX_BYTE2;
                    end
                end

                RX_BYTE2: begin
                    if (link_rx_valid) begin
                        rx_pkt_neuron[1:0]   <= link_rx_data[7:6];
                        rx_pkt_payload[7:2]  <= link_rx_data[5:0];
                        rx_state <= RX_BYTE3;
                    end
                end

                RX_BYTE3: begin
                    if (link_rx_valid) begin
                        rx_pkt_payload[1:0] <= link_rx_data[7:6];
                        rx_push <= 1;
                        rx_state <= RX_IDLE;
                    end
                end
            endcase
        end
    end

    reg  [RX_PKT_W-1:0] rx_fifo [0:RX_DEPTH-1];
    reg  [8:0] rx_wr_ptr, rx_rd_ptr;
    wire [8:0] rx_count = rx_wr_ptr - rx_rd_ptr;
    assign rx_empty = (rx_wr_ptr == rx_rd_ptr);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            rx_wr_ptr <= 0;
        else if (rx_push && rx_count < RX_DEPTH) begin
            rx_fifo[rx_wr_ptr[7:0]] <= {rx_pkt_core, rx_pkt_neuron,
                                         {{(DATA_WIDTH-8){1'b0}}, rx_pkt_payload}};
            rx_wr_ptr <= rx_wr_ptr + 1;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            rx_rd_ptr <= 0;
        else if (rx_pop && !rx_empty)
            rx_rd_ptr <= rx_rd_ptr + 1;
    end

    wire [RX_PKT_W-1:0] rx_top = rx_fifo[rx_rd_ptr[7:0]];
    assign rx_core    = rx_top[RX_PKT_W-1 -: CORE_ID_BITS];
    assign rx_neuron  = rx_top[DATA_WIDTH +: NEURON_BITS];
    assign rx_current = rx_top[DATA_WIDTH-1:0];

endmodule
