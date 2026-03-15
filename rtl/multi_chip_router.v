`timescale 1ns/1ps

module multi_chip_router #(
    parameter NUM_LINKS    = 1,
    parameter CHIP_ID_BITS = 14,
    parameter CORE_ID_BITS = 7,
    parameter NEURON_BITS  = 10,
    parameter DATA_WIDTH   = 16,
    parameter TX_DEPTH     = 256,
    parameter RX_DEPTH     = 256
)(
    input  wire        clk,
    input  wire        rst_n,

    input  wire [CHIP_ID_BITS-1:0] my_chip_id,

    input  wire                        tx_push,
    input  wire [CHIP_ID_BITS-1:0]     tx_dest_chip,
    input  wire [CORE_ID_BITS-1:0]     tx_core,
    input  wire [NEURON_BITS-1:0]      tx_neuron,
    input  wire [7:0]                  tx_payload,
    output wire                        tx_full,

    output wire [CHIP_ID_BITS-1:0]     rx_src_chip,
    output wire [CORE_ID_BITS-1:0]     rx_core,
    output wire [NEURON_BITS-1:0]      rx_neuron,
    output wire signed [DATA_WIDTH-1:0] rx_current,
    input  wire                        rx_pop,
    output wire                        rx_empty,

    input  wire                        barrier_tx_send,
    output reg                         barrier_rx,

    input  wire                        mgmt_tx_push,
    input  wire [CORE_ID_BITS-1:0]     mgmt_tx_core,
    input  wire [NEURON_BITS-1:0]      mgmt_tx_neuron,
    input  wire [7:0]                  mgmt_tx_data,
    input  wire                        mgmt_tx_is_write,
    input  wire [CHIP_ID_BITS-1:0]     mgmt_tx_dest_chip,
    output reg                         mgmt_rx_valid,
    output reg  [CHIP_ID_BITS-1:0]     mgmt_rx_src_chip,
    output reg  [CORE_ID_BITS-1:0]     mgmt_rx_core,
    output reg  [NEURON_BITS-1:0]      mgmt_rx_neuron,
    output reg  [7:0]                  mgmt_rx_data,
    output reg                         mgmt_rx_is_write,

    input  wire                        preempt_request,
    output reg                         preempt_rx,

    output wire [NUM_LINKS*8-1:0]      link_tx_data,
    output wire [NUM_LINKS-1:0]        link_tx_valid,
    input  wire [NUM_LINKS-1:0]        link_tx_ready,
    input  wire [NUM_LINKS*8-1:0]      link_rx_data,
    input  wire [NUM_LINKS-1:0]        link_rx_valid,
    output wire [NUM_LINKS-1:0]        link_rx_ready
);

    localparam MSG_SPIKE   = 2'b00;
    localparam MSG_BARRIER = 2'b01;
    localparam MSG_MGMT    = 2'b10;
    localparam MSG_PREEMPT = 2'b11;

    localparam TX_FLAT_W    = 1 + 2 + 2*CHIP_ID_BITS + CORE_ID_BITS + NEURON_BITS + 8;
    localparam TX_NUM_BYTES = (TX_FLAT_W + 7) / 8;
    localparam TX_PAD_W     = TX_NUM_BYTES * 8;

    localparam MSGTYPE_OFFSET = TX_PAD_W - 1 - 1;
    localparam DEST_OFFSET = MSGTYPE_OFFSET - 2;
    localparam SRC_OFFSET  = DEST_OFFSET - CHIP_ID_BITS;
    localparam CORE_OFFSET = SRC_OFFSET - CHIP_ID_BITS;
    localparam NRN_OFFSET  = CORE_OFFSET - CORE_ID_BITS;
    localparam PAY_OFFSET  = NRN_OFFSET - NEURON_BITS;

    localparam PKT_W = 2 + CHIP_ID_BITS + CORE_ID_BITS + NEURON_BITS + 8;

    reg [PKT_W-1:0] tx_fifo [0:TX_DEPTH-1];
    reg [8:0] tx_wr_ptr, tx_rd_ptr;
    wire [8:0] tx_count = tx_wr_ptr - tx_rd_ptr;
    wire        tx_fifo_empty = (tx_wr_ptr == tx_rd_ptr);
    assign      tx_full = (tx_count >= TX_DEPTH);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            tx_wr_ptr <= 0;
        else if (tx_push && !tx_full) begin
            tx_fifo[tx_wr_ptr[7:0]] <= {MSG_SPIKE, tx_dest_chip, tx_core, tx_neuron, tx_payload};
            tx_wr_ptr <= tx_wr_ptr + 1;
        end else if (mgmt_tx_push && !tx_full) begin
            tx_fifo[tx_wr_ptr[7:0]] <= {MSG_MGMT, mgmt_tx_dest_chip, mgmt_tx_core, mgmt_tx_neuron,
                                         mgmt_tx_is_write, mgmt_tx_data[6:0]};
            tx_wr_ptr <= tx_wr_ptr + 1;
        end
    end

    wire [PKT_W-1:0] tx_head = tx_fifo[tx_rd_ptr[7:0]];
    wire [1:0] tx_head_msgtype = tx_head[PKT_W-1 -: 2];
    wire [CHIP_ID_BITS-1:0] tx_head_chip = tx_head[PKT_W-3 -: CHIP_ID_BITS];

    wire [CHIP_ID_BITS-1:0] tx_link_sel = tx_head_chip % NUM_LINKS;

    reg [TX_PAD_W-1:0] txs_shift;
    reg [$clog2(TX_NUM_BYTES+1)-1:0] txs_cnt;
    reg txs_active;
    reg [CHIP_ID_BITS-1:0] txs_link;

    reg [NUM_LINKS*8-1:0] ltx_data;
    reg [NUM_LINKS-1:0]   ltx_valid;
    assign link_tx_data  = ltx_data;
    assign link_tx_valid = ltx_valid;

    wire [TX_PAD_W-1:0] tx_flat = {1'b1, tx_head_msgtype, tx_head_chip, my_chip_id,
        tx_head[CORE_ID_BITS+NEURON_BITS+7 : 0],
        {(TX_PAD_W - TX_FLAT_W){1'b0}}};

    wire [TX_PAD_W-1:0] barrier_flat = {1'b1, MSG_BARRIER, {CHIP_ID_BITS{1'b1}}, my_chip_id,
        {(CORE_ID_BITS+NEURON_BITS+8){1'b0}},
        {(TX_PAD_W - TX_FLAT_W){1'b0}}};
    wire [TX_PAD_W-1:0] preempt_flat = {1'b1, MSG_PREEMPT, {CHIP_ID_BITS{1'b1}}, my_chip_id,
        {(CORE_ID_BITS+NEURON_BITS+8){1'b0}},
        {(TX_PAD_W - TX_FLAT_W){1'b0}}};

    reg                     bcast_active;
    reg [TX_PAD_W-1:0]      bcast_shift;
    reg [$clog2(TX_NUM_BYTES+1)-1:0] bcast_cnt;
    reg [CHIP_ID_BITS-1:0]  bcast_link;
    reg [CHIP_ID_BITS-1:0]  bcast_link_max;
    reg [1:0]               bcast_msg_type;
    reg                     bcast_pending;
    reg [TX_PAD_W-1:0]      bcast_flat_save;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            txs_active     <= 0;
            txs_cnt        <= 0;
            txs_shift      <= 0;
            txs_link       <= 0;
            tx_rd_ptr      <= 0;
            ltx_data       <= 0;
            ltx_valid      <= 0;
            bcast_active   <= 0;
            bcast_shift    <= 0;
            bcast_cnt      <= 0;
            bcast_link     <= 0;
            bcast_link_max <= 0;
            bcast_msg_type <= 0;
            bcast_pending  <= 0;
            bcast_flat_save <= 0;
        end else begin
            ltx_valid <= 0;

            if (bcast_active) begin
                ltx_data[bcast_link*8 +: 8] <= bcast_shift[TX_PAD_W-1 -: 8];
                ltx_valid[bcast_link] <= 1;

                if (link_tx_ready[bcast_link]) begin
                    bcast_shift <= bcast_shift << 8;
                    if (bcast_cnt == TX_NUM_BYTES - 1) begin
                        if (bcast_link < NUM_LINKS - 1) begin
                            bcast_link  <= bcast_link + 1;
                            bcast_shift <= bcast_flat_save;
                            bcast_cnt   <= 0;
                        end else begin
                            bcast_active <= 0;
                        end
                    end else begin
                        bcast_cnt <= bcast_cnt + 1;
                    end
                end
            end else if (!txs_active) begin
                if (barrier_tx_send) begin
                    bcast_active    <= 1;
                    bcast_flat_save <= barrier_flat;
                    bcast_shift     <= barrier_flat;
                    bcast_cnt       <= 0;
                    bcast_link      <= 0;
                    bcast_msg_type  <= MSG_BARRIER;
                end else if (preempt_request) begin
                    bcast_active    <= 1;
                    bcast_flat_save <= preempt_flat;
                    bcast_shift     <= preempt_flat;
                    bcast_cnt       <= 0;
                    bcast_link      <= 0;
                    bcast_msg_type  <= MSG_PREEMPT;
                end else if (!tx_fifo_empty) begin
                    ltx_data[tx_link_sel*8 +: 8] <= tx_flat[TX_PAD_W-1 -: 8];
                    ltx_valid[tx_link_sel] <= 1;
                    txs_shift  <= tx_flat << 8;
                    txs_link   <= tx_link_sel;
                    txs_cnt    <= 1;
                    txs_active <= 1;
                    tx_rd_ptr  <= tx_rd_ptr + 1;
                end
            end else begin
                ltx_data[txs_link*8 +: 8] <= txs_shift[TX_PAD_W-1 -: 8];
                ltx_valid[txs_link] <= 1;

                if (link_tx_ready[txs_link]) begin
                    txs_shift <= txs_shift << 8;
                    if (txs_cnt == TX_NUM_BYTES - 1)
                        txs_active <= 0;
                    else
                        txs_cnt <= txs_cnt + 1;
                end
            end
        end
    end

    localparam RX_PKT_W = CHIP_ID_BITS + CORE_ID_BITS + NEURON_BITS + DATA_WIDTH;

    reg [TX_PAD_W-1:0] rxs_accum [0:NUM_LINKS-1];
    reg [$clog2(TX_NUM_BYTES+1)-1:0] rxs_cnt [0:NUM_LINKS-1];
    reg [NUM_LINKS-1:0] rxs_push;

    assign link_rx_ready = (rx_count < RX_DEPTH - 4) ? {NUM_LINKS{1'b1}} : {NUM_LINKS{1'b0}};

    genvar li;
    generate
        for (li = 0; li < NUM_LINKS; li = li + 1) begin : gen_rx
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    rxs_cnt[li]  <= 0;
                    rxs_push[li] <= 0;
                    rxs_accum[li] <= 0;
                end else begin
                    rxs_push[li] <= 0;

                    if (link_rx_valid[li]) begin
                        rxs_accum[li] <= {rxs_accum[li][TX_PAD_W-9:0], link_rx_data[li*8 +: 8]};

                        if (rxs_cnt[li] == 0) begin
                            if (link_rx_data[li*8 + 7]) begin
                                rxs_accum[li] <= {{(TX_PAD_W-8){1'b0}}, link_rx_data[li*8 +: 8]};
                                rxs_cnt[li] <= 1;
                            end
                        end else begin
                            if (rxs_cnt[li] == TX_NUM_BYTES - 1) begin
                                rxs_push[li] <= 1;
                                rxs_cnt[li]  <= 0;
                            end else begin
                                rxs_cnt[li] <= rxs_cnt[li] + 1;
                            end
                        end
                    end
                end
            end
        end
    endgenerate

    reg [RX_PKT_W-1:0] rx_fifo [0:RX_DEPTH-1];
    reg [8:0] rx_wr_ptr, rx_rd_ptr;
    wire [8:0] rx_count = rx_wr_ptr - rx_rd_ptr;
    assign rx_empty = (rx_wr_ptr == rx_rd_ptr);

    always @(posedge clk or negedge rst_n) begin : rx_fifo_wr
        integer k;
        reg [1:0] rx_msg_type;
        if (!rst_n) begin
            rx_wr_ptr    <= 0;
            barrier_rx   <= 0;
            preempt_rx   <= 0;
            mgmt_rx_valid <= 0;
            mgmt_rx_src_chip <= 0;
            mgmt_rx_core     <= 0;
            mgmt_rx_neuron   <= 0;
            mgmt_rx_data     <= 0;
            mgmt_rx_is_write <= 0;
        end else begin
            barrier_rx    <= 0;
            preempt_rx    <= 0;
            mgmt_rx_valid <= 0;

            for (k = 0; k < NUM_LINKS; k = k + 1) begin
                if (rxs_push[k]) begin
                    rx_msg_type = rxs_accum[k][MSGTYPE_OFFSET -: 2];

                    case (rx_msg_type)
                        MSG_SPIKE: begin
                            if (rx_count < RX_DEPTH) begin
                                rx_fifo[rx_wr_ptr[7:0]] <= {
                                    rxs_accum[k][SRC_OFFSET -: CHIP_ID_BITS],
                                    rxs_accum[k][CORE_OFFSET -: CORE_ID_BITS],
                                    rxs_accum[k][NRN_OFFSET -: NEURON_BITS],
                                    {{(DATA_WIDTH-8){1'b0}},
                                     rxs_accum[k][PAY_OFFSET -: 8]}
                                };
                                rx_wr_ptr <= rx_wr_ptr + 1;
                            end
                        end

                        MSG_BARRIER: begin
                            barrier_rx <= 1;
                        end

                        MSG_MGMT: begin
                            mgmt_rx_valid    <= 1;
                            mgmt_rx_src_chip <= rxs_accum[k][SRC_OFFSET -: CHIP_ID_BITS];
                            mgmt_rx_core     <= rxs_accum[k][CORE_OFFSET -: CORE_ID_BITS];
                            mgmt_rx_neuron   <= rxs_accum[k][NRN_OFFSET -: NEURON_BITS];
                            mgmt_rx_is_write <= rxs_accum[k][PAY_OFFSET];
                            mgmt_rx_data     <= {1'b0, rxs_accum[k][PAY_OFFSET-1 -: 7]};
                        end

                        MSG_PREEMPT: begin
                            preempt_rx <= 1;
                        end
                    endcase
                end
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            rx_rd_ptr <= 0;
        else if (rx_pop && !rx_empty)
            rx_rd_ptr <= rx_rd_ptr + 1;
    end

    wire [RX_PKT_W-1:0] rx_top = rx_fifo[rx_rd_ptr[7:0]];
    assign rx_src_chip = rx_top[RX_PKT_W-1 -: CHIP_ID_BITS];
    assign rx_core     = rx_top[NEURON_BITS+DATA_WIDTH +: CORE_ID_BITS];
    assign rx_neuron   = rx_top[DATA_WIDTH +: NEURON_BITS];
    assign rx_current  = rx_top[DATA_WIDTH-1:0];

endmodule
