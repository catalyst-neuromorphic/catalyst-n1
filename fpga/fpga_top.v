module fpga_top #(
    parameter CLK_FREQ  = 100_000_000,
    parameter BAUD      = 115200,
    parameter POR_BITS  = 20
)(
    input  wire       clk,
    input  wire       btn_rst,
    input  wire       uart_rxd,
    output wire       uart_txd,
    output reg  [3:0] led
);

    reg [POR_BITS-1:0] debounce_cnt;
    reg        btn_sync1, btn_sync2;
    reg        btn_stable;
    wire       rst_n;

    always @(posedge clk) begin
        btn_sync1 <= btn_rst;
        btn_sync2 <= btn_sync1;
    end

    always @(posedge clk) begin
        if (btn_sync2 != btn_stable) begin
            debounce_cnt <= debounce_cnt + 1;
            if (debounce_cnt == {POR_BITS{1'b1}}) begin
                btn_stable   <= btn_sync2;
                debounce_cnt <= 0;
            end
        end else begin
            debounce_cnt <= 0;
        end
    end

    reg [POR_BITS-1:0] por_cnt;
    reg                por_done;

    always @(posedge clk) begin
        if (!por_done) begin
            por_cnt <= por_cnt + 1;
            if (por_cnt == {POR_BITS{1'b1}})
                por_done <= 1;
        end
    end

    initial begin
        por_cnt    = 0;
        por_done   = 0;
        btn_stable = 0;
        debounce_cnt = 0;
    end

    assign rst_n = por_done & ~btn_stable;

    neuromorphic_top #(
        .CLK_FREQ       (CLK_FREQ),
        .BAUD           (BAUD),
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
        .uart_rxd       (uart_rxd),
        .uart_txd       (uart_txd),
        .link_tx_data   (),
        .link_tx_valid  (),
        .link_tx_ready  (1'b0),
        .link_rx_data   (8'd0),
        .link_rx_valid  (1'b0),
        .link_rx_ready  (),
        .rx_data_ext    (8'd0),
        .rx_valid_ext   (1'b0),
        .tx_data_ext    (),
        .tx_valid_ext   (),
        .tx_ready_ext   (1'b0)
    );

    reg [25:0] heartbeat_cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            heartbeat_cnt <= 0;
        else
            heartbeat_cnt <= heartbeat_cnt + 1;
    end

    reg [22:0] rx_blink_cnt;
    wire       rx_activity;
    reg        rxd_prev;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rxd_prev     <= 1;
            rx_blink_cnt <= 0;
        end else begin
            rxd_prev <= uart_rxd;
            if (rxd_prev && !uart_rxd)
                rx_blink_cnt <= {23{1'b1}};
            else if (rx_blink_cnt != 0)
                rx_blink_cnt <= rx_blink_cnt - 1;
        end
    end
    assign rx_activity = (rx_blink_cnt != 0);

    reg        txd_prev;
    reg [22:0] tx_blink_cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            txd_prev     <= 1;
            tx_blink_cnt <= 0;
        end else begin
            txd_prev <= uart_txd;
            if (txd_prev && !uart_txd)
                tx_blink_cnt <= {23{1'b1}};
            else if (tx_blink_cnt != 0)
                tx_blink_cnt <= tx_blink_cnt - 1;
        end
    end

    reg [22:0] activity_cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            activity_cnt <= 0;
        else if (rx_activity || tx_blink_cnt != 0)
            activity_cnt <= {23{1'b1}};
        else if (activity_cnt != 0)
            activity_cnt <= activity_cnt - 1;
    end

    always @(*) begin
        led[0] = heartbeat_cnt[25];
        led[1] = rx_activity;
        led[2] = (tx_blink_cnt != 0);
        led[3] = (activity_cnt != 0);
    end

endmodule
