module uart_rx #(
    parameter CLK_FREQ = 100_000_000,
    parameter BAUD     = 115200
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       rx,
    output reg  [7:0] data,
    output reg        valid
);

    localparam CLKS_PER_BIT = CLK_FREQ / BAUD;
    localparam HALF_BIT     = CLKS_PER_BIT / 2;

    localparam S_IDLE  = 2'd0;
    localparam S_START = 2'd1;
    localparam S_DATA  = 2'd2;
    localparam S_STOP  = 2'd3;

    reg [1:0]  state;
    reg [15:0] clk_cnt;
    reg [2:0]  bit_idx;
    reg [7:0]  shift;
    reg        rx_s1, rx_s2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_s1 <= 1;
            rx_s2 <= 1;
        end else begin
            rx_s1 <= rx;
            rx_s2 <= rx_s1;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= S_IDLE;
            valid   <= 0;
            clk_cnt <= 0;
            bit_idx <= 0;
            shift   <= 0;
            data    <= 0;
        end else begin
            valid <= 0;
            case (state)
                S_IDLE: begin
                    if (!rx_s2) begin
                        clk_cnt <= 0;
                        state   <= S_START;
                    end
                end
                S_START: begin
                    if (clk_cnt == HALF_BIT - 1) begin
                        if (!rx_s2) begin
                            clk_cnt <= 0;
                            bit_idx <= 0;
                            state   <= S_DATA;
                        end else
                            state <= S_IDLE;
                    end else
                        clk_cnt <= clk_cnt + 1;
                end
                S_DATA: begin
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt <= 0;
                        shift   <= {rx_s2, shift[7:1]};
                        if (bit_idx == 7)
                            state <= S_STOP;
                        else
                            bit_idx <= bit_idx + 1;
                    end else
                        clk_cnt <= clk_cnt + 1;
                end
                S_STOP: begin
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        data  <= shift;
                        valid <= 1;
                        state <= S_IDLE;
                    end else
                        clk_cnt <= clk_cnt + 1;
                end
            endcase
        end
    end

endmodule
