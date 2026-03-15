module uart_tx #(
    parameter CLK_FREQ = 100_000_000,
    parameter BAUD     = 115200
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire [7:0] data,
    input  wire       valid,
    output reg        tx,
    output wire       ready
);

    localparam CLKS_PER_BIT = CLK_FREQ / BAUD;

    localparam S_IDLE  = 2'd0;
    localparam S_START = 2'd1;
    localparam S_DATA  = 2'd2;
    localparam S_STOP  = 2'd3;

    reg [1:0]  state;
    reg [15:0] clk_cnt;
    reg [2:0]  bit_idx;
    reg [7:0]  shift;

    assign ready = (state == S_IDLE);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= S_IDLE;
            tx      <= 1;
            clk_cnt <= 0;
            bit_idx <= 0;
            shift   <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    tx <= 1;
                    if (valid) begin
                        shift   <= data;
                        state   <= S_START;
                        clk_cnt <= 0;
                    end
                end
                S_START: begin
                    tx <= 0;
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt <= 0;
                        bit_idx <= 0;
                        state   <= S_DATA;
                    end else
                        clk_cnt <= clk_cnt + 1;
                end
                S_DATA: begin
                    tx <= shift[0];
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt <= 0;
                        shift   <= {1'b0, shift[7:1]};
                        if (bit_idx == 7)
                            state <= S_STOP;
                        else
                            bit_idx <= bit_idx + 1;
                    end else
                        clk_cnt <= clk_cnt + 1;
                end
                S_STOP: begin
                    tx <= 1;
                    if (clk_cnt == CLKS_PER_BIT - 1)
                        state <= S_IDLE;
                    else
                        clk_cnt <= clk_cnt + 1;
                end
            endcase
        end
    end

endmodule
