module async_fifo #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_BITS  = 4
)(
    input  wire                  wr_clk,
    input  wire                  wr_rst_n,
    input  wire [DATA_WIDTH-1:0] wr_data,
    input  wire                  wr_en,
    output wire                  wr_full,

    input  wire                  rd_clk,
    input  wire                  rd_rst_n,
    input  wire                  rd_en,
    output wire [DATA_WIDTH-1:0] rd_data,
    output wire                  rd_empty
);

    localparam DEPTH = 1 << ADDR_BITS;

    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    reg [ADDR_BITS:0] wr_bin, wr_gray;
    wire [ADDR_BITS:0] wr_bin_next  = wr_bin + 1;
    wire [ADDR_BITS:0] wr_gray_next = wr_bin_next ^ (wr_bin_next >> 1);

    reg [ADDR_BITS:0] rd_bin, rd_gray;
    wire [ADDR_BITS:0] rd_bin_next  = rd_bin + 1;
    wire [ADDR_BITS:0] rd_gray_next = rd_bin_next ^ (rd_bin_next >> 1);

    reg [ADDR_BITS:0] wr_gray_rd_s1, wr_gray_rd_s2;
    reg [ADDR_BITS:0] rd_gray_wr_s1, rd_gray_wr_s2;

    always @(posedge wr_clk or negedge wr_rst_n)
        if (!wr_rst_n) begin
            wr_bin  <= 0;
            wr_gray <= 0;
        end else if (wr_en && !wr_full) begin
            mem[wr_bin[ADDR_BITS-1:0]] <= wr_data;
            wr_bin  <= wr_bin_next;
            wr_gray <= wr_gray_next;
        end

    always @(posedge rd_clk or negedge rd_rst_n)
        if (!rd_rst_n) begin
            rd_bin  <= 0;
            rd_gray <= 0;
        end else if (rd_en && !rd_empty) begin
            rd_bin  <= rd_bin_next;
            rd_gray <= rd_gray_next;
        end

    always @(posedge rd_clk or negedge rd_rst_n)
        if (!rd_rst_n) begin
            wr_gray_rd_s1 <= 0;
            wr_gray_rd_s2 <= 0;
        end else begin
            wr_gray_rd_s1 <= wr_gray;
            wr_gray_rd_s2 <= wr_gray_rd_s1;
        end

    always @(posedge wr_clk or negedge wr_rst_n)
        if (!wr_rst_n) begin
            rd_gray_wr_s1 <= 0;
            rd_gray_wr_s2 <= 0;
        end else begin
            rd_gray_wr_s1 <= rd_gray;
            rd_gray_wr_s2 <= rd_gray_wr_s1;
        end

    assign wr_full  = (wr_gray == {~rd_gray_wr_s2[ADDR_BITS:ADDR_BITS-1],
                                     rd_gray_wr_s2[ADDR_BITS-2:0]});

    assign rd_empty = (rd_gray == wr_gray_rd_s2);

    assign rd_data  = mem[rd_bin[ADDR_BITS-1:0]];

endmodule
