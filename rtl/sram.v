module sram #(
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 6,
    parameter DEPTH      = (1 << ADDR_WIDTH),
    parameter [DATA_WIDTH-1:0] INIT_VALUE = {DATA_WIDTH{1'b0}}
)(
    input  wire                    clk,

    input  wire                    we_a,
    input  wire [ADDR_WIDTH-1:0]   addr_a,
    input  wire [DATA_WIDTH-1:0]   wdata_a,
    output reg  [DATA_WIDTH-1:0]   rdata_a,

    input  wire [ADDR_WIDTH-1:0]   addr_b,
    output reg  [DATA_WIDTH-1:0]   rdata_b
);

    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (we_a)
            mem[addr_a] <= wdata_a;
        rdata_a <= mem[addr_a];
    end

    always @(posedge clk) begin
        rdata_b <= mem[addr_b];
    end

    integer i;
    initial begin
        for (i = 0; i < DEPTH; i = i + 1)
            mem[i] = INIT_VALUE;
    end

endmodule
