module spike_fifo #(
    parameter ID_WIDTH = 8,
    parameter DEPTH    = 64,
    parameter PTR_BITS = 6
)(
    input  wire                clk,
    input  wire                rst_n,
    input  wire                clear,

    input  wire                push,
    input  wire [ID_WIDTH-1:0] push_data,

    input  wire                pop,
    output wire [ID_WIDTH-1:0] pop_data,

    output wire                empty,
    output wire                full,
    output wire [PTR_BITS:0]   count
);

    reg [ID_WIDTH-1:0] mem [0:DEPTH-1];

    reg [PTR_BITS:0] wr_ptr;
    reg [PTR_BITS:0] rd_ptr;

    assign count = wr_ptr - rd_ptr;
    assign empty = (wr_ptr == rd_ptr);
    assign full  = (count == DEPTH);

    assign pop_data = mem[rd_ptr[PTR_BITS-1:0]];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 0;
            rd_ptr <= 0;
        end else if (clear) begin
            wr_ptr <= 0;
            rd_ptr <= 0;
        end else begin
            if (push && !full) begin
                mem[wr_ptr[PTR_BITS-1:0]] <= push_data;
                wr_ptr <= wr_ptr + 1;
            end
            if (pop && !empty) begin
                rd_ptr <= rd_ptr + 1;
            end
        end
    end

endmodule
