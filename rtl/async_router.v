`timescale 1ns/1ps

module async_router #(
    parameter PACKET_W      = 34,
    parameter COORD_BITS    = 4,
    parameter FIFO_DEPTH    = 16,
    parameter FIFO_PTR_BITS = 4
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire [COORD_BITS-1:0]   my_x,
    input  wire [COORD_BITS-1:0]   my_y,

    input  wire                    local_in_valid,
    output wire                    local_in_ready,
    input  wire [PACKET_W-1:0]     local_in_data,
    output wire                    local_out_valid,
    input  wire                    local_out_ready,
    output wire [PACKET_W-1:0]     local_out_data,

    input  wire                    north_in_valid,
    output wire                    north_in_ready,
    input  wire [PACKET_W-1:0]     north_in_data,
    output wire                    north_out_valid,
    input  wire                    north_out_ready,
    output wire [PACKET_W-1:0]     north_out_data,

    input  wire                    south_in_valid,
    output wire                    south_in_ready,
    input  wire [PACKET_W-1:0]     south_in_data,
    output wire                    south_out_valid,
    input  wire                    south_out_ready,
    output wire [PACKET_W-1:0]     south_out_data,

    input  wire                    east_in_valid,
    output wire                    east_in_ready,
    input  wire [PACKET_W-1:0]     east_in_data,
    output wire                    east_out_valid,
    input  wire                    east_out_ready,
    output wire [PACKET_W-1:0]     east_out_data,

    input  wire                    west_in_valid,
    output wire                    west_in_ready,
    input  wire [PACKET_W-1:0]     west_in_data,
    output wire                    west_out_valid,
    input  wire                    west_out_ready,
    output wire [PACKET_W-1:0]     west_out_data,

    output wire                    idle
);

    localparam P_LOCAL = 0, P_NORTH = 1, P_SOUTH = 2, P_EAST = 3, P_WEST = 4;

    localparam DX_MSB = PACKET_W - 1;
    localparam DX_LSB = PACKET_W - COORD_BITS;
    localparam DY_MSB = DX_LSB - 1;
    localparam DY_LSB = DX_LSB - COORD_BITS;

    wire [4:0] fifo_empty, fifo_full;
    wire [PACKET_W-1:0] fifo_head [0:4];
    wire [4:0] fifo_push;
    reg  [4:0] fifo_pop;

    assign fifo_push[P_LOCAL] = local_in_valid && !fifo_full[P_LOCAL];
    assign fifo_push[P_NORTH] = north_in_valid && !fifo_full[P_NORTH];
    assign fifo_push[P_SOUTH] = south_in_valid && !fifo_full[P_SOUTH];
    assign fifo_push[P_EAST]  = east_in_valid  && !fifo_full[P_EAST];
    assign fifo_push[P_WEST]  = west_in_valid  && !fifo_full[P_WEST];

    assign local_in_ready = !fifo_full[P_LOCAL];
    assign north_in_ready = !fifo_full[P_NORTH];
    assign south_in_ready = !fifo_full[P_SOUTH];
    assign east_in_ready  = !fifo_full[P_EAST];
    assign west_in_ready  = !fifo_full[P_WEST];

    wire [PACKET_W-1:0] in_data [0:4];
    assign in_data[P_LOCAL] = local_in_data;
    assign in_data[P_NORTH] = north_in_data;
    assign in_data[P_SOUTH] = south_in_data;
    assign in_data[P_EAST]  = east_in_data;
    assign in_data[P_WEST]  = west_in_data;

    genvar gi;
    generate
        for (gi = 0; gi < 5; gi = gi + 1) begin : gen_fifo
            spike_fifo #(
                .ID_WIDTH  (PACKET_W),
                .DEPTH     (FIFO_DEPTH),
                .PTR_BITS  (FIFO_PTR_BITS)
            ) input_fifo (
                .clk       (clk),
                .rst_n     (rst_n),
                .push      (fifo_push[gi]),
                .pop       (fifo_pop[gi]),
                .clear     (1'b0),
                .push_data (in_data[gi]),
                .pop_data  (fifo_head[gi]),
                .empty     (fifo_empty[gi]),
                .full      (fifo_full[gi])
            );
        end
    endgenerate

    function [2:0] xy_route;
        input [COORD_BITS-1:0] dx, dy, cx, cy;
        begin
            if      (dx > cx) xy_route = P_EAST;
            else if (dx < cx) xy_route = P_WEST;
            else if (dy > cy) xy_route = P_NORTH;
            else if (dy < cy) xy_route = P_SOUTH;
            else              xy_route = P_LOCAL;
        end
    endfunction

    wire [2:0] head_route [0:4];
    generate
        for (gi = 0; gi < 5; gi = gi + 1) begin : gen_route
            assign head_route[gi] = xy_route(
                fifo_head[gi][DX_MSB:DX_LSB],
                fifo_head[gi][DY_MSB:DY_LSB],
                my_x, my_y
            );
        end
    endgenerate

    reg  [4:0] out_valid_r;
    reg  [PACKET_W-1:0] out_data_r [0:4];

    wire [4:0] out_ready;
    assign out_ready[P_LOCAL] = local_out_ready;
    assign out_ready[P_NORTH] = north_out_ready;
    assign out_ready[P_SOUTH] = south_out_ready;
    assign out_ready[P_EAST]  = east_out_ready;
    assign out_ready[P_WEST]  = west_out_ready;

    assign local_out_valid = out_valid_r[P_LOCAL];
    assign local_out_data  = out_data_r[P_LOCAL];
    assign north_out_valid = out_valid_r[P_NORTH];
    assign north_out_data  = out_data_r[P_NORTH];
    assign south_out_valid = out_valid_r[P_SOUTH];
    assign south_out_data  = out_data_r[P_SOUTH];
    assign east_out_valid  = out_valid_r[P_EAST];
    assign east_out_data   = out_data_r[P_EAST];
    assign west_out_valid  = out_valid_r[P_WEST];
    assign west_out_data   = out_data_r[P_WEST];

    reg [2:0] arb_ptr;

    reg [4:0] comb_grant;
    reg [4:0] comb_out_claim;

    always @(*) begin : grant_logic
        integer p, idx;
        comb_grant = 5'b0;
        comb_out_claim = 5'b0;
        for (p = 0; p < 5; p = p + 1) begin
            idx = arb_ptr + p;
            if (idx >= 5) idx = idx - 5;
            if (!fifo_empty[idx] && !comb_grant[idx]) begin
                if (!out_valid_r[head_route[idx]] && !comb_out_claim[head_route[idx]]) begin
                    comb_grant[idx] = 1'b1;
                    comb_out_claim[head_route[idx]] = 1'b1;
                end
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin : seq_logic
        integer i;
        if (!rst_n) begin
            out_valid_r <= 5'b0;
            arb_ptr <= 3'd0;
            for (i = 0; i < 5; i = i + 1)
                out_data_r[i] <= {PACKET_W{1'b0}};
        end else begin
            for (i = 0; i < 5; i = i + 1)
                if (out_valid_r[i] && out_ready[i])
                    out_valid_r[i] <= 1'b0;

            for (i = 0; i < 5; i = i + 1) begin
                if (comb_grant[i]) begin
                    out_valid_r[head_route[i]] <= 1'b1;
                    out_data_r[head_route[i]] <= fifo_head[i];
                end
            end

            arb_ptr <= (arb_ptr == 3'd4) ? 3'd0 : arb_ptr + 3'd1;
        end
    end

    always @(*) fifo_pop = comb_grant;

    assign idle = (&fifo_empty) &&
                  !out_valid_r[P_NORTH] && !out_valid_r[P_SOUTH] &&
                  !out_valid_r[P_EAST]  && !out_valid_r[P_WEST];

endmodule
