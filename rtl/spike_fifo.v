// ============================================================================
// Spike FIFO
// ============================================================================
//
// Copyright 2026 Henry Arthur Shulayev Barnes / Catalyst Neuromorphic Ltd
// Company No. 17054540 — UK Patent Application No. 2602902.6
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================

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
