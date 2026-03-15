// ============================================================================
// SRAM
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
