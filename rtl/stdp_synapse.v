// ============================================================================
// STDP Synapse
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

module stdp_synapse #(
    parameter DATA_WIDTH   = 16,
    parameter TRACE_WIDTH  = 8,
    parameter TRACE_MAX    = 8'd127,
    parameter TRACE_DECAY  = 8'd4,
    parameter LEARN_RATE   = 8'd4,
    parameter WEIGHT_MAX   = 16'd800,
    parameter WEIGHT_MIN   = -16'sd800,
    parameter WEIGHT_INIT  = 16'd0
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          learn_enable,
    input  wire                          pre_spike,
    input  wire                          post_spike,
    output reg  signed [DATA_WIDTH-1:0]  weight,
    output reg  signed [DATA_WIDTH-1:0]  post_current,
    output wire [TRACE_WIDTH-1:0]        pre_trace_out,
    output wire [TRACE_WIDTH-1:0]        post_trace_out
);

    reg [TRACE_WIDTH-1:0] pre_trace;
    reg [TRACE_WIDTH-1:0] post_trace;

    assign pre_trace_out  = pre_trace;
    assign post_trace_out = post_trace;

    wire signed [DATA_WIDTH-1:0] ltp_delta;
    wire signed [DATA_WIDTH-1:0] ltd_delta;

    assign ltp_delta = {{(DATA_WIDTH-TRACE_WIDTH){1'b0}}, pre_trace} >>> LEARN_RATE;
    assign ltd_delta = {{(DATA_WIDTH-TRACE_WIDTH){1'b0}}, post_trace} >>> LEARN_RATE;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pre_trace    <= 0;
            post_trace   <= 0;
            weight       <= WEIGHT_INIT;
            post_current <= 0;

        end else begin
            if (pre_spike) begin
                pre_trace <= TRACE_MAX;
            end else if (pre_trace > TRACE_DECAY) begin
                pre_trace <= pre_trace - TRACE_DECAY;
            end else begin
                pre_trace <= 0;
            end

            if (post_spike) begin
                post_trace <= TRACE_MAX;
            end else if (post_trace > TRACE_DECAY) begin
                post_trace <= post_trace - TRACE_DECAY;
            end else begin
                post_trace <= 0;
            end

            if (learn_enable) begin
                if (post_spike && pre_trace > 0) begin
                    if (weight + ltp_delta > WEIGHT_MAX)
                        weight <= WEIGHT_MAX;
                    else
                        weight <= weight + ltp_delta;
                end

                if (pre_spike && post_trace > 0) begin
                    if (weight - ltd_delta < WEIGHT_MIN)
                        weight <= WEIGHT_MIN;
                    else
                        weight <= weight - ltd_delta;
                end
            end

            if (pre_spike) begin
                post_current <= weight;
            end else begin
                post_current <= 0;
            end
        end
    end

endmodule
