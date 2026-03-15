// ============================================================================
// Neuron Core with STDP Learning
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

module neuron_core_stdp #(
    parameter NUM_NEURONS  = 4,
    parameter DATA_WIDTH   = 16,
    parameter THRESHOLD    = 16'd1000,
    parameter LEAK_RATE    = 16'd2,
    parameter WEIGHT_INIT  = 16'd100,
    parameter WEIGHT_MAX   = 16'd800,
    parameter LEARN_RATE   = 8'd3
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    enable,
    input  wire                    learn_enable,

    input  wire signed [DATA_WIDTH-1:0] ext_input_0,
    input  wire signed [DATA_WIDTH-1:0] ext_input_1,
    input  wire signed [DATA_WIDTH-1:0] ext_input_2,
    input  wire signed [DATA_WIDTH-1:0] ext_input_3,

    output wire [NUM_NEURONS-1:0] spikes,

    output wire [DATA_WIDTH-1:0] membrane_0,
    output wire [DATA_WIDTH-1:0] membrane_1,
    output wire [DATA_WIDTH-1:0] membrane_2,
    output wire [DATA_WIDTH-1:0] membrane_3,

    output wire signed [DATA_WIDTH-1:0] w_out_01, w_out_02, w_out_03,
    output wire signed [DATA_WIDTH-1:0] w_out_10, w_out_12, w_out_13,
    output wire signed [DATA_WIDTH-1:0] w_out_20, w_out_21, w_out_23,
    output wire signed [DATA_WIDTH-1:0] w_out_30, w_out_31, w_out_32
);

    wire signed [DATA_WIDTH-1:0] syn_current [0:3][0:3];
    wire signed [DATA_WIDTH-1:0] syn_weight  [0:3][0:3];
    wire signed [DATA_WIDTH-1:0] total_input [0:3];

    wire signed [DATA_WIDTH-1:0] ext_inputs [0:3];
    assign ext_inputs[0] = ext_input_0;
    assign ext_inputs[1] = ext_input_1;
    assign ext_inputs[2] = ext_input_2;
    assign ext_inputs[3] = ext_input_3;

    genvar src, dst;
    generate
        for (src = 0; src < NUM_NEURONS; src = src + 1) begin : syn_src
            for (dst = 0; dst < NUM_NEURONS; dst = dst + 1) begin : syn_dst
                if (src != dst) begin : real_syn
                    stdp_synapse #(
                        .DATA_WIDTH  (DATA_WIDTH),
                        .WEIGHT_INIT (WEIGHT_INIT),
                        .WEIGHT_MAX  (WEIGHT_MAX),
                        .LEARN_RATE  (LEARN_RATE)
                    ) syn_inst (
                        .clk           (clk),
                        .rst_n         (rst_n),
                        .learn_enable  (learn_enable),
                        .pre_spike     (spikes[src]),
                        .post_spike    (spikes[dst]),
                        .weight        (syn_weight[src][dst]),
                        .post_current  (syn_current[src][dst]),
                        .pre_trace_out (),
                        .post_trace_out()
                    );
                end else begin : no_self
                    assign syn_current[src][dst] = 0;
                    assign syn_weight[src][dst]  = 0;
                end
            end
        end
    endgenerate

    assign total_input[0] = ext_inputs[0] + syn_current[0][0] + syn_current[1][0] + syn_current[2][0] + syn_current[3][0];
    assign total_input[1] = ext_inputs[1] + syn_current[0][1] + syn_current[1][1] + syn_current[2][1] + syn_current[3][1];
    assign total_input[2] = ext_inputs[2] + syn_current[0][2] + syn_current[1][2] + syn_current[2][2] + syn_current[3][2];
    assign total_input[3] = ext_inputs[3] + syn_current[0][3] + syn_current[1][3] + syn_current[2][3] + syn_current[3][3];

    generate
        for (dst = 0; dst < NUM_NEURONS; dst = dst + 1) begin : neurons
            lif_neuron #(
                .DATA_WIDTH (DATA_WIDTH),
                .THRESHOLD  (THRESHOLD),
                .LEAK_RATE  (LEAK_RATE)
            ) neuron_inst (
                .clk            (clk),
                .rst_n          (rst_n),
                .enable         (enable),
                .synaptic_input (total_input[dst]),
                .spike          (spikes[dst]),
                .membrane_pot   ()
            );
        end
    endgenerate

    assign membrane_0 = neurons[0].neuron_inst.membrane_pot;
    assign membrane_1 = neurons[1].neuron_inst.membrane_pot;
    assign membrane_2 = neurons[2].neuron_inst.membrane_pot;
    assign membrane_3 = neurons[3].neuron_inst.membrane_pot;

    assign w_out_01 = syn_weight[0][1];
    assign w_out_02 = syn_weight[0][2];
    assign w_out_03 = syn_weight[0][3];
    assign w_out_10 = syn_weight[1][0];
    assign w_out_12 = syn_weight[1][2];
    assign w_out_13 = syn_weight[1][3];
    assign w_out_20 = syn_weight[2][0];
    assign w_out_21 = syn_weight[2][1];
    assign w_out_23 = syn_weight[2][3];
    assign w_out_30 = syn_weight[3][0];
    assign w_out_31 = syn_weight[3][1];
    assign w_out_32 = syn_weight[3][2];

endmodule
