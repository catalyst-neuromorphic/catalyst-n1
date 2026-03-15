// ============================================================================
// Neuron Core
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

module neuron_core #(
    parameter NUM_NEURONS = 4,
    parameter DATA_WIDTH  = 16,
    parameter THRESHOLD   = 16'd1000,
    parameter LEAK_RATE   = 16'd2
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    enable,

    input  wire signed [DATA_WIDTH-1:0] ext_input_0,
    input  wire signed [DATA_WIDTH-1:0] ext_input_1,
    input  wire signed [DATA_WIDTH-1:0] ext_input_2,
    input  wire signed [DATA_WIDTH-1:0] ext_input_3,

    input  wire signed [DATA_WIDTH-1:0] weight_00, weight_01, weight_02, weight_03,
    input  wire signed [DATA_WIDTH-1:0] weight_10, weight_11, weight_12, weight_13,
    input  wire signed [DATA_WIDTH-1:0] weight_20, weight_21, weight_22, weight_23,
    input  wire signed [DATA_WIDTH-1:0] weight_30, weight_31, weight_32, weight_33,

    output wire [NUM_NEURONS-1:0] spikes,

    output wire [DATA_WIDTH-1:0] membrane_0,
    output wire [DATA_WIDTH-1:0] membrane_1,
    output wire [DATA_WIDTH-1:0] membrane_2,
    output wire [DATA_WIDTH-1:0] membrane_3
);

    wire signed [DATA_WIDTH-1:0] syn_current [0:3][0:3];
    wire signed [DATA_WIDTH-1:0] total_input [0:3];
    wire signed [DATA_WIDTH-1:0] weights [0:3][0:3];

    assign weights[0][0] = weight_00; assign weights[0][1] = weight_01;
    assign weights[0][2] = weight_02; assign weights[0][3] = weight_03;
    assign weights[1][0] = weight_10; assign weights[1][1] = weight_11;
    assign weights[1][2] = weight_12; assign weights[1][3] = weight_13;
    assign weights[2][0] = weight_20; assign weights[2][1] = weight_21;
    assign weights[2][2] = weight_22; assign weights[2][3] = weight_23;
    assign weights[3][0] = weight_30; assign weights[3][1] = weight_31;
    assign weights[3][2] = weight_32; assign weights[3][3] = weight_33;

    wire signed [DATA_WIDTH-1:0] ext_inputs [0:3];
    assign ext_inputs[0] = ext_input_0;
    assign ext_inputs[1] = ext_input_1;
    assign ext_inputs[2] = ext_input_2;
    assign ext_inputs[3] = ext_input_3;

    genvar src, dst;
    generate
        for (src = 0; src < NUM_NEURONS; src = src + 1) begin : syn_src
            for (dst = 0; dst < NUM_NEURONS; dst = dst + 1) begin : syn_dst
                synapse #(
                    .DATA_WIDTH(DATA_WIDTH)
                ) syn_inst (
                    .clk         (clk),
                    .rst_n       (rst_n),
                    .pre_spike   (spikes[src]),
                    .weight      (weights[src][dst]),
                    .post_current(syn_current[src][dst])
                );
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

endmodule
