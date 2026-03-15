// ============================================================================
// Leaky Integrate-and-Fire (LIF) Neuron
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

module lif_neuron #(
    parameter DATA_WIDTH    = 16,
    parameter THRESHOLD     = 16'd1000,
    parameter LEAK_RATE     = 16'd2,
    parameter RESTING_POT   = 16'd0,
    parameter REFRAC_CYCLES = 4
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    enable,
    input  wire signed [DATA_WIDTH-1:0] synaptic_input,
    output reg                     spike,
    output reg  [DATA_WIDTH-1:0]   membrane_pot
);

    reg [DATA_WIDTH-1:0] potential;
    reg [3:0]            refrac_counter;

    wire in_refractory = (refrac_counter > 0);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            potential      <= RESTING_POT;
            spike          <= 1'b0;
            refrac_counter <= 4'd0;
            membrane_pot   <= RESTING_POT;

        end else if (enable) begin
            spike <= 1'b0;

            if (in_refractory) begin
                refrac_counter <= refrac_counter - 1;
                potential      <= RESTING_POT;

            end else begin
                if (potential + synaptic_input > THRESHOLD) begin
                    spike          <= 1'b1;
                    potential      <= RESTING_POT;
                    refrac_counter <= REFRAC_CYCLES[3:0];
                end else if (potential + synaptic_input < RESTING_POT + LEAK_RATE) begin
                    potential <= RESTING_POT;
                end else begin
                    potential <= potential + synaptic_input - LEAK_RATE;
                end
            end

            membrane_pot <= potential;
        end
    end

endmodule
