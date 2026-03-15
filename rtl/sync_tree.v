// ============================================================================
// Sync Tree
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

`timescale 1ns/1ps

module sync_tree #(
    parameter NUM_LEAVES = 4
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire [NUM_LEAVES-1:0] leaf_done,
    output wire                  all_done,
    input  wire                  root_start,
    output wire [NUM_LEAVES-1:0] leaf_start
);

    assign all_done = &leaf_done;

    assign leaf_start = {NUM_LEAVES{root_start}};

endmodule
