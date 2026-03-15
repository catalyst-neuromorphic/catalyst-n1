// ============================================================================
// Scalable Neuron Core
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

module scalable_core #(
    parameter NUM_NEURONS   = 64,
    parameter DATA_WIDTH    = 16,
    parameter NEURON_BITS   = 6,
    parameter WEIGHT_BITS   = 12,
    parameter THRESHOLD     = 16'sd1000,
    parameter LEAK_RATE     = 16'sd3,
    parameter RESTING_POT   = 16'sd0,
    parameter REFRAC_CYCLES = 4,
    parameter TRACE_MAX     = 8'd100,
    parameter TRACE_DECAY   = 8'd3,
    parameter LEARN_SHIFT   = 3
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    start,
    input  wire                    learn_enable,

    input  wire                    ext_valid,
    input  wire [NEURON_BITS-1:0]  ext_neuron_id,
    input  wire signed [DATA_WIDTH-1:0] ext_current,

    input  wire                    inject_spike_valid,
    input  wire [NEURON_BITS-1:0]  inject_spike_id,

    input  wire                    weight_we,
    input  wire [WEIGHT_BITS-1:0]  weight_addr,
    input  wire signed [DATA_WIDTH-1:0] weight_data,

    output reg                     timestep_done,
    output reg                     spike_out_valid,
    output reg  [NEURON_BITS-1:0]  spike_out_id,

    output wire [3:0]              state_out,
    output reg  [15:0]             total_spikes,
    output reg  [15:0]             timestep_count
);

    localparam S_IDLE         = 4'd0;
    localparam S_DELIVER_INIT = 4'd1;
    localparam S_DELIVER_READ = 4'd2;
    localparam S_DELIVER_ACC  = 4'd3;
    localparam S_DELIVER_NEXT = 4'd4;
    localparam S_UPDATE_INIT  = 4'd5;
    localparam S_UPDATE_READ  = 4'd6;
    localparam S_UPDATE_CALC  = 4'd7;
    localparam S_UPDATE_WRITE = 4'd8;
    localparam S_LEARN        = 4'd9;
    localparam S_LEARN_WRITE  = 4'd10;
    localparam S_DONE         = 4'd11;

    reg [3:0] state;
    assign state_out = state;

    reg                    mem_we;
    reg  [NEURON_BITS-1:0] mem_addr;
    reg  signed [DATA_WIDTH-1:0] mem_wdata;
    wire signed [DATA_WIDTH-1:0] mem_rdata;

    sram #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(NEURON_BITS)) neuron_mem (
        .clk(clk),
        .we_a(mem_we), .addr_a(mem_addr), .wdata_a(mem_wdata), .rdata_a(mem_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    reg                    ref_we;
    reg  [NEURON_BITS-1:0] ref_addr;
    reg  [3:0]             ref_wdata;
    wire [3:0]             ref_rdata_raw;

    sram #(.DATA_WIDTH(4), .ADDR_WIDTH(NEURON_BITS)) refrac_mem (
        .clk(clk),
        .we_a(ref_we), .addr_a(ref_addr), .wdata_a(ref_wdata), .rdata_a(ref_rdata_raw),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    wire                   wt_we_internal;
    reg                    wt_we_core;
    reg  [WEIGHT_BITS-1:0] wt_addr_core;
    reg  signed [DATA_WIDTH-1:0] wt_wdata_core;
    wire signed [DATA_WIDTH-1:0] wt_rdata;

    wire                   wt_we_mux   = (state == S_IDLE) ? weight_we : wt_we_core;
    wire [WEIGHT_BITS-1:0] wt_addr_mux = (state == S_IDLE) ? weight_addr : wt_addr_core;
    wire signed [DATA_WIDTH-1:0] wt_wdata_mux = (state == S_IDLE) ? weight_data : wt_wdata_core;

    sram #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(WEIGHT_BITS)) weight_mem (
        .clk(clk),
        .we_a(wt_we_mux), .addr_a(wt_addr_mux), .wdata_a(wt_wdata_mux), .rdata_a(wt_rdata),
        .addr_b({WEIGHT_BITS{1'b0}}), .rdata_b()
    );

    reg                    acc_we;
    reg  [NEURON_BITS-1:0] acc_addr;
    reg  signed [DATA_WIDTH-1:0] acc_wdata;
    wire signed [DATA_WIDTH-1:0] acc_rdata;

    sram #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(NEURON_BITS)) acc_mem (
        .clk(clk),
        .we_a(acc_we), .addr_a(acc_addr), .wdata_a(acc_wdata), .rdata_a(acc_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    reg                    trace_we;
    reg  [NEURON_BITS-1:0] trace_addr;
    reg  [7:0]             trace_wdata;
    wire [7:0]             trace_rdata;

    sram #(.DATA_WIDTH(8), .ADDR_WIDTH(NEURON_BITS)) trace_mem (
        .clk(clk),
        .we_a(trace_we), .addr_a(trace_addr), .wdata_a(trace_wdata), .rdata_a(trace_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    reg [NUM_NEURONS-1:0] spike_buf_prev;
    reg [NUM_NEURONS-1:0] spike_buf_curr;
    reg [NUM_NEURONS-1:0] spike_buf_temp;

    reg [NEURON_BITS-1:0]       proc_neuron;
    reg [NEURON_BITS:0]         deliver_src;
    reg [NEURON_BITS:0]         deliver_dst;
    reg signed [DATA_WIDTH-1:0] proc_potential;
    reg [3:0]                   proc_refrac;
    reg signed [DATA_WIDTH-1:0] proc_input;
    reg                         proc_spiked;

    reg [NEURON_BITS-1:0] spike_scan_idx;
    reg                   found_spike;

    wire ext_acc_we = ext_valid && (state == S_IDLE || state == S_DONE);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            spike_buf_prev <= 0;
            spike_buf_curr <= 0;
            timestep_done  <= 0;
            spike_out_valid <= 0;
            total_spikes   <= 0;
            timestep_count <= 0;
            mem_we <= 0; ref_we <= 0; acc_we <= 0;
            wt_we_core <= 0; trace_we <= 0;
            proc_neuron <= 0;
            deliver_src <= 0;
            deliver_dst <= 0;
            spike_scan_idx <= 0;
        end else begin
            mem_we <= 0;
            ref_we <= 0;
            acc_we <= 0;
            wt_we_core <= 0;
            trace_we <= 0;
            timestep_done <= 0;
            spike_out_valid <= 0;

            if (inject_spike_valid) begin
                spike_buf_curr[inject_spike_id] <= 1'b1;
            end

            if (ext_valid && state == S_IDLE) begin
                acc_we    <= 1;
                acc_addr  <= ext_neuron_id;
                acc_wdata <= ext_current;
            end

            case (state)
                S_IDLE: begin
                    if (start) begin
                        state       <= S_DELIVER_INIT;
                        deliver_src <= 0;
                        deliver_dst <= 0;
                    end
                end

                S_DELIVER_INIT: begin
                    if (deliver_src < NUM_NEURONS) begin
                        if (spike_buf_prev[deliver_src[NEURON_BITS-1:0]]) begin
                            deliver_dst <= 0;
                            wt_addr_core <= {deliver_src[NEURON_BITS-1:0], {NEURON_BITS{1'b0}}};
                            acc_addr <= 0;
                            state <= S_DELIVER_READ;
                        end else begin
                            deliver_src <= deliver_src + 1;
                        end
                    end else begin
                        state       <= S_UPDATE_INIT;
                        proc_neuron <= 0;
                    end
                end

                S_DELIVER_READ: begin
                    wt_addr_core <= {deliver_src[NEURON_BITS-1:0], deliver_dst[NEURON_BITS-1:0]};
                    acc_addr     <= deliver_dst[NEURON_BITS-1:0];
                    state        <= S_DELIVER_ACC;
                end

                S_DELIVER_ACC: begin
                    if (deliver_src[NEURON_BITS-1:0] != deliver_dst[NEURON_BITS-1:0]) begin
                        acc_we    <= 1;
                        acc_addr  <= deliver_dst[NEURON_BITS-1:0];
                        acc_wdata <= acc_rdata + wt_rdata;
                    end
                    state <= S_DELIVER_NEXT;
                end

                S_DELIVER_NEXT: begin
                    if (deliver_dst < NUM_NEURONS - 1) begin
                        deliver_dst  <= deliver_dst + 1;
                        wt_addr_core <= {deliver_src[NEURON_BITS-1:0], deliver_dst[NEURON_BITS-1:0] + {{(NEURON_BITS-1){1'b0}}, 1'b1}};
                        acc_addr     <= deliver_dst[NEURON_BITS-1:0] + 1;
                        state        <= S_DELIVER_READ;
                    end else begin
                        deliver_src <= deliver_src + 1;
                        state       <= S_DELIVER_INIT;
                    end
                end

                S_UPDATE_INIT: begin
                    mem_addr  <= proc_neuron;
                    ref_addr  <= proc_neuron;
                    acc_addr  <= proc_neuron;
                    trace_addr <= proc_neuron;
                    state     <= S_UPDATE_READ;
                end

                S_UPDATE_READ: begin
                    mem_addr   <= proc_neuron;
                    ref_addr   <= proc_neuron;
                    acc_addr   <= proc_neuron;
                    trace_addr <= proc_neuron;
                    state      <= S_UPDATE_CALC;
                end

                S_UPDATE_CALC: begin
                    proc_potential <= mem_rdata;
                    proc_refrac   <= ref_rdata_raw;
                    proc_input    <= acc_rdata;
                    proc_spiked   <= 0;

                    if (ref_rdata_raw > 0) begin
                        proc_potential <= RESTING_POT;
                        proc_refrac   <= ref_rdata_raw - 1;
                        if (trace_rdata > TRACE_DECAY)
                            trace_wdata <= trace_rdata - TRACE_DECAY;
                        else
                            trace_wdata <= 0;
                    end else begin
                        if (mem_rdata + acc_rdata - LEAK_RATE >= THRESHOLD) begin
                            proc_potential <= RESTING_POT;
                            proc_refrac   <= REFRAC_CYCLES[3:0];
                            proc_spiked   <= 1;
                            trace_wdata   <= TRACE_MAX;
                        end else if (mem_rdata + acc_rdata > LEAK_RATE) begin
                            proc_potential <= mem_rdata + acc_rdata - LEAK_RATE;
                            if (trace_rdata > TRACE_DECAY)
                                trace_wdata <= trace_rdata - TRACE_DECAY;
                            else
                                trace_wdata <= 0;
                        end else begin
                            proc_potential <= RESTING_POT;
                            if (trace_rdata > TRACE_DECAY)
                                trace_wdata <= trace_rdata - TRACE_DECAY;
                            else
                                trace_wdata <= 0;
                        end
                    end

                    state <= S_UPDATE_WRITE;
                end

                S_UPDATE_WRITE: begin
                    mem_we    <= 1;
                    mem_addr  <= proc_neuron;
                    mem_wdata <= proc_potential;

                    ref_we    <= 1;
                    ref_addr  <= proc_neuron;
                    ref_wdata <= proc_refrac;

                    acc_we    <= 1;
                    acc_addr  <= proc_neuron;
                    acc_wdata <= 0;

                    trace_we   <= 1;
                    trace_addr <= proc_neuron;

                    if (proc_spiked) begin
                        spike_buf_curr[proc_neuron] <= 1'b1;
                        spike_out_valid <= 1;
                        spike_out_id    <= proc_neuron;
                        total_spikes    <= total_spikes + 1;
                    end

                    if (proc_neuron < NUM_NEURONS - 1) begin
                        proc_neuron <= proc_neuron + 1;
                        state       <= S_UPDATE_INIT;
                    end else begin
                        if (learn_enable)
                            state <= S_LEARN;
                        else
                            state <= S_DONE;
                        deliver_src <= 0;
                        deliver_dst <= 0;
                    end
                end

                S_LEARN: begin
                    if (deliver_src < NUM_NEURONS) begin
                        if (spike_buf_curr[deliver_src[NEURON_BITS-1:0]]) begin
                            if (deliver_dst < NUM_NEURONS) begin
                                if (deliver_dst[NEURON_BITS-1:0] != deliver_src[NEURON_BITS-1:0]) begin
                                    wt_addr_core <= {deliver_dst[NEURON_BITS-1:0], deliver_src[NEURON_BITS-1:0]};
                                    trace_addr   <= deliver_dst[NEURON_BITS-1:0];
                                    state        <= S_LEARN_WRITE;
                                end else begin
                                    deliver_dst <= deliver_dst + 1;
                                end
                            end else begin
                                deliver_src <= deliver_src + 1;
                                deliver_dst <= 0;
                            end
                        end else begin
                            deliver_src <= deliver_src + 1;
                            deliver_dst <= 0;
                        end
                    end else begin
                        state <= S_DONE;
                    end
                end

                S_LEARN_WRITE: begin
                    if (trace_rdata > 0) begin
                        wt_we_core   <= 1;
                        wt_addr_core <= {deliver_dst[NEURON_BITS-1:0], deliver_src[NEURON_BITS-1:0]};
                        if (wt_rdata + (trace_rdata >> LEARN_SHIFT) > $signed(THRESHOLD))
                            wt_wdata_core <= THRESHOLD;
                        else
                            wt_wdata_core <= wt_rdata + (trace_rdata >> LEARN_SHIFT);
                    end

                    deliver_dst <= deliver_dst + 1;
                    state       <= S_LEARN;
                end

                S_DONE: begin
                    spike_buf_prev <= spike_buf_curr;
                    spike_buf_curr <= 0;

                    timestep_done  <= 1;
                    timestep_count <= timestep_count + 1;
                    proc_neuron    <= 0;
                    deliver_src    <= 0;

                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
