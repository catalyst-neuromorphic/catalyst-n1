// ============================================================================
// MMIO Bridge
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

module mmio_bridge #(
    parameter CORE_ID_BITS   = 7,
    parameter NEURON_BITS    = 10,
    parameter DATA_WIDTH     = 16,
    parameter POOL_ADDR_BITS = 15,
    parameter ROUTE_SLOT_BITS = 3,
    parameter GLOBAL_ROUTE_SLOT_BITS = 2,
    parameter COUNT_BITS     = 12
)(
    input  wire        clk,
    input  wire        rst_n,

    input  wire        mgmt_phase,

    input  wire        mmio_valid,
    input  wire        mmio_we,
    input  wire [15:0] mmio_addr,
    input  wire [31:0] mmio_wdata,
    output reg  [31:0] mmio_rdata,
    output reg         mmio_ready,

    output reg                          mesh_start,
    output reg                          ext_valid,
    output reg  [CORE_ID_BITS-1:0]     ext_core,
    output reg  [NEURON_BITS-1:0]      ext_neuron_id,
    output reg  signed [DATA_WIDTH-1:0] ext_current,

    output reg                          prog_param_we,
    output reg  [CORE_ID_BITS-1:0]     prog_param_core,
    output reg  [NEURON_BITS-1:0]      prog_param_neuron,
    output reg  [4:0]                   prog_param_id,
    output reg  signed [DATA_WIDTH-1:0] prog_param_value,

    output reg                          probe_read,
    output reg  [CORE_ID_BITS-1:0]     probe_core,
    output reg  [NEURON_BITS-1:0]      probe_neuron,
    output reg  [3:0]                   probe_state_id,
    input  wire signed [DATA_WIDTH-1:0] probe_data,
    input  wire                         probe_valid,

    output reg  [7:0]  uart_tx_data,
    output reg         uart_tx_valid,
    input  wire        uart_tx_ready,
    input  wire [7:0]  uart_rx_data,
    input  wire        uart_rx_valid,

    input  wire        rv_halted,
    input  wire        rv_running,
    input  wire [31:0] timestep_count,

    output reg         learn_enable,
    output reg         graded_enable,
    output reg         dendritic_enable,
    output reg         async_enable,
    output reg         threefactor_enable,
    output reg         noise_enable,
    output reg         skip_idle_enable,

    output reg  signed [DATA_WIDTH-1:0] reward_value,

    output reg                              prog_route_we,
    output reg  [CORE_ID_BITS-1:0]         prog_route_src_core,
    output reg  [NEURON_BITS-1:0]          prog_route_src_neuron,
    output reg  [ROUTE_SLOT_BITS-1:0]      prog_route_slot,
    output reg  [CORE_ID_BITS-1:0]         prog_route_dest_core,
    output reg  [NEURON_BITS-1:0]          prog_route_dest_neuron,
    output reg  signed [DATA_WIDTH-1:0]    prog_route_weight,

    output reg                              prog_delay_we,
    output reg  [CORE_ID_BITS-1:0]         prog_delay_core,
    output reg  [POOL_ADDR_BITS-1:0]       prog_delay_addr,
    output reg  [5:0]                      prog_delay_value,

    output reg                              prog_ucode_we,
    output reg  [CORE_ID_BITS-1:0]         prog_ucode_core,
    output reg  [7:0]                      prog_ucode_addr,
    output reg  [31:0]                     prog_ucode_data,

    output reg  [7:0]                      dvfs_stall,

    output reg                              prog_index_we,
    output reg  [CORE_ID_BITS-1:0]         prog_index_core,
    output reg  [NEURON_BITS-1:0]          prog_index_neuron,
    output reg  [POOL_ADDR_BITS-1:0]       prog_index_base,
    output reg  [COUNT_BITS-1:0]           prog_index_count,

    output reg                              prog_noise_seed_we,
    output reg  [CORE_ID_BITS-1:0]         prog_noise_seed_core,
    output reg  [31:0]                     prog_noise_seed_value,

    output reg                              prog_dend_parent_we,
    output reg  [CORE_ID_BITS-1:0]         prog_dend_parent_core,
    output reg  [NEURON_BITS-1:0]          prog_dend_parent_neuron,
    output reg  [7:0]                      prog_dend_parent_data,

    output reg                              prog_global_route_we,
    output reg  [CORE_ID_BITS-1:0]         prog_global_route_src_core,
    output reg  [NEURON_BITS-1:0]          prog_global_route_src_neuron,
    output reg  [GLOBAL_ROUTE_SLOT_BITS-1:0] prog_global_route_slot,
    output reg  [CORE_ID_BITS-1:0]         prog_global_route_dest_core,
    output reg  [NEURON_BITS-1:0]          prog_global_route_dest_neuron,
    output reg  signed [DATA_WIDTH-1:0]    prog_global_route_weight,

    input  wire [31:0] perf_spike_count,
    input  wire [31:0] perf_synop_count,
    input  wire [31:0] perf_active_cycles,
    input  wire [31:0] perf_power_estimate,

    output reg                             perf_reset_we,
    output reg  [CORE_ID_BITS-1:0]        perf_reset_core,

    output reg  [31:0] debug_bp_addr_0,
    output reg  [31:0] debug_bp_addr_1,
    output reg  [31:0] debug_bp_addr_2,
    output reg  [31:0] debug_bp_addr_3,
    output reg  [3:0]  debug_bp_enable,
    output reg         debug_resume,
    output reg         debug_halt_req,
    output reg         debug_single_step
);

    reg [CORE_ID_BITS-1:0]  sel_core;
    reg [NEURON_BITS-1:0]   sel_neuron;
    reg [POOL_ADDR_BITS-1:0] sel_pool_addr;

    reg [CORE_ID_BITS-1:0]         route_dest_core;
    reg [NEURON_BITS-1:0]          route_dest_neuron;
    reg signed [DATA_WIDTH-1:0]    route_weight;

    reg [POOL_ADDR_BITS-1:0]       index_base;

    reg [7:0]                      ucode_addr;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mmio_rdata     <= 32'd0;
            mmio_ready     <= 1'b0;
            mesh_start     <= 1'b0;
            ext_valid      <= 1'b0;
            ext_core       <= 0;
            ext_neuron_id  <= 0;
            ext_current    <= 0;
            prog_param_we  <= 1'b0;
            prog_param_core   <= 0;
            prog_param_neuron <= 0;
            prog_param_id     <= 0;
            prog_param_value  <= 0;
            probe_read     <= 1'b0;
            probe_core     <= 0;
            probe_neuron   <= 0;
            probe_state_id <= 0;
            uart_tx_data   <= 8'd0;
            uart_tx_valid  <= 1'b0;
            sel_core       <= 0;
            sel_neuron     <= 0;
            sel_pool_addr  <= 0;
            learn_enable       <= 1'b0;
            graded_enable      <= 1'b0;
            dendritic_enable   <= 1'b0;
            async_enable       <= 1'b0;
            threefactor_enable <= 1'b0;
            noise_enable       <= 1'b0;
            skip_idle_enable   <= 1'b0;
            reward_value       <= 0;
            prog_route_we          <= 1'b0;
            prog_route_src_core    <= 0;
            prog_route_src_neuron  <= 0;
            prog_route_slot        <= 0;
            prog_route_dest_core   <= 0;
            prog_route_dest_neuron <= 0;
            prog_route_weight      <= 0;
            route_dest_core        <= 0;
            route_dest_neuron      <= 0;
            route_weight           <= 0;
            prog_delay_we    <= 1'b0;
            prog_delay_core  <= 0;
            prog_delay_addr  <= 0;
            prog_delay_value <= 0;
            prog_ucode_we   <= 1'b0;
            prog_ucode_core <= 0;
            prog_ucode_addr <= 0;
            prog_ucode_data <= 0;
            ucode_addr      <= 0;
            dvfs_stall       <= 8'd0;
            prog_index_we     <= 1'b0;
            prog_index_core   <= 0;
            prog_index_neuron <= 0;
            prog_index_base   <= 0;
            prog_index_count  <= 0;
            index_base        <= 0;
            prog_noise_seed_we    <= 1'b0;
            prog_noise_seed_core  <= 0;
            prog_noise_seed_value <= 0;
            prog_dend_parent_we     <= 1'b0;
            prog_dend_parent_core   <= 0;
            prog_dend_parent_neuron <= 0;
            prog_dend_parent_data   <= 0;
            prog_global_route_we          <= 1'b0;
            prog_global_route_src_core    <= 0;
            prog_global_route_src_neuron  <= 0;
            prog_global_route_slot        <= 0;
            prog_global_route_dest_core   <= 0;
            prog_global_route_dest_neuron <= 0;
            prog_global_route_weight      <= 0;
            perf_reset_we   <= 1'b0;
            perf_reset_core <= 0;
            debug_bp_addr_0    <= 32'd0;
            debug_bp_addr_1    <= 32'd0;
            debug_bp_addr_2    <= 32'd0;
            debug_bp_addr_3    <= 32'd0;
            debug_bp_enable    <= 4'd0;
            debug_resume       <= 1'b0;
            debug_halt_req     <= 1'b0;
            debug_single_step  <= 1'b0;
        end else begin
            mmio_ready     <= 1'b0;
            mesh_start     <= 1'b0;
            ext_valid      <= 1'b0;
            prog_param_we  <= 1'b0;
            probe_read     <= 1'b0;
            uart_tx_valid  <= 1'b0;
            prog_route_we        <= 1'b0;
            prog_delay_we        <= 1'b0;
            prog_ucode_we        <= 1'b0;
            prog_index_we        <= 1'b0;
            prog_noise_seed_we   <= 1'b0;
            prog_dend_parent_we  <= 1'b0;
            prog_global_route_we <= 1'b0;
            perf_reset_we        <= 1'b0;
            debug_resume         <= 1'b0;
            debug_halt_req       <= 1'b0;
            debug_single_step    <= 1'b0;

            if (mmio_valid && !mmio_ready) begin
                mmio_ready <= 1'b1;

                if (mmio_we) begin
                    case (mmio_addr)
                        16'h0000: begin
                            if (mmio_wdata[0]) mesh_start <= 1'b1;
                        end
                        16'h0004: sel_core   <= mmio_wdata[CORE_ID_BITS-1:0];
                        16'h0008: sel_neuron <= mmio_wdata[NEURON_BITS-1:0];
                        16'h000C: begin
                            prog_param_we     <= mgmt_phase;
                            prog_param_core   <= sel_core;
                            prog_param_neuron <= sel_neuron;
                            prog_param_id     <= mmio_wdata[20:16];
                            prog_param_value  <= mmio_wdata[DATA_WIDTH-1:0];
                        end
                        16'h0010: sel_pool_addr <= mmio_wdata[POOL_ADDR_BITS-1:0];
                        16'h0018: begin
                            ext_valid     <= 1'b1;
                            ext_core      <= sel_core;
                            ext_neuron_id <= mmio_wdata[NEURON_BITS-1:0];
                            ext_current   <= mmio_wdata[DATA_WIDTH+NEURON_BITS-1:NEURON_BITS];
                        end
                        16'h0020: begin
                            uart_tx_data  <= mmio_wdata[7:0];
                            uart_tx_valid <= 1'b1;
                        end


                        16'h0030: begin
                            if (mgmt_phase) begin
                                learn_enable       <= mmio_wdata[0];
                                graded_enable      <= mmio_wdata[1];
                                dendritic_enable   <= mmio_wdata[2];
                                async_enable       <= mmio_wdata[3];
                                threefactor_enable <= mmio_wdata[4];
                                noise_enable       <= mmio_wdata[5];
                                skip_idle_enable   <= mmio_wdata[6];
                            end
                        end

                        16'h0034: begin
                            if (mgmt_phase)
                                reward_value <= mmio_wdata[DATA_WIDTH-1:0];
                        end

                        16'h0038: begin
                            route_dest_core <= mmio_wdata[CORE_ID_BITS-1:0];
                        end

                        16'h003C: begin
                            route_dest_neuron <= mmio_wdata[NEURON_BITS-1:0];
                        end

                        16'h0040: begin
                            route_weight <= mmio_wdata[DATA_WIDTH-1:0];
                        end

                        16'h0044: begin
                            if (mgmt_phase) begin
                                prog_route_we          <= 1'b1;
                                prog_route_src_core    <= sel_core;
                                prog_route_src_neuron  <= sel_neuron;
                                prog_route_slot        <= mmio_wdata[ROUTE_SLOT_BITS-1:0];
                                prog_route_dest_core   <= route_dest_core;
                                prog_route_dest_neuron <= route_dest_neuron;
                                prog_route_weight      <= route_weight;
                            end
                        end

                        16'h0048: begin
                            if (mgmt_phase) begin
                                prog_delay_we    <= 1'b1;
                                prog_delay_core  <= sel_core;
                                prog_delay_addr  <= sel_pool_addr;
                                prog_delay_value <= mmio_wdata[5:0];
                            end
                        end

                        16'h004C: begin
                            ucode_addr <= mmio_wdata[7:0];
                        end

                        16'h0050: begin
                            if (mgmt_phase) begin
                                prog_ucode_we   <= 1'b1;
                                prog_ucode_core <= sel_core;
                                prog_ucode_addr <= ucode_addr;
                                prog_ucode_data <= mmio_wdata;
                            end
                        end

                        16'h0054: begin
                            if (mgmt_phase)
                                dvfs_stall <= mmio_wdata[7:0];
                        end

                        16'h0058: begin
                            if (mgmt_phase) begin
                                perf_reset_we   <= 1'b1;
                                perf_reset_core <= sel_core;
                            end
                        end

                        16'h005C: begin
                            index_base <= mmio_wdata[POOL_ADDR_BITS-1:0];
                        end

                        16'h0060: begin
                            if (mgmt_phase) begin
                                prog_index_we     <= 1'b1;
                                prog_index_core   <= sel_core;
                                prog_index_neuron <= sel_neuron;
                                prog_index_base   <= index_base;
                                prog_index_count  <= mmio_wdata[COUNT_BITS-1:0];
                            end
                        end

                        16'h0064: begin
                            if (mgmt_phase) begin
                                prog_noise_seed_we    <= 1'b1;
                                prog_noise_seed_core  <= sel_core;
                                prog_noise_seed_value <= mmio_wdata;
                            end
                        end

                        16'h0068: begin
                            if (mgmt_phase) begin
                                prog_dend_parent_we     <= 1'b1;
                                prog_dend_parent_core   <= sel_core;
                                prog_dend_parent_neuron <= sel_neuron;
                                prog_dend_parent_data   <= mmio_wdata[7:0];
                            end
                        end

                        16'h006C: begin
                            if (mgmt_phase) begin
                                prog_global_route_we          <= 1'b1;
                                prog_global_route_src_core    <= sel_core;
                                prog_global_route_src_neuron  <= sel_neuron;
                                prog_global_route_slot        <= mmio_wdata[GLOBAL_ROUTE_SLOT_BITS-1:0];
                                prog_global_route_dest_core   <= route_dest_core;
                                prog_global_route_dest_neuron <= route_dest_neuron;
                                prog_global_route_weight      <= route_weight;
                            end
                        end


                        16'h0090: begin
                            debug_resume      <= mmio_wdata[0];
                            debug_halt_req    <= mmio_wdata[1];
                            debug_single_step <= mmio_wdata[2];
                        end

                        16'h0094: debug_bp_addr_0 <= mmio_wdata;
                        16'h0098: debug_bp_addr_1 <= mmio_wdata;
                        16'h009C: debug_bp_addr_2 <= mmio_wdata;
                        16'h00A0: debug_bp_addr_3 <= mmio_wdata;
                        16'h00A4: debug_bp_enable <= mmio_wdata[3:0];

                        default: ;
                    endcase
                end else begin
                    case (mmio_addr)
                        16'h0000: mmio_rdata <= {30'd0, rv_running, rv_halted};
                        16'h0004: mmio_rdata <= {{(32-CORE_ID_BITS){1'b0}}, sel_core};
                        16'h0008: mmio_rdata <= {{(32-NEURON_BITS){1'b0}}, sel_neuron};
                        16'h000C: begin
                            probe_read     <= 1'b1;
                            probe_core     <= sel_core;
                            probe_neuron   <= sel_neuron;
                            probe_state_id <= mmio_wdata[3:0];
                            mmio_rdata     <= {{(32-DATA_WIDTH){probe_data[DATA_WIDTH-1]}}, probe_data};
                        end
                        16'h0024: mmio_rdata <= {24'd0, uart_rx_data};
                        16'h0028: mmio_rdata <= {30'd0, uart_rx_valid, uart_tx_ready};
                        16'h002C: mmio_rdata <= timestep_count;

                        16'h0070: mmio_rdata <= perf_spike_count;
                        16'h0074: mmio_rdata <= perf_synop_count;
                        16'h0078: mmio_rdata <= perf_active_cycles;
                        16'h007C: mmio_rdata <= perf_power_estimate;

                        default:  mmio_rdata <= 32'd0;
                    endcase
                end
            end
        end
    end

endmodule
