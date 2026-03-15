module host_interface #(
    parameter NUM_CORES      = 4,
    parameter CORE_ID_BITS   = 2,
    parameter NUM_NEURONS    = 1024,
    parameter NEURON_BITS    = 10,
    parameter DATA_WIDTH     = 16,
    parameter POOL_ADDR_BITS = 15,
    parameter COUNT_BITS     = 12,
    parameter ROUTE_SLOT_BITS = 3,
    parameter GLOBAL_ROUTE_SLOT_BITS = 2
)(
    input  wire       clk,
    input  wire       rst_n,

    input  wire [7:0] rx_data,
    input  wire       rx_valid,
    output reg  [7:0] tx_data,
    output reg        tx_valid,
    input  wire       tx_ready,

    output reg                         mesh_start,

    output reg                              mesh_prog_pool_we,
    output reg  [CORE_ID_BITS-1:0]         mesh_prog_pool_core,
    output reg  [POOL_ADDR_BITS-1:0]       mesh_prog_pool_addr,
    output reg  [NEURON_BITS-1:0]          mesh_prog_pool_src,
    output reg  [NEURON_BITS-1:0]          mesh_prog_pool_target,
    output reg  signed [DATA_WIDTH-1:0]    mesh_prog_pool_weight,
    output reg  [1:0]                      mesh_prog_pool_comp,

    output reg                              mesh_prog_index_we,
    output reg  [CORE_ID_BITS-1:0]         mesh_prog_index_core,
    output reg  [NEURON_BITS-1:0]          mesh_prog_index_neuron,
    output reg  [POOL_ADDR_BITS-1:0]       mesh_prog_index_base,
    output reg  [COUNT_BITS-1:0]           mesh_prog_index_count,
    output reg  [1:0]                      mesh_prog_index_format,

    output reg                              mesh_prog_route_we,
    output reg  [CORE_ID_BITS-1:0]         mesh_prog_route_src_core,
    output reg  [NEURON_BITS-1:0]          mesh_prog_route_src_neuron,
    output reg  [ROUTE_SLOT_BITS-1:0]      mesh_prog_route_slot,
    output reg  [CORE_ID_BITS-1:0]         mesh_prog_route_dest_core,
    output reg  [NEURON_BITS-1:0]          mesh_prog_route_dest_neuron,
    output reg  signed [DATA_WIDTH-1:0]    mesh_prog_route_weight,

    output reg                              mesh_prog_global_route_we,
    output reg  [CORE_ID_BITS-1:0]         mesh_prog_global_route_src_core,
    output reg  [NEURON_BITS-1:0]          mesh_prog_global_route_src_neuron,
    output reg  [GLOBAL_ROUTE_SLOT_BITS-1:0] mesh_prog_global_route_slot,
    output reg  [CORE_ID_BITS-1:0]         mesh_prog_global_route_dest_core,
    output reg  [NEURON_BITS-1:0]          mesh_prog_global_route_dest_neuron,
    output reg  signed [DATA_WIDTH-1:0]    mesh_prog_global_route_weight,

    output reg                         mesh_ext_valid,
    output reg  [CORE_ID_BITS-1:0]    mesh_ext_core,
    output reg  [NEURON_BITS-1:0]     mesh_ext_neuron_id,
    output reg  signed [DATA_WIDTH-1:0] mesh_ext_current,

    output reg        mesh_learn_enable,
    output reg        mesh_graded_enable,
    output reg        mesh_dendritic_enable,
    output reg        mesh_async_enable,
    output reg        mesh_threefactor_enable,
    output reg signed [DATA_WIDTH-1:0] mesh_reward_value,
    output reg        mesh_noise_enable,
    output reg        mesh_skip_idle_enable,
    output reg        mesh_scale_u_enable,

    output reg                              mesh_prog_delay_we,
    output reg  [CORE_ID_BITS-1:0]         mesh_prog_delay_core,
    output reg  [POOL_ADDR_BITS-1:0]       mesh_prog_delay_addr,
    output reg  [5:0]                      mesh_prog_delay_value,

    output reg                              mesh_prog_ucode_we,
    output reg  [CORE_ID_BITS-1:0]         mesh_prog_ucode_core,
    output reg  [7:0]                      mesh_prog_ucode_addr,
    output reg  [31:0]                     mesh_prog_ucode_data,

    output reg                         mesh_prog_param_we,
    output reg  [CORE_ID_BITS-1:0]    mesh_prog_param_core,
    output reg  [NEURON_BITS-1:0]     mesh_prog_param_neuron,
    output reg  [4:0]                 mesh_prog_param_id,
    output reg  signed [DATA_WIDTH-1:0] mesh_prog_param_value,

    output reg                              mesh_probe_read,
    output reg  [CORE_ID_BITS-1:0]         mesh_probe_core,
    output reg  [NEURON_BITS-1:0]          mesh_probe_neuron,
    output reg  [4:0]                      mesh_probe_state_id,
    output reg  [POOL_ADDR_BITS-1:0]       mesh_probe_pool_addr,
    input  wire signed [DATA_WIDTH-1:0]    mesh_probe_data,
    input  wire                            mesh_probe_valid,

    output reg  [7:0]  mesh_dvfs_stall,

    input  wire       mesh_timestep_done,
    input  wire [5:0] mesh_state,
    input  wire [31:0] mesh_total_spikes,
    input  wire [31:0] mesh_timestep_count
);

    localparam CMD_PROG_POOL   = 8'h01;
    localparam CMD_PROG_ROUTE  = 8'h02;
    localparam CMD_STIMULUS    = 8'h03;
    localparam CMD_RUN         = 8'h04;
    localparam CMD_STATUS      = 8'h05;
    localparam CMD_LEARN_CFG   = 8'h06;
    localparam CMD_PROG_NEURON = 8'h07;
    localparam CMD_PROG_INDEX  = 8'h08;
    localparam CMD_REWARD      = 8'h09;
    localparam CMD_PROG_DELAY  = 8'h0A;
    localparam CMD_PROG_FORMAT = 8'h0B;
    localparam CMD_PROG_LEARN  = 8'h0C;
    localparam CMD_NOISE_SEED  = 8'h0D;
    localparam CMD_READ_WEIGHT = 8'h0E;
    localparam CMD_PROG_DEND_TREE = 8'h0F;
    localparam CMD_PROG_GLOBAL_ROUTE = 8'h10;
    localparam CMD_DVFS_CFG    = 8'h1C;
    localparam CMD_RESET_PERF  = 8'h1D;

    localparam RESP_ACK  = 8'hAA;
    localparam RESP_DONE = 8'hDD;

    localparam HI_IDLE        = 6'd0;
    localparam HI_RECV        = 6'd1;
    localparam HI_EXEC_POOL   = 6'd2;
    localparam HI_EXEC_ROUTE  = 6'd3;
    localparam HI_EXEC_STIM   = 6'd4;
    localparam HI_SEND_ACK    = 6'd5;
    localparam HI_RUN_START   = 6'd6;
    localparam HI_RUN_WAIT    = 6'd7;
    localparam HI_RUN_LOOP    = 6'd8;
    localparam HI_SEND_RESP   = 6'd9;
    localparam HI_EXEC_STATUS = 6'd10;
    localparam HI_SEND_WAIT   = 6'd11;
    localparam HI_EXEC_LEARN  = 6'd12;
    localparam HI_EXEC_PARAM  = 6'd13;
    localparam HI_EXEC_INDEX  = 6'd14;
    localparam HI_EXEC_REWARD = 6'd15;
    localparam HI_EXEC_DELAY     = 6'd16;
    localparam HI_EXEC_FORMAT    = 6'd17;
    localparam HI_EXEC_LEARN_MC  = 6'd18;
    localparam HI_EXEC_SEED      = 6'd19;
    localparam HI_EXEC_READ_WT   = 6'd20;
    localparam HI_EXEC_GLOBAL_ROUTE = 6'd21;
    localparam HI_PROBE_WAIT    = 6'd22;
    localparam HI_PROBE_RESP    = 6'd23;
    localparam HI_EXEC_DEND_TREE = 6'd24;
    localparam HI_EXEC_DVFS      = 6'd25;
    localparam HI_EXEC_RESET_PERF = 6'd26;

    reg [5:0]  state;
    reg [7:0]  cmd;
    reg [4:0]  byte_cnt;
    reg [4:0]  payload_len;
    reg [7:0]  payload [0:15];

    reg [15:0] run_remaining;
    reg [31:0] run_spike_base;

    reg [7:0]  resp_buf [0:4];
    reg [2:0]  resp_len;
    reg [2:0]  resp_idx;

    function [4:0] cmd_payload_len;
        input [7:0] opcode;
        case (opcode)
            CMD_PROG_POOL:   cmd_payload_len = 5'd8;
            CMD_PROG_ROUTE:  cmd_payload_len = 5'd9;
            CMD_STIMULUS:    cmd_payload_len = 5'd5;
            CMD_RUN:         cmd_payload_len = 5'd2;
            CMD_STATUS:      cmd_payload_len = 5'd0;
            CMD_LEARN_CFG:   cmd_payload_len = 5'd1;
            CMD_PROG_NEURON: cmd_payload_len = 5'd6;
            CMD_PROG_INDEX:  cmd_payload_len = 5'd7;
            CMD_REWARD:      cmd_payload_len = 5'd2;
            CMD_PROG_DELAY:  cmd_payload_len = 5'd4;
            CMD_PROG_FORMAT: cmd_payload_len = 5'd4;
            CMD_PROG_LEARN:  cmd_payload_len = 5'd6;
            CMD_NOISE_SEED:  cmd_payload_len = 5'd3;
            CMD_READ_WEIGHT: cmd_payload_len = 5'd4;
            CMD_PROG_DEND_TREE: cmd_payload_len = 5'd4;
            CMD_PROG_GLOBAL_ROUTE: cmd_payload_len = 5'd9;
            CMD_DVFS_CFG:    cmd_payload_len = 5'd1;
            CMD_RESET_PERF:  cmd_payload_len = 5'd1;
            default:         cmd_payload_len = 5'd0;
        endcase
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state              <= HI_IDLE;
            cmd                <= 0;
            byte_cnt           <= 0;
            payload_len        <= 0;
            tx_data            <= 0;
            tx_valid           <= 0;
            mesh_start         <= 0;
            mesh_prog_pool_we  <= 0;
            mesh_prog_pool_core   <= 0;
            mesh_prog_pool_addr   <= 0;
            mesh_prog_pool_src    <= 0;
            mesh_prog_pool_target <= 0;
            mesh_prog_pool_weight <= 0;
            mesh_prog_pool_comp   <= 0;
            mesh_prog_index_we     <= 0;
            mesh_prog_index_core   <= 0;
            mesh_prog_index_neuron <= 0;
            mesh_prog_index_base   <= 0;
            mesh_prog_index_count  <= 0;
            mesh_prog_index_format <= 0;
            mesh_prog_route_we <= 0;
            mesh_prog_route_src_core   <= 0;
            mesh_prog_route_src_neuron <= 0;
            mesh_prog_route_slot       <= 0;
            mesh_prog_route_dest_core  <= 0;
            mesh_prog_route_dest_neuron<= 0;
            mesh_prog_route_weight     <= 0;
            mesh_prog_global_route_we          <= 0;
            mesh_prog_global_route_src_core    <= 0;
            mesh_prog_global_route_src_neuron  <= 0;
            mesh_prog_global_route_slot        <= 0;
            mesh_prog_global_route_dest_core   <= 0;
            mesh_prog_global_route_dest_neuron <= 0;
            mesh_prog_global_route_weight      <= 0;
            mesh_ext_valid     <= 0;
            mesh_ext_core      <= 0;
            mesh_ext_neuron_id <= 0;
            mesh_ext_current   <= 0;
            mesh_learn_enable     <= 0;
            mesh_graded_enable    <= 0;
            mesh_dendritic_enable <= 0;
            mesh_async_enable     <= 0;
            mesh_threefactor_enable <= 0;
            mesh_noise_enable     <= 0;
            mesh_skip_idle_enable <= 0;
            mesh_scale_u_enable   <= 0;
            mesh_reward_value     <= 0;
            mesh_prog_delay_we     <= 0;
            mesh_prog_delay_core   <= 0;
            mesh_prog_delay_addr   <= 0;
            mesh_prog_delay_value  <= 0;
            mesh_prog_ucode_we     <= 0;
            mesh_prog_ucode_core   <= 0;
            mesh_prog_ucode_addr   <= 0;
            mesh_prog_ucode_data   <= 0;
            mesh_prog_param_we <= 0;
            mesh_prog_param_core   <= 0;
            mesh_prog_param_neuron <= 0;
            mesh_prog_param_id     <= 0;
            mesh_prog_param_value  <= 0;
            mesh_probe_read    <= 0;
            mesh_probe_core    <= 0;
            mesh_probe_neuron  <= 0;
            mesh_probe_state_id <= 0;
            mesh_probe_pool_addr <= 0;
            mesh_dvfs_stall    <= 0;
            run_remaining      <= 0;
            run_spike_base     <= 0;
            resp_len           <= 0;
            resp_idx           <= 0;
        end else begin
            mesh_prog_pool_we  <= 0;
            mesh_prog_index_we <= 0;
            mesh_prog_route_we <= 0;
            mesh_prog_global_route_we <= 0;
            mesh_prog_delay_we <= 0;
            mesh_prog_ucode_we <= 0;
            mesh_prog_param_we <= 0;
            mesh_probe_read    <= 0;
            mesh_ext_valid     <= 0;
            mesh_start         <= 0;
            tx_valid           <= 0;

            case (state)

                HI_IDLE: begin
                    if (rx_valid) begin
                        cmd         <= rx_data;
                        payload_len <= cmd_payload_len(rx_data);
                        byte_cnt    <= 0;
                        if (cmd_payload_len(rx_data) == 0) begin
                            case (rx_data)
                                CMD_STATUS: state <= HI_EXEC_STATUS;
                                default:    state <= HI_IDLE;
                            endcase
                        end else begin
                            state <= HI_RECV;
                        end
                    end
                end

                HI_RECV: begin
                    if (rx_valid) begin
                        payload[byte_cnt] <= rx_data;
                        if (byte_cnt == payload_len - 1) begin
                            case (cmd)
                                CMD_PROG_POOL:   state <= HI_EXEC_POOL;
                                CMD_PROG_ROUTE:  state <= HI_EXEC_ROUTE;
                                CMD_STIMULUS:    state <= HI_EXEC_STIM;
                                CMD_RUN:         state <= HI_RUN_START;
                                CMD_LEARN_CFG:   state <= HI_EXEC_LEARN;
                                CMD_PROG_NEURON: state <= HI_EXEC_PARAM;
                                CMD_PROG_INDEX:  state <= HI_EXEC_INDEX;
                                CMD_REWARD:      state <= HI_EXEC_REWARD;
                                CMD_PROG_DELAY:  state <= HI_EXEC_DELAY;
                                CMD_PROG_FORMAT: state <= HI_EXEC_FORMAT;
                                CMD_PROG_LEARN:  state <= HI_EXEC_LEARN_MC;
                                CMD_NOISE_SEED:  state <= HI_EXEC_SEED;
                                CMD_READ_WEIGHT: state <= HI_EXEC_READ_WT;
                                CMD_PROG_DEND_TREE: state <= HI_EXEC_DEND_TREE;
                                CMD_PROG_GLOBAL_ROUTE: state <= HI_EXEC_GLOBAL_ROUTE;
                                CMD_DVFS_CFG:    state <= HI_EXEC_DVFS;
                                CMD_RESET_PERF:  state <= HI_EXEC_RESET_PERF;
                                default:         state <= HI_IDLE;
                            endcase
                        end else begin
                            byte_cnt <= byte_cnt + 1;
                        end
                    end
                end

                HI_EXEC_POOL: begin
                    mesh_prog_pool_we     <= 1;
                    mesh_prog_pool_core   <= payload[0][CORE_ID_BITS-1:0];
                    mesh_prog_pool_addr   <= {payload[1], payload[2]};
                    mesh_prog_pool_comp   <= payload[3][7:6];
                    mesh_prog_pool_src    <= {payload[3][5:4], payload[4]};
                    mesh_prog_pool_target <= {payload[3][3:2], payload[5]};
                    mesh_prog_pool_weight <= {payload[6], payload[7]};
                    state <= HI_SEND_ACK;
                end

                HI_EXEC_INDEX: begin
                    mesh_prog_index_we     <= 1;
                    mesh_prog_index_core   <= payload[0][CORE_ID_BITS-1:0];
                    mesh_prog_index_neuron <= {payload[1], payload[2]};
                    mesh_prog_index_base   <= {payload[3], payload[4]};
                    mesh_prog_index_count  <= {payload[5], payload[6]};
                    mesh_prog_index_format <= payload[5][7:6];
                    state <= HI_SEND_ACK;
                end

                HI_EXEC_REWARD: begin
                    mesh_reward_value <= {payload[0], payload[1]};
                    state <= HI_SEND_ACK;
                end

                HI_EXEC_ROUTE: begin
                    mesh_prog_route_we         <= 1;
                    mesh_prog_route_src_core   <= payload[0][CORE_ID_BITS-1:0];
                    mesh_prog_route_src_neuron <= {payload[1], payload[2]};
                    mesh_prog_route_slot       <= payload[3][ROUTE_SLOT_BITS-1:0];
                    mesh_prog_route_dest_core  <= payload[4][CORE_ID_BITS-1:0];
                    mesh_prog_route_dest_neuron<= {payload[5], payload[6]};
                    mesh_prog_route_weight     <= {payload[7], payload[8]};
                    state <= HI_SEND_ACK;
                end

                HI_EXEC_STIM: begin
                    mesh_ext_valid     <= 1;
                    mesh_ext_core      <= payload[0][CORE_ID_BITS-1:0];
                    mesh_ext_neuron_id <= {payload[1], payload[2]};
                    mesh_ext_current   <= {payload[3], payload[4]};
                    state <= HI_SEND_ACK;
                end

                HI_EXEC_LEARN: begin
                    mesh_learn_enable     <= payload[0][0];
                    mesh_graded_enable    <= payload[0][1];
                    mesh_dendritic_enable <= payload[0][2];
                    mesh_async_enable     <= payload[0][3];
                    mesh_threefactor_enable <= payload[0][4];
                    mesh_noise_enable      <= payload[0][5];
                    mesh_skip_idle_enable  <= payload[0][6];
                    mesh_scale_u_enable    <= payload[0][7];
                    state <= HI_SEND_ACK;
                end

                HI_EXEC_PARAM: begin
                    mesh_prog_param_we     <= 1;
                    mesh_prog_param_core   <= payload[0][CORE_ID_BITS-1:0];
                    mesh_prog_param_neuron <= {payload[1], payload[2]};
                    mesh_prog_param_id     <= payload[3][4:0];
                    mesh_prog_param_value  <= {payload[4], payload[5]};
                    state <= HI_SEND_ACK;
                end

                HI_SEND_ACK: begin
                    if (tx_ready) begin
                        tx_data  <= RESP_ACK;
                        tx_valid <= 1;
                        state    <= HI_IDLE;
                    end
                end

                HI_RUN_START: begin
                    run_remaining  <= {payload[0], payload[1]};
                    run_spike_base <= mesh_total_spikes;
                    mesh_start     <= 1;
                    state          <= HI_RUN_WAIT;
                end

                HI_RUN_WAIT: begin
                    if (mesh_timestep_done) begin
                        state <= HI_RUN_LOOP;
                    end
                end

                HI_RUN_LOOP: begin
                    if (run_remaining <= 1) begin
                        resp_buf[0] <= RESP_DONE;
                        resp_buf[1] <= (mesh_total_spikes - run_spike_base) >> 24;
                        resp_buf[2] <= (mesh_total_spikes - run_spike_base) >> 16;
                        resp_buf[3] <= (mesh_total_spikes - run_spike_base) >> 8;
                        resp_buf[4] <= (mesh_total_spikes - run_spike_base);
                        resp_len    <= 5;
                        resp_idx    <= 0;
                        state       <= HI_SEND_RESP;
                    end else begin
                        run_remaining <= run_remaining - 1;
                        mesh_start    <= 1;
                        state         <= HI_RUN_WAIT;
                    end
                end

                HI_EXEC_STATUS: begin
                    resp_buf[0] <= {3'b0, mesh_state};
                    resp_buf[1] <= mesh_timestep_count >> 24;
                    resp_buf[2] <= mesh_timestep_count >> 16;
                    resp_buf[3] <= mesh_timestep_count >> 8;
                    resp_buf[4] <= mesh_timestep_count;
                    resp_len    <= 5;
                    resp_idx    <= 0;
                    state       <= HI_SEND_RESP;
                end

                HI_SEND_RESP: begin
                    if (tx_ready) begin
                        tx_data  <= resp_buf[resp_idx];
                        tx_valid <= 1;
                        state    <= HI_SEND_WAIT;
                    end
                end

                HI_SEND_WAIT: begin
                    if (resp_idx == resp_len - 1) begin
                        state <= HI_IDLE;
                    end else begin
                        resp_idx <= resp_idx + 1;
                        state    <= HI_SEND_RESP;
                    end
                end

                HI_EXEC_DELAY: begin
                    mesh_prog_delay_we    <= 1;
                    mesh_prog_delay_core  <= payload[0][CORE_ID_BITS-1:0];
                    mesh_prog_delay_addr  <= {payload[1], payload[2]};
                    mesh_prog_delay_value <= payload[3][5:0];
                    state <= HI_SEND_ACK;
                end
                HI_EXEC_FORMAT:   state <= HI_SEND_ACK;

                HI_EXEC_LEARN_MC: begin
                    mesh_prog_ucode_we   <= 1;
                    mesh_prog_ucode_core <= payload[0][CORE_ID_BITS-1:0];
                    mesh_prog_ucode_addr <= payload[1][7:0];
                    mesh_prog_ucode_data <= {payload[2], payload[3], payload[4], payload[5]};
                    state <= HI_SEND_ACK;
                end
                HI_EXEC_SEED:     state <= HI_SEND_ACK;

                HI_EXEC_READ_WT: begin
                    mesh_probe_read     <= 1;
                    mesh_probe_core     <= payload[0][CORE_ID_BITS-1:0];
                    mesh_probe_neuron   <= {payload[1], payload[2]};
                    mesh_probe_state_id <= payload[3][4:0];
                    mesh_probe_pool_addr <= {payload[1], payload[2]};
                    state <= HI_PROBE_WAIT;
                end

                HI_PROBE_WAIT: begin
                    if (mesh_probe_valid) begin
                        resp_buf[0] <= mesh_probe_data[15:8];
                        resp_buf[1] <= mesh_probe_data[7:0];
                        resp_len    <= 2;
                        resp_idx    <= 0;
                        state       <= HI_SEND_RESP;
                    end
                end

                HI_EXEC_GLOBAL_ROUTE: begin
                    mesh_prog_global_route_we          <= 1;
                    mesh_prog_global_route_src_core    <= payload[0][CORE_ID_BITS-1:0];
                    mesh_prog_global_route_src_neuron  <= {payload[1], payload[2]};
                    mesh_prog_global_route_slot        <= payload[3][GLOBAL_ROUTE_SLOT_BITS-1:0];
                    mesh_prog_global_route_dest_core   <= payload[4][CORE_ID_BITS-1:0];
                    mesh_prog_global_route_dest_neuron <= {payload[5], payload[6]};
                    mesh_prog_global_route_weight      <= {payload[7], payload[8]};
                    state <= HI_SEND_ACK;
                end

                HI_EXEC_DEND_TREE: begin
                    mesh_prog_param_we     <= 1;
                    mesh_prog_param_core   <= payload[0][CORE_ID_BITS-1:0];
                    mesh_prog_param_neuron <= {payload[1], payload[2]};
                    mesh_prog_param_id     <= 5'd15;
                    mesh_prog_param_value  <= {{(DATA_WIDTH-6){1'b0}}, payload[3][5:0]};
                    state <= HI_SEND_ACK;
                end

                HI_EXEC_DVFS: begin
                    mesh_dvfs_stall <= payload[0];
                    state <= HI_SEND_ACK;
                end

                HI_EXEC_RESET_PERF: begin
                    mesh_prog_param_we     <= 1;
                    mesh_prog_param_core   <= payload[0][CORE_ID_BITS-1:0];
                    mesh_prog_param_neuron <= 0;
                    mesh_prog_param_id     <= 5'd28;
                    mesh_prog_param_value  <= 0;
                    state <= HI_SEND_ACK;
                end

                default: state <= HI_IDLE;
            endcase
        end
    end

endmodule
