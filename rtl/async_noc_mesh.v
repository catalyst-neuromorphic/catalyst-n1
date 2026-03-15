`timescale 1ns/1ps

module async_noc_mesh #(
    parameter NUM_CORES      = 4,
    parameter CORE_ID_BITS   = 2,
    parameter NUM_NEURONS    = 1024,
    parameter NEURON_BITS    = 10,
    parameter DATA_WIDTH     = 16,
    parameter POOL_DEPTH     = 32768,
    parameter POOL_ADDR_BITS = 15,
    parameter COUNT_BITS     = 12,
    parameter REV_FANIN      = 32,
    parameter REV_SLOT_BITS  = 5,
    parameter THRESHOLD      = 16'sd1000,
    parameter LEAK_RATE      = 16'sd3,
    parameter REFRAC_CYCLES  = 3,
    parameter GRADE_SHIFT    = 7,
    parameter ROUTE_FANOUT     = 8,
    parameter ROUTE_SLOT_BITS  = 3,
    parameter ROUTE_ADDR_W   = CORE_ID_BITS + NEURON_BITS + ROUTE_SLOT_BITS,
    parameter ROUTE_DATA_W   = 1 + CORE_ID_BITS + NEURON_BITS + DATA_WIDTH,
    parameter CLUSTER_SIZE          = 4,
    parameter GLOBAL_ROUTE_SLOTS    = 4,
    parameter GLOBAL_ROUTE_SLOT_BITS = 2,
    parameter GLOBAL_ROUTE_ADDR_W   = CORE_ID_BITS + NEURON_BITS + GLOBAL_ROUTE_SLOT_BITS,
    parameter CHIP_LINK_EN = 0,
    parameter DUAL_NOC = 0,
    parameter MESH_X = 2,
    parameter MESH_Y = 2
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    start,
    input  wire                         prog_pool_we,
    input  wire [CORE_ID_BITS-1:0]      prog_pool_core,
    input  wire [POOL_ADDR_BITS-1:0]    prog_pool_addr,
    input  wire [NEURON_BITS-1:0]       prog_pool_src,
    input  wire [NEURON_BITS-1:0]       prog_pool_target,
    input  wire signed [DATA_WIDTH-1:0] prog_pool_weight,
    input  wire [1:0]                   prog_pool_comp,
    input  wire                         prog_index_we,
    input  wire [CORE_ID_BITS-1:0]      prog_index_core,
    input  wire [NEURON_BITS-1:0]       prog_index_neuron,
    input  wire [POOL_ADDR_BITS-1:0]    prog_index_base,
    input  wire [COUNT_BITS-1:0]        prog_index_count,
    input  wire [1:0]                   prog_index_format,
    input  wire                        prog_route_we,
    input  wire [CORE_ID_BITS-1:0]     prog_route_src_core,
    input  wire [NEURON_BITS-1:0]      prog_route_src_neuron,
    input  wire [ROUTE_SLOT_BITS-1:0]  prog_route_slot,
    input  wire [CORE_ID_BITS-1:0]     prog_route_dest_core,
    input  wire [NEURON_BITS-1:0]      prog_route_dest_neuron,
    input  wire signed [DATA_WIDTH-1:0] prog_route_weight,
    input  wire                        prog_global_route_we,
    input  wire [CORE_ID_BITS-1:0]     prog_global_route_src_core,
    input  wire [NEURON_BITS-1:0]      prog_global_route_src_neuron,
    input  wire [GLOBAL_ROUTE_SLOT_BITS-1:0] prog_global_route_slot,
    input  wire [CORE_ID_BITS-1:0]     prog_global_route_dest_core,
    input  wire [NEURON_BITS-1:0]      prog_global_route_dest_neuron,
    input  wire signed [DATA_WIDTH-1:0] prog_global_route_weight,
    input  wire                        learn_enable,
    input  wire                        graded_enable,
    input  wire                        dendritic_enable,
    input  wire                        async_enable,
    input  wire                        threefactor_enable,
    input  wire                        noise_enable,
    input  wire                        skip_idle_enable,
    input  wire                        scale_u_enable,
    input  wire signed [DATA_WIDTH-1:0] reward_value,
    input  wire                        prog_delay_we,
    input  wire [CORE_ID_BITS-1:0]     prog_delay_core,
    input  wire [POOL_ADDR_BITS-1:0]   prog_delay_addr,
    input  wire [5:0]                  prog_delay_value,
    input  wire                        prog_ucode_we,
    input  wire [CORE_ID_BITS-1:0]     prog_ucode_core,
    input  wire [7:0]                  prog_ucode_addr,
    input  wire [31:0]                 prog_ucode_data,
    input  wire                        prog_param_we,
    input  wire [CORE_ID_BITS-1:0]     prog_param_core,
    input  wire [NEURON_BITS-1:0]      prog_param_neuron,
    input  wire [4:0]                  prog_param_id,
    input  wire signed [DATA_WIDTH-1:0] prog_param_value,
    input  wire                        ext_valid,
    input  wire [CORE_ID_BITS-1:0]     ext_core,
    input  wire [NEURON_BITS-1:0]      ext_neuron_id,
    input  wire signed [DATA_WIDTH-1:0] ext_current,
    input  wire                        probe_read,
    input  wire [CORE_ID_BITS-1:0]     probe_core,
    input  wire [NEURON_BITS-1:0]      probe_neuron,
    input  wire [4:0]                  probe_state_id,
    input  wire [POOL_ADDR_BITS-1:0]   probe_pool_addr,
    output wire signed [DATA_WIDTH-1:0] probe_data,
    output wire                         probe_valid,
    output reg                         timestep_done,
    output wire [NUM_CORES-1:0]        spike_valid_bus,
    output wire [NUM_CORES*NEURON_BITS-1:0] spike_id_bus,
    output wire [5:0]                  mesh_state_out,
    output reg  [31:0]                 total_spikes,
    output reg  [31:0]                 timestep_count,
    output wire [NUM_CORES-1:0]        core_idle_bus,
    output wire                        link_tx_push,
    output wire [CORE_ID_BITS-1:0]     link_tx_core,
    output wire [NEURON_BITS-1:0]      link_tx_neuron,
    output wire [7:0]                  link_tx_payload,
    input  wire                        link_tx_full,
    input  wire [CORE_ID_BITS-1:0]     link_rx_core,
    input  wire [NEURON_BITS-1:0]      link_rx_neuron,
    input  wire signed [DATA_WIDTH-1:0] link_rx_current,
    output wire                        link_rx_pop,
    input  wire                        link_rx_empty
);

    assign link_tx_push = 0;
    assign link_tx_core = 0;
    assign link_tx_neuron = 0;
    assign link_tx_payload = 0;
    assign link_rx_pop = 0;

    localparam COORD_BITS = 4;
    localparam PACKET_W   = 2*COORD_BITS + NEURON_BITS + DATA_WIDTH;

    function [COORD_BITS-1:0] core_to_x;
        input [CORE_ID_BITS-1:0] cid;
        core_to_x = cid % MESH_X;
    endfunction

    function [COORD_BITS-1:0] core_to_y;
        input [CORE_ID_BITS-1:0] cid;
        core_to_y = cid / MESH_X;
    endfunction

    localparam SM_IDLE       = 4'd0;
    localparam SM_PKT_DRAIN  = 4'd1;
    localparam SM_START      = 4'd2;
    localparam SM_RUN_WAIT   = 4'd3;
    localparam SM_ROUTE_POP  = 4'd4;
    localparam SM_ROUTE_ADDR = 4'd5;
    localparam SM_ROUTE_WAIT = 4'd6;
    localparam SM_ROUTE_READ = 4'd7;
    localparam SM_GRT_ADDR   = 4'd8;
    localparam SM_GRT_WAIT   = 4'd9;
    localparam SM_GRT_READ   = 4'd10;
    localparam SM_DONE       = 4'd11;

    reg [3:0] mesh_state;
    assign mesh_state_out = {2'b0, mesh_state};

    reg                      rt_we;
    reg  [ROUTE_ADDR_W-1:0]  rt_addr;
    wire [ROUTE_DATA_W-1:0]  rt_rdata;

    wire                     rt_we_mux   = (mesh_state == SM_IDLE) ? prog_route_we : rt_we;
    wire [ROUTE_ADDR_W-1:0]  rt_addr_mux = (mesh_state == SM_IDLE) ?
        {prog_route_src_core, prog_route_src_neuron, prog_route_slot} : rt_addr;
    wire [ROUTE_DATA_W-1:0]  rt_wdata_mux = (mesh_state == SM_IDLE) ?
        {1'b1, prog_route_dest_core, prog_route_dest_neuron, prog_route_weight} : {ROUTE_DATA_W{1'b0}};

    sram #(.DATA_WIDTH(ROUTE_DATA_W), .ADDR_WIDTH(ROUTE_ADDR_W)) route_table (
        .clk(clk), .we_a(rt_we_mux), .addr_a(rt_addr_mux),
        .wdata_a(rt_wdata_mux), .rdata_a(rt_rdata),
        .addr_b({ROUTE_ADDR_W{1'b0}}), .rdata_b()
    );

    wire                       rt_valid     = rt_rdata[ROUTE_DATA_W-1];
    wire [CORE_ID_BITS-1:0]    rt_dest_core = rt_rdata[NEURON_BITS+DATA_WIDTH +: CORE_ID_BITS];
    wire [NEURON_BITS-1:0]     rt_dest_nrn  = rt_rdata[DATA_WIDTH +: NEURON_BITS];
    wire signed [DATA_WIDTH-1:0] rt_weight  = rt_rdata[DATA_WIDTH-1:0];

    reg                               grt_we;
    reg  [GLOBAL_ROUTE_ADDR_W-1:0]   grt_addr;
    wire [ROUTE_DATA_W-1:0]          grt_rdata;

    wire grt_we_mux = (mesh_state == SM_IDLE) ? prog_global_route_we : grt_we;
    wire [GLOBAL_ROUTE_ADDR_W-1:0] grt_addr_mux = (mesh_state == SM_IDLE) ?
        {prog_global_route_src_core, prog_global_route_src_neuron, prog_global_route_slot} : grt_addr;
    wire [ROUTE_DATA_W-1:0] grt_wdata_mux = (mesh_state == SM_IDLE) ?
        {1'b1, prog_global_route_dest_core, prog_global_route_dest_neuron, prog_global_route_weight} : {ROUTE_DATA_W{1'b0}};

    sram #(.DATA_WIDTH(ROUTE_DATA_W), .ADDR_WIDTH(GLOBAL_ROUTE_ADDR_W)) global_route_table (
        .clk(clk), .we_a(grt_we_mux), .addr_a(grt_addr_mux),
        .wdata_a(grt_wdata_mux), .rdata_a(grt_rdata),
        .addr_b({GLOBAL_ROUTE_ADDR_W{1'b0}}), .rdata_b()
    );

    wire                       grt_valid     = grt_rdata[ROUTE_DATA_W-1];
    wire [CORE_ID_BITS-1:0]    grt_dest_core = grt_rdata[NEURON_BITS+DATA_WIDTH +: CORE_ID_BITS];
    wire [NEURON_BITS-1:0]     grt_dest_nrn  = grt_rdata[DATA_WIDTH +: NEURON_BITS];
    wire signed [DATA_WIDTH-1:0] grt_weight  = grt_rdata[DATA_WIDTH-1:0];

    wire [NUM_CORES-1:0]                core_done;
    wire [NUM_CORES-1:0]                core_spike_valid;
    wire [NUM_CORES*NEURON_BITS-1:0]    core_spike_id;
    wire [NUM_CORES*8-1:0]              core_spike_payload;
    reg  [NUM_CORES-1:0]                core_start_r;

    reg  [NUM_CORES-1:0] core_done_latch;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            core_done_latch <= 0;
        else if (mesh_state == SM_START)
            core_done_latch <= 0;
        else
            core_done_latch <= core_done_latch | core_done;
    end

    assign spike_valid_bus = core_spike_valid;
    assign spike_id_bus    = core_spike_id;

    wire sync_all_done = &core_done_latch;

    localparam CAP_WIDTH = NEURON_BITS + 8;
    reg  [NUM_CORES-1:0] cap_pop;
    reg  [NUM_CORES-1:0] cap_clear;
    wire [NUM_CORES-1:0] cap_empty;
    wire [NUM_CORES*CAP_WIDTH-1:0] cap_data;

    wire [NUM_CORES-1:0] core_probe_valid;
    wire [NUM_CORES*DATA_WIDTH-1:0] core_probe_data;
    assign probe_data  = core_probe_data[probe_core*DATA_WIDTH +: DATA_WIDTH];
    assign probe_valid = core_probe_valid[probe_core];

    function [31:0] popcount;
        input [NUM_CORES-1:0] bits;
        integer k;
    begin
        popcount = 0;
        for (k = 0; k < NUM_CORES; k = k + 1)
            popcount = popcount + bits[k];
    end
    endfunction

    wire [NUM_CORES-1:0] rtr_idle;
    wire [NUM_CORES-1:0] rtr_local_out_valid;
    wire [NUM_CORES*PACKET_W-1:0] rtr_local_out_data;
    wire [NUM_CORES-1:0] rtr_local_in_ready;

    reg  [NUM_CORES-1:0] rtr_local_in_valid;
    reg  [NUM_CORES*PACKET_W-1:0] rtr_local_in_data;

    wire [NUM_CORES-1:0] rtr_local_out_ready =
        (mesh_state == SM_PKT_DRAIN) ? {NUM_CORES{1'b1}} : {NUM_CORES{1'b0}};

    wire [NUM_CORES-1:0] rtr_n_out_v, rtr_s_out_v, rtr_e_out_v, rtr_w_out_v;
    wire [NUM_CORES*PACKET_W-1:0] rtr_n_out_d, rtr_s_out_d, rtr_e_out_d, rtr_w_out_d;
    wire [NUM_CORES-1:0] rtr_n_in_r, rtr_s_in_r, rtr_e_in_r, rtr_w_in_r;

    wire [NUM_CORES-1:0] rtr_b_idle;
    wire [NUM_CORES-1:0] rtr_b_local_out_valid;
    wire [NUM_CORES*PACKET_W-1:0] rtr_b_local_out_data;
    wire [NUM_CORES-1:0] rtr_b_local_in_ready;

    reg  [NUM_CORES-1:0] rtr_b_local_in_valid;
    reg  [NUM_CORES*PACKET_W-1:0] rtr_b_local_in_data;

    wire [NUM_CORES-1:0] rtr_b_local_out_ready =
        (mesh_state == SM_PKT_DRAIN) ? ~rtr_local_out_valid : {NUM_CORES{1'b0}};

    wire [NUM_CORES-1:0] rtr_b_n_out_v, rtr_b_s_out_v, rtr_b_e_out_v, rtr_b_w_out_v;
    wire [NUM_CORES*PACKET_W-1:0] rtr_b_n_out_d, rtr_b_s_out_d, rtr_b_e_out_d, rtr_b_w_out_d;
    wire [NUM_CORES-1:0] rtr_b_n_in_r, rtr_b_s_in_r, rtr_b_e_in_r, rtr_b_w_in_r;

    genvar gi;
    generate
        for (gi = 0; gi < NUM_CORES; gi = gi + 1) begin : gen_core

            wire this_ext_valid =
                (mesh_state == SM_IDLE && ext_valid && ext_core == gi[CORE_ID_BITS-1:0]) ||
                (mesh_state == SM_PKT_DRAIN && (rtr_local_out_valid[gi] || rtr_b_local_out_valid[gi]));

            wire [PACKET_W-1:0] drain_pkt = rtr_local_out_valid[gi] ?
                rtr_local_out_data[gi*PACKET_W +: PACKET_W] :
                rtr_b_local_out_data[gi*PACKET_W +: PACKET_W];
            wire [NEURON_BITS-1:0] this_ext_nid =
                (mesh_state == SM_PKT_DRAIN) ? drain_pkt[DATA_WIDTH +: NEURON_BITS] : ext_neuron_id;
            wire signed [DATA_WIDTH-1:0] this_ext_cur =
                (mesh_state == SM_PKT_DRAIN) ? drain_pkt[DATA_WIDTH-1:0] : ext_current;

            wire this_pool_we = prog_pool_we && (prog_pool_core == gi[CORE_ID_BITS-1:0]) &&
                                (mesh_state == SM_IDLE);
            wire this_index_we = prog_index_we && (prog_index_core == gi[CORE_ID_BITS-1:0]) &&
                                 (mesh_state == SM_IDLE);
            wire this_param_we = prog_param_we && (prog_param_core == gi[CORE_ID_BITS-1:0]) &&
                                 (mesh_state == SM_IDLE);
            wire this_delay_we = prog_delay_we && (prog_delay_core == gi[CORE_ID_BITS-1:0]) &&
                                 (mesh_state == SM_IDLE);
            wire this_ucode_we = prog_ucode_we && (prog_ucode_core == gi[CORE_ID_BITS-1:0]) &&
                                 (mesh_state == SM_IDLE);

            scalable_core_v2 #(
                .NUM_NEURONS(NUM_NEURONS), .NEURON_BITS(NEURON_BITS),
                .DATA_WIDTH(DATA_WIDTH), .POOL_DEPTH(POOL_DEPTH),
                .POOL_ADDR_BITS(POOL_ADDR_BITS), .COUNT_BITS(COUNT_BITS),
                .REV_FANIN(REV_FANIN), .REV_SLOT_BITS(REV_SLOT_BITS),
                .THRESHOLD(THRESHOLD), .LEAK_RATE(LEAK_RATE),
                .REFRAC_CYCLES(REFRAC_CYCLES), .GRADE_SHIFT(GRADE_SHIFT)
            ) core (
                .clk(clk), .rst_n(rst_n),
                .start(core_start_r[gi]),
                .learn_enable(learn_enable), .graded_enable(graded_enable),
                .dendritic_enable(dendritic_enable),
                .threefactor_enable(threefactor_enable),
                .noise_enable(noise_enable), .skip_idle_enable(skip_idle_enable),
                .scale_u_enable(scale_u_enable),
                .reward_value(reward_value),
                .ext_valid(this_ext_valid),
                .ext_neuron_id(this_ext_nid),
                .ext_current(this_ext_cur),
                .pool_we(this_pool_we), .pool_addr_in(prog_pool_addr),
                .pool_src_in(prog_pool_src), .pool_target_in(prog_pool_target),
                .pool_weight_in(prog_pool_weight), .pool_comp_in(prog_pool_comp),
                .index_we(this_index_we), .index_neuron_in(prog_index_neuron),
                .index_base_in(prog_index_base), .index_count_in(prog_index_count),
                .index_format_in(prog_index_format),
                .delay_we(this_delay_we), .delay_addr_in(prog_delay_addr),
                .delay_value_in(prog_delay_value),
                .ucode_prog_we(this_ucode_we), .ucode_prog_addr(prog_ucode_addr),
                .ucode_prog_data(prog_ucode_data),
                .prog_param_we(this_param_we), .prog_param_neuron(prog_param_neuron),
                .prog_param_id(prog_param_id), .prog_param_value(prog_param_value),
                .probe_read(probe_read && (probe_core == gi[CORE_ID_BITS-1:0])),
                .probe_neuron(probe_neuron), .probe_state_id(probe_state_id),
                .probe_pool_addr(probe_pool_addr),
                .probe_data(core_probe_data[gi*DATA_WIDTH +: DATA_WIDTH]),
                .probe_valid(core_probe_valid[gi]),
                .timestep_done(core_done[gi]),
                .spike_out_valid(core_spike_valid[gi]),
                .spike_out_id(core_spike_id[gi*NEURON_BITS +: NEURON_BITS]),
                .spike_out_payload(core_spike_payload[gi*8 +: 8]),
                .state_out(), .total_spikes(), .timestep_count(),
                .core_idle(core_idle_bus[gi])
            );

            spike_fifo #(.ID_WIDTH(CAP_WIDTH), .DEPTH(64), .PTR_BITS(6)) capture_fifo (
                .clk(clk), .rst_n(rst_n), .clear(cap_clear[gi]),
                .push(core_spike_valid[gi] && (mesh_state == SM_RUN_WAIT)),
                .push_data({core_spike_id[gi*NEURON_BITS +: NEURON_BITS],
                            core_spike_payload[gi*8 +: 8]}),
                .pop(cap_pop[gi]),
                .pop_data(cap_data[gi*CAP_WIDTH +: CAP_WIDTH]),
                .empty(cap_empty[gi]), .full(), .count()
            );

            localparam RX = gi % MESH_X;
            localparam RY = gi / MESH_X;
            localparam HAS_N = (RY < MESH_Y - 1) ? 1 : 0;
            localparam HAS_S = (RY > 0) ? 1 : 0;
            localparam HAS_E = (RX < MESH_X - 1) ? 1 : 0;
            localparam HAS_W = (RX > 0) ? 1 : 0;
            localparam N_ID = HAS_N ? ((RY+1)*MESH_X + RX) : 0;
            localparam S_ID = HAS_S ? ((RY-1)*MESH_X + RX) : 0;
            localparam E_ID = HAS_E ? (RY*MESH_X + (RX+1)) : 0;
            localparam W_ID = HAS_W ? (RY*MESH_X + (RX-1)) : 0;

            wire n_in_v = HAS_N ? rtr_s_out_v[N_ID] : 1'b0;
            wire [PACKET_W-1:0] n_in_d = HAS_N ? rtr_s_out_d[N_ID*PACKET_W +: PACKET_W] : {PACKET_W{1'b0}};
            wire n_out_r = HAS_N ? rtr_s_in_r[N_ID] : 1'b1;

            wire s_in_v = HAS_S ? rtr_n_out_v[S_ID] : 1'b0;
            wire [PACKET_W-1:0] s_in_d = HAS_S ? rtr_n_out_d[S_ID*PACKET_W +: PACKET_W] : {PACKET_W{1'b0}};
            wire s_out_r = HAS_S ? rtr_n_in_r[S_ID] : 1'b1;

            wire e_in_v = HAS_E ? rtr_w_out_v[E_ID] : 1'b0;
            wire [PACKET_W-1:0] e_in_d = HAS_E ? rtr_w_out_d[E_ID*PACKET_W +: PACKET_W] : {PACKET_W{1'b0}};
            wire e_out_r = HAS_E ? rtr_w_in_r[E_ID] : 1'b1;

            wire w_in_v = HAS_W ? rtr_e_out_v[W_ID] : 1'b0;
            wire [PACKET_W-1:0] w_in_d = HAS_W ? rtr_e_out_d[W_ID*PACKET_W +: PACKET_W] : {PACKET_W{1'b0}};
            wire w_out_r = HAS_W ? rtr_e_in_r[W_ID] : 1'b1;

            async_router #(
                .PACKET_W(PACKET_W), .COORD_BITS(COORD_BITS),
                .FIFO_DEPTH(16), .FIFO_PTR_BITS(4)
            ) router (
                .clk(clk), .rst_n(rst_n),
                .my_x(core_to_x(gi[CORE_ID_BITS-1:0])),
                .my_y(core_to_y(gi[CORE_ID_BITS-1:0])),
                .local_in_valid (rtr_local_in_valid[gi]),
                .local_in_ready (rtr_local_in_ready[gi]),
                .local_in_data  (rtr_local_in_data[gi*PACKET_W +: PACKET_W]),
                .local_out_valid(rtr_local_out_valid[gi]),
                .local_out_ready(rtr_local_out_ready[gi]),
                .local_out_data (rtr_local_out_data[gi*PACKET_W +: PACKET_W]),
                .north_in_valid (n_in_v),
                .north_in_ready (rtr_n_in_r[gi]),
                .north_in_data  (n_in_d),
                .north_out_valid(rtr_n_out_v[gi]),
                .north_out_ready(n_out_r),
                .north_out_data (rtr_n_out_d[gi*PACKET_W +: PACKET_W]),
                .south_in_valid (s_in_v),
                .south_in_ready (rtr_s_in_r[gi]),
                .south_in_data  (s_in_d),
                .south_out_valid(rtr_s_out_v[gi]),
                .south_out_ready(s_out_r),
                .south_out_data (rtr_s_out_d[gi*PACKET_W +: PACKET_W]),
                .east_in_valid  (e_in_v),
                .east_in_ready  (rtr_e_in_r[gi]),
                .east_in_data   (e_in_d),
                .east_out_valid (rtr_e_out_v[gi]),
                .east_out_ready (e_out_r),
                .east_out_data  (rtr_e_out_d[gi*PACKET_W +: PACKET_W]),
                .west_in_valid  (w_in_v),
                .west_in_ready  (rtr_w_in_r[gi]),
                .west_in_data   (w_in_d),
                .west_out_valid (rtr_w_out_v[gi]),
                .west_out_ready (w_out_r),
                .west_out_data  (rtr_w_out_d[gi*PACKET_W +: PACKET_W]),
                .idle           (rtr_idle[gi])
            );
        end
    endgenerate

    generate if (DUAL_NOC) begin : gen_net_b
        genvar bi;
        for (bi = 0; bi < NUM_CORES; bi = bi + 1) begin : gen_rtr_b
            localparam BRX = bi % MESH_X;
            localparam BRY = bi / MESH_X;
            localparam B_HAS_N = (BRY < MESH_Y - 1) ? 1 : 0;
            localparam B_HAS_S = (BRY > 0) ? 1 : 0;
            localparam B_HAS_E = (BRX < MESH_X - 1) ? 1 : 0;
            localparam B_HAS_W = (BRX > 0) ? 1 : 0;
            localparam BN_ID = B_HAS_N ? ((BRY+1)*MESH_X + BRX) : 0;
            localparam BS_ID = B_HAS_S ? ((BRY-1)*MESH_X + BRX) : 0;
            localparam BE_ID = B_HAS_E ? (BRY*MESH_X + (BRX+1)) : 0;
            localparam BW_ID = B_HAS_W ? (BRY*MESH_X + (BRX-1)) : 0;

            wire bn_in_v = B_HAS_N ? rtr_b_s_out_v[BN_ID] : 1'b0;
            wire [PACKET_W-1:0] bn_in_d = B_HAS_N ?
                rtr_b_s_out_d[BN_ID*PACKET_W +: PACKET_W] : {PACKET_W{1'b0}};
            wire bn_out_r = B_HAS_N ? rtr_b_s_in_r[BN_ID] : 1'b1;

            wire bs_in_v = B_HAS_S ? rtr_b_n_out_v[BS_ID] : 1'b0;
            wire [PACKET_W-1:0] bs_in_d = B_HAS_S ?
                rtr_b_n_out_d[BS_ID*PACKET_W +: PACKET_W] : {PACKET_W{1'b0}};
            wire bs_out_r = B_HAS_S ? rtr_b_n_in_r[BS_ID] : 1'b1;

            wire be_in_v = B_HAS_E ? rtr_b_w_out_v[BE_ID] : 1'b0;
            wire [PACKET_W-1:0] be_in_d = B_HAS_E ?
                rtr_b_w_out_d[BE_ID*PACKET_W +: PACKET_W] : {PACKET_W{1'b0}};
            wire be_out_r = B_HAS_E ? rtr_b_w_in_r[BE_ID] : 1'b1;

            wire bw_in_v = B_HAS_W ? rtr_b_e_out_v[BW_ID] : 1'b0;
            wire [PACKET_W-1:0] bw_in_d = B_HAS_W ?
                rtr_b_e_out_d[BW_ID*PACKET_W +: PACKET_W] : {PACKET_W{1'b0}};
            wire bw_out_r = B_HAS_W ? rtr_b_e_in_r[BW_ID] : 1'b1;

            async_router #(
                .PACKET_W(PACKET_W), .COORD_BITS(COORD_BITS),
                .FIFO_DEPTH(16), .FIFO_PTR_BITS(4)
            ) router_b (
                .clk(clk), .rst_n(rst_n),
                .my_x(core_to_x(bi[CORE_ID_BITS-1:0])),
                .my_y(core_to_y(bi[CORE_ID_BITS-1:0])),
                .local_in_valid (rtr_b_local_in_valid[bi]),
                .local_in_ready (rtr_b_local_in_ready[bi]),
                .local_in_data  (rtr_b_local_in_data[bi*PACKET_W +: PACKET_W]),
                .local_out_valid(rtr_b_local_out_valid[bi]),
                .local_out_ready(rtr_b_local_out_ready[bi]),
                .local_out_data (rtr_b_local_out_data[bi*PACKET_W +: PACKET_W]),
                .north_in_valid (bn_in_v),
                .north_in_ready (rtr_b_n_in_r[bi]),
                .north_in_data  (bn_in_d),
                .north_out_valid(rtr_b_n_out_v[bi]),
                .north_out_ready(bn_out_r),
                .north_out_data (rtr_b_n_out_d[bi*PACKET_W +: PACKET_W]),
                .south_in_valid (bs_in_v),
                .south_in_ready (rtr_b_s_in_r[bi]),
                .south_in_data  (bs_in_d),
                .south_out_valid(rtr_b_s_out_v[bi]),
                .south_out_ready(bs_out_r),
                .south_out_data (rtr_b_s_out_d[bi*PACKET_W +: PACKET_W]),
                .east_in_valid  (be_in_v),
                .east_in_ready  (rtr_b_e_in_r[bi]),
                .east_in_data   (be_in_d),
                .east_out_valid (rtr_b_e_out_v[bi]),
                .east_out_ready (be_out_r),
                .east_out_data  (rtr_b_e_out_d[bi*PACKET_W +: PACKET_W]),
                .west_in_valid  (bw_in_v),
                .west_in_ready  (rtr_b_w_in_r[bi]),
                .west_in_data   (bw_in_d),
                .west_out_valid (rtr_b_w_out_v[bi]),
                .west_out_ready (bw_out_r),
                .west_out_data  (rtr_b_w_out_d[bi*PACKET_W +: PACKET_W]),
                .idle           (rtr_b_idle[bi])
            );
        end
    end else begin : gen_no_net_b
        assign rtr_b_idle = {NUM_CORES{1'b1}};
        assign rtr_b_local_out_valid = {NUM_CORES{1'b0}};
        assign rtr_b_local_out_data = {NUM_CORES*PACKET_W{1'b0}};
        assign rtr_b_local_in_ready = {NUM_CORES{1'b1}};
    end endgenerate

    reg [CORE_ID_BITS-1:0]     route_core_idx;
    reg [NEURON_BITS-1:0]      route_neuron;
    reg [7:0]                  route_payload;
    reg [ROUTE_SLOT_BITS-1:0]  route_slot;
    reg [GLOBAL_ROUTE_SLOT_BITS-1:0] global_slot;
    reg [3:0]                  drain_wait;

    wire signed [31:0] route_weight_ext = rt_weight;
    wire signed [31:0] route_payload_ext = {24'd0, route_payload};
    wire signed [31:0] route_graded_product = route_weight_ext * route_payload_ext;
    wire signed [DATA_WIDTH-1:0] route_graded_current = route_graded_product >>> GRADE_SHIFT;

    wire signed [31:0] grt_weight_ext = grt_weight;
    wire signed [31:0] grt_graded_product = grt_weight_ext * route_payload_ext;
    wire signed [DATA_WIDTH-1:0] grt_graded_current = grt_graded_product >>> GRADE_SHIFT;

    wire signed [DATA_WIDTH-1:0] rt_eff_weight = graded_enable ? route_graded_current : rt_weight;
    wire signed [DATA_WIDTH-1:0] grt_eff_weight = graded_enable ? grt_graded_current : grt_weight;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mesh_state     <= SM_IDLE;
            timestep_done  <= 0;
            total_spikes   <= 0;
            timestep_count <= 0;
            core_start_r   <= 0;
            route_core_idx <= 0;
            route_neuron   <= 0;
            route_payload  <= 0;
            route_slot     <= 0;
            global_slot    <= 0;
            drain_wait     <= 0;
            rt_we          <= 0;
            rt_addr        <= 0;
            grt_we         <= 0;
            grt_addr       <= 0;
            cap_pop        <= 0;
            cap_clear      <= 0;
            rtr_local_in_valid <= 0;
            rtr_local_in_data  <= 0;
            rtr_b_local_in_valid <= 0;
            rtr_b_local_in_data  <= 0;
        end else begin
            timestep_done      <= 0;
            core_start_r       <= 0;
            rt_we              <= 0;
            grt_we             <= 0;
            cap_pop            <= 0;
            cap_clear          <= 0;
            rtr_local_in_valid <= 0;
            rtr_b_local_in_valid <= 0;

            total_spikes <= total_spikes + popcount(core_spike_valid);

            case (mesh_state)
                SM_IDLE: begin
                    if (start) begin
                        drain_wait <= 0;
                        mesh_state <= SM_PKT_DRAIN;
                    end
                end

                SM_PKT_DRAIN: begin
                    if ((&rtr_idle) && (&rtr_b_idle) && !(|rtr_local_out_valid) && !(|rtr_b_local_out_valid)) begin
                        drain_wait <= drain_wait + 1;
                        if (drain_wait >= 4'd3)
                            mesh_state <= SM_START;
                    end else begin
                        drain_wait <= 0;
                    end
                end

                SM_START: begin
                    core_start_r <= {NUM_CORES{1'b1}};
                    mesh_state   <= SM_RUN_WAIT;
                end

                SM_RUN_WAIT: begin
                    if (sync_all_done) begin
                        route_core_idx <= 0;
                        mesh_state     <= SM_ROUTE_POP;
                    end
                end

                SM_ROUTE_POP: begin
                    if (cap_empty[route_core_idx]) begin
                        if (route_core_idx == NUM_CORES - 1)
                            mesh_state <= SM_DONE;
                        else
                            route_core_idx <= route_core_idx + 1;
                    end else begin
                        cap_pop[route_core_idx] <= 1;
                        route_neuron  <= cap_data[route_core_idx * CAP_WIDTH + 8 +: NEURON_BITS];
                        route_payload <= cap_data[route_core_idx * CAP_WIDTH +: 8];
                        route_slot    <= 0;
                        mesh_state    <= SM_ROUTE_ADDR;
                    end
                end

                SM_ROUTE_ADDR: begin
                    rt_addr    <= {route_core_idx, route_neuron, route_slot};
                    mesh_state <= SM_ROUTE_WAIT;
                end

                SM_ROUTE_WAIT: begin
                    mesh_state <= SM_ROUTE_READ;
                end

                SM_ROUTE_READ: begin
                    if (rt_valid) begin
                        if (route_core_idx[0] == 1'b0 || !DUAL_NOC) begin
                            if (rtr_local_in_ready[route_core_idx]) begin
                                rtr_local_in_valid[route_core_idx] <= 1;
                                rtr_local_in_data[route_core_idx*PACKET_W +: PACKET_W] <=
                                    {core_to_x(rt_dest_core), core_to_y(rt_dest_core),
                                     rt_dest_nrn, rt_eff_weight};
                            end
                        end else begin
                            if (rtr_b_local_in_ready[route_core_idx]) begin
                                rtr_b_local_in_valid[route_core_idx] <= 1;
                                rtr_b_local_in_data[route_core_idx*PACKET_W +: PACKET_W] <=
                                    {core_to_x(rt_dest_core), core_to_y(rt_dest_core),
                                     rt_dest_nrn, rt_eff_weight};
                            end
                        end
                    end
                    if (route_slot < ROUTE_FANOUT - 1) begin
                        route_slot <= route_slot + 1;
                        mesh_state <= SM_ROUTE_ADDR;
                    end else begin
                        global_slot <= 0;
                        mesh_state  <= SM_GRT_ADDR;
                    end
                end

                SM_GRT_ADDR: begin
                    grt_addr   <= {route_core_idx, route_neuron, global_slot};
                    mesh_state <= SM_GRT_WAIT;
                end

                SM_GRT_WAIT: begin
                    mesh_state <= SM_GRT_READ;
                end

                SM_GRT_READ: begin
                    if (grt_valid) begin
                        if (route_core_idx[0] == 1'b0 || !DUAL_NOC) begin
                            if (rtr_local_in_ready[route_core_idx]) begin
                                rtr_local_in_valid[route_core_idx] <= 1;
                                rtr_local_in_data[route_core_idx*PACKET_W +: PACKET_W] <=
                                    {core_to_x(grt_dest_core), core_to_y(grt_dest_core),
                                     grt_dest_nrn, grt_eff_weight};
                            end
                        end else begin
                            if (rtr_b_local_in_ready[route_core_idx]) begin
                                rtr_b_local_in_valid[route_core_idx] <= 1;
                                rtr_b_local_in_data[route_core_idx*PACKET_W +: PACKET_W] <=
                                    {core_to_x(grt_dest_core), core_to_y(grt_dest_core),
                                     grt_dest_nrn, grt_eff_weight};
                            end
                        end
                    end
                    if (global_slot < GLOBAL_ROUTE_SLOTS - 1) begin
                        global_slot <= global_slot + 1;
                        mesh_state  <= SM_GRT_ADDR;
                    end else begin
                        mesh_state <= SM_ROUTE_POP;
                    end
                end

                SM_DONE: begin
                    cap_clear      <= {NUM_CORES{1'b1}};
                    timestep_done  <= 1;
                    timestep_count <= timestep_count + 1;
                    mesh_state     <= SM_IDLE;
                end

                default: mesh_state <= SM_IDLE;
            endcase
        end
    end

endmodule
