module neuromorphic_mesh #(
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

    parameter CHIP_LINK_EN = 0
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
    input  wire signed [DATA_WIDTH-1:0] reward_value,

    input  wire                        noise_enable,

    input  wire                        skip_idle_enable,

    input  wire                        scale_u_enable,

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
    output reg  signed [DATA_WIDTH-1:0] probe_data,
    output reg                          probe_valid,

    output reg                         timestep_done,
    output wire [NUM_CORES-1:0]        spike_valid_bus,
    output wire [NUM_CORES*NEURON_BITS-1:0] spike_id_bus,
    output wire [5:0]                  mesh_state_out,
    output reg  [31:0]                 total_spikes,
    output reg  [31:0]                 timestep_count,

    output wire [NUM_CORES-1:0]        core_idle_bus,

    input  wire [7:0]                  dvfs_stall,

    output wire [NUM_CORES-1:0]        core_clock_en,
    output reg  [31:0]                 energy_counter,
    output wire                        power_idle_hint,

    output reg                         link_tx_push,
    output reg  [CORE_ID_BITS-1:0]     link_tx_core,
    output reg  [NEURON_BITS-1:0]      link_tx_neuron,
    output reg  [7:0]                  link_tx_payload,
    input  wire                        link_tx_full,
    input  wire [CORE_ID_BITS-1:0]     link_rx_core,
    input  wire [NEURON_BITS-1:0]      link_rx_neuron,
    input  wire signed [DATA_WIDTH-1:0] link_rx_current,
    output reg                         link_rx_pop,
    input  wire                        link_rx_empty
);

    localparam SM_IDLE       = 6'd0;
    localparam SM_INJECT     = 6'd1;
    localparam SM_START      = 6'd2;
    localparam SM_RUN_WAIT   = 6'd3;
    localparam SM_ROUTE_POP  = 6'd4;
    localparam SM_ROUTE_ADDR = 6'd5;
    localparam SM_ROUTE_WAIT = 6'd6;
    localparam SM_ROUTE_READ = 6'd7;
    localparam SM_DONE       = 6'd8;

    localparam SM_ASYNC_ACTIVE     = 6'd9;
    localparam SM_ASYNC_INJECT     = 6'd10;
    localparam SM_ASYNC_ROUTE_POP  = 6'd11;
    localparam SM_ASYNC_ROUTE_ADDR = 6'd12;
    localparam SM_ASYNC_ROUTE_WAIT = 6'd13;
    localparam SM_ASYNC_ROUTE_READ = 6'd14;
    localparam SM_ASYNC_DONE       = 6'd15;

    localparam SM_GLOBAL_ROUTE_ADDR = 6'd16;
    localparam SM_GLOBAL_ROUTE_WAIT = 6'd17;
    localparam SM_GLOBAL_ROUTE_READ = 6'd18;

    localparam SM_LINK_RX_DRAIN = 6'd19;
    localparam SM_LINK_RX_WAIT  = 6'd20;

    localparam SM_DVFS_WAIT     = 6'd21;

    reg [5:0] mesh_state;
    assign mesh_state_out = mesh_state;
    reg [7:0] dvfs_wait_cnt;

    reg                      rt_we;
    reg  [ROUTE_ADDR_W-1:0]  rt_addr;
    reg  [ROUTE_DATA_W-1:0]  rt_wdata;
    wire [ROUTE_DATA_W-1:0]  rt_rdata;

    wire                     rt_we_mux   = (mesh_state == SM_IDLE) ? prog_route_we : rt_we;
    wire [ROUTE_ADDR_W-1:0]  rt_addr_mux = (mesh_state == SM_IDLE) ?
        {prog_route_src_core, prog_route_src_neuron, prog_route_slot} : rt_addr;
    wire [ROUTE_DATA_W-1:0]  rt_wdata_mux = (mesh_state == SM_IDLE) ?
        {1'b1, prog_route_dest_core, prog_route_dest_neuron, prog_route_weight} : rt_wdata;

    sram #(.DATA_WIDTH(ROUTE_DATA_W), .ADDR_WIDTH(ROUTE_ADDR_W)) route_table (
        .clk(clk),
        .we_a(rt_we_mux), .addr_a(rt_addr_mux),
        .wdata_a(rt_wdata_mux), .rdata_a(rt_rdata),
        .addr_b({ROUTE_ADDR_W{1'b0}}), .rdata_b()
    );

    wire                       rt_valid      = rt_rdata[ROUTE_DATA_W-1];
    localparam RT_DEST_CORE_LO = NEURON_BITS + DATA_WIDTH;
    localparam RT_DEST_CORE_HI = NEURON_BITS + DATA_WIDTH + CORE_ID_BITS - 1;
    wire [CORE_ID_BITS-1:0]    rt_dest_core  = rt_rdata[RT_DEST_CORE_HI:RT_DEST_CORE_LO];
    localparam RT_DEST_NRN_LO = DATA_WIDTH;
    localparam RT_DEST_NRN_HI = DATA_WIDTH + NEURON_BITS - 1;
    wire [NEURON_BITS-1:0]     rt_dest_nrn   = rt_rdata[RT_DEST_NRN_HI:RT_DEST_NRN_LO];
    wire signed [DATA_WIDTH-1:0] rt_weight   = rt_rdata[DATA_WIDTH-1:0];

    reg                               grt_we;
    reg  [GLOBAL_ROUTE_ADDR_W-1:0]   grt_addr;
    wire [ROUTE_DATA_W-1:0]          grt_rdata;

    wire                              grt_we_mux   = (mesh_state == SM_IDLE) ? prog_global_route_we : grt_we;
    wire [GLOBAL_ROUTE_ADDR_W-1:0]   grt_addr_mux = (mesh_state == SM_IDLE) ?
        {prog_global_route_src_core, prog_global_route_src_neuron, prog_global_route_slot} : grt_addr;
    wire [ROUTE_DATA_W-1:0]          grt_wdata_mux = (mesh_state == SM_IDLE) ?
        {1'b1, prog_global_route_dest_core, prog_global_route_dest_neuron, prog_global_route_weight} : {ROUTE_DATA_W{1'b0}};

    sram #(.DATA_WIDTH(ROUTE_DATA_W), .ADDR_WIDTH(GLOBAL_ROUTE_ADDR_W)) global_route_table (
        .clk(clk),
        .we_a(grt_we_mux), .addr_a(grt_addr_mux),
        .wdata_a(grt_wdata_mux), .rdata_a(grt_rdata),
        .addr_b({GLOBAL_ROUTE_ADDR_W{1'b0}}), .rdata_b()
    );

    wire                       grt_valid      = grt_rdata[ROUTE_DATA_W-1];
    localparam GRT_DEST_CORE_LO = NEURON_BITS + DATA_WIDTH;
    localparam GRT_DEST_CORE_HI = NEURON_BITS + DATA_WIDTH + CORE_ID_BITS - 1;
    wire [CORE_ID_BITS-1:0]    grt_dest_core  = grt_rdata[GRT_DEST_CORE_HI:GRT_DEST_CORE_LO];
    localparam GRT_DEST_NRN_LO = DATA_WIDTH;
    localparam GRT_DEST_NRN_HI = DATA_WIDTH + NEURON_BITS - 1;
    wire [NEURON_BITS-1:0]     grt_dest_nrn   = grt_rdata[GRT_DEST_NRN_HI:GRT_DEST_NRN_LO];
    wire signed [DATA_WIDTH-1:0] grt_weight   = grt_rdata[DATA_WIDTH-1:0];

    wire signed [31:0] grt_weight_ext      = grt_weight;
    wire signed [31:0] grt_graded_product  = grt_weight_ext * route_payload_ext;
    wire signed [DATA_WIDTH-1:0] grt_graded_current = grt_graded_product >>> GRADE_SHIFT;

    localparam INJECT_WIDTH = CORE_ID_BITS + NEURON_BITS + DATA_WIDTH;

    reg                        inj_push, inj_pop, inj_clear;
    reg  [INJECT_WIDTH-1:0]    inj_push_data;
    wire [INJECT_WIDTH-1:0]    inj_pop_data;
    wire                       inj_empty, inj_full;

    spike_fifo #(.ID_WIDTH(INJECT_WIDTH), .DEPTH(512), .PTR_BITS(9)) inject_fifo (
        .clk(clk), .rst_n(rst_n), .clear(inj_clear),
        .push(inj_push), .push_data(inj_push_data),
        .pop(inj_pop), .pop_data(inj_pop_data),
        .empty(inj_empty), .full(inj_full), .count()
    );

    localparam INJ_DEST_CORE_HI = INJECT_WIDTH - 1;
    localparam INJ_DEST_CORE_LO = INJECT_WIDTH - CORE_ID_BITS;
    wire [CORE_ID_BITS-1:0]      inj_dest_core = inj_pop_data[INJ_DEST_CORE_HI:INJ_DEST_CORE_LO];
    localparam INJ_DEST_NRN_LO = DATA_WIDTH;
    localparam INJ_DEST_NRN_HI = DATA_WIDTH + NEURON_BITS - 1;
    wire [NEURON_BITS-1:0]       inj_dest_nrn  = inj_pop_data[INJ_DEST_NRN_HI:INJ_DEST_NRN_LO];
    wire signed [DATA_WIDTH-1:0] inj_weight    = inj_pop_data[DATA_WIDTH-1:0];

    wire [NUM_CORES-1:0]                    core_done;
    wire [NUM_CORES-1:0]                    core_spike_valid;
    wire [NUM_CORES*NEURON_BITS-1:0]        core_spike_id;
    wire [NUM_CORES*8-1:0]                  core_spike_payload;

    reg  [NUM_CORES-1:0]                    core_start_r;

    reg  [NUM_CORES-1:0] core_done_latch;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            core_done_latch <= 0;
        else if (mesh_state == SM_START)
            core_done_latch <= 0;
        else
            core_done_latch <= core_done_latch | core_done;
    end

    reg [NUM_CORES-1:0] core_running;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            core_running <= 0;
        else
            core_running <= (core_running | core_start_r) & ~core_done;
    end

    reg [NUM_CORES-1:0] core_produced_spike;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            core_produced_spike <= 0;
        else
            core_produced_spike <= (core_produced_spike & ~core_start_r)
                                   | (core_spike_valid & core_running);
    end

    reg [NUM_CORES-1:0] core_needs_restart;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            core_needs_restart <= 0;
        else if (mesh_state == SM_ASYNC_DONE)
            core_needs_restart <= 0;
        else
            core_needs_restart <= (core_needs_restart
                                   | (core_done & (core_produced_spike | core_spike_valid)))
                                  & ~core_start_r;
    end

    assign spike_valid_bus = core_spike_valid;
    assign spike_id_bus    = core_spike_id;

    localparam PCF_WIDTH = NEURON_BITS + DATA_WIDTH;

    reg  [NUM_CORES-1:0]           pcif_push;
    reg  [NUM_CORES-1:0]           pcif_pop;
    reg  [NUM_CORES-1:0]           pcif_clear;
    reg  [PCF_WIDTH-1:0]           pcif_push_data;
    wire [NUM_CORES-1:0]           pcif_empty;
    wire [NUM_CORES-1:0]           pcif_full;
    wire [NUM_CORES*PCF_WIDTH-1:0] pcif_data;

    reg [CORE_ID_BITS-1:0] inject_core_idx;

    reg [PCF_WIDTH-1:0] active_pcif_entry;
    always @(*) begin
        active_pcif_entry = pcif_data >> (inject_core_idx * PCF_WIDTH);
    end
    localparam PCIF_NID_LO = DATA_WIDTH;
    localparam PCIF_NID_HI = DATA_WIDTH + NEURON_BITS - 1;
    wire [NEURON_BITS-1:0]         pcif_nid = active_pcif_entry[PCIF_NID_HI:PCIF_NID_LO];
    wire signed [DATA_WIDTH-1:0]   pcif_cur = active_pcif_entry[DATA_WIDTH-1:0];

    wire [NEURON_BITS-1:0] mesh_ext_nid =
        (mesh_state == SM_INJECT)       ? inj_dest_nrn :
        (mesh_state == SM_ASYNC_INJECT) ? pcif_nid :
                                          ext_neuron_id;

    wire signed [DATA_WIDTH-1:0] mesh_ext_cur =
        (mesh_state == SM_INJECT)       ? inj_weight :
        (mesh_state == SM_ASYNC_INJECT) ? pcif_cur :
                                          ext_current;

    localparam CAP_WIDTH = NEURON_BITS + 8;

    reg  [NUM_CORES-1:0] cap_pop;
    reg  [NUM_CORES-1:0] cap_clear;
    wire [NUM_CORES-1:0] cap_empty;
    wire [NUM_CORES*CAP_WIDTH-1:0] cap_data;

    wire [NUM_CORES-1:0] core_probe_valid;
    wire [NUM_CORES*DATA_WIDTH-1:0] core_probe_data;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            probe_data  <= {DATA_WIDTH{1'b0}};
            probe_valid <= 1'b0;
        end else begin
            probe_data  <= core_probe_data >> (probe_core * DATA_WIDTH);
            probe_valid <= core_probe_valid[probe_core];
        end
    end

    genvar gi;
    generate
        for (gi = 0; gi < NUM_CORES; gi = gi + 1) begin : gen_core

            localparam [CORE_ID_BITS-1:0] GI_CORE_ID = gi;

            wire this_ext_valid =
                (mesh_state == SM_IDLE && ext_valid && ext_core == GI_CORE_ID && !async_enable) ||
                (mesh_state == SM_INJECT && !inj_empty && inj_dest_core == GI_CORE_ID) ||
                (mesh_state == SM_ASYNC_INJECT && inject_core_idx == GI_CORE_ID && !pcif_empty[gi]);

            wire this_pool_we = prog_pool_we && (prog_pool_core == GI_CORE_ID) &&
                                (mesh_state == SM_IDLE);

            wire this_index_we = prog_index_we && (prog_index_core == GI_CORE_ID) &&
                                 (mesh_state == SM_IDLE);

            wire this_param_we = prog_param_we && (prog_param_core == GI_CORE_ID) &&
                                 (mesh_state == SM_IDLE);

            wire this_delay_we = prog_delay_we && (prog_delay_core == GI_CORE_ID) &&
                                 (mesh_state == SM_IDLE);

            wire this_ucode_we = prog_ucode_we && (prog_ucode_core == GI_CORE_ID) &&
                                 (mesh_state == SM_IDLE);

            scalable_core_v2 #(
                .NUM_NEURONS   (NUM_NEURONS),
                .NEURON_BITS   (NEURON_BITS),
                .DATA_WIDTH    (DATA_WIDTH),
                .POOL_DEPTH    (POOL_DEPTH),
                .POOL_ADDR_BITS(POOL_ADDR_BITS),
                .COUNT_BITS    (COUNT_BITS),
                .REV_FANIN     (REV_FANIN),
                .REV_SLOT_BITS (REV_SLOT_BITS),
                .THRESHOLD     (THRESHOLD),
                .LEAK_RATE     (LEAK_RATE),
                .REFRAC_CYCLES (REFRAC_CYCLES),
                .TRACE_MAX     (8'd100),
                .TRACE_DECAY   (8'd3),
                .LEARN_SHIFT   (3),
                .GRADE_SHIFT   (GRADE_SHIFT)
            ) core (
                .clk            (clk),
                .rst_n          (rst_n),
                .start          (core_start_r[gi]),
                .learn_enable   (learn_enable),
                .graded_enable  (graded_enable),
                .dendritic_enable(dendritic_enable),
                .threefactor_enable(threefactor_enable),
                .noise_enable   (noise_enable),
                .skip_idle_enable(skip_idle_enable),
                .scale_u_enable (scale_u_enable),
                .reward_value   (reward_value),
                .ext_valid      (this_ext_valid),
                .ext_neuron_id  (mesh_ext_nid),
                .ext_current    (mesh_ext_cur),
                .pool_we        (this_pool_we),
                .pool_addr_in   (prog_pool_addr),
                .pool_src_in    (prog_pool_src),
                .pool_target_in (prog_pool_target),
                .pool_weight_in (prog_pool_weight),
                .pool_comp_in   (prog_pool_comp),
                .index_we       (this_index_we),
                .index_neuron_in(prog_index_neuron),
                .index_base_in  (prog_index_base),
                .index_count_in (prog_index_count),
                .index_format_in(prog_index_format),
                .delay_we        (this_delay_we),
                .delay_addr_in   (prog_delay_addr),
                .delay_value_in  (prog_delay_value),
                .ucode_prog_we   (this_ucode_we),
                .ucode_prog_addr (prog_ucode_addr),
                .ucode_prog_data (prog_ucode_data),
                .prog_param_we    (this_param_we),
                .prog_param_neuron(prog_param_neuron),
                .prog_param_id    (prog_param_id),
                .prog_param_value (prog_param_value),

                .probe_read     (probe_read && (probe_core == GI_CORE_ID)),
                .probe_neuron   (probe_neuron),
                .probe_state_id (probe_state_id),
                .probe_pool_addr(probe_pool_addr),
                .probe_data     (core_probe_data[gi*DATA_WIDTH +: DATA_WIDTH]),
                .probe_valid    (core_probe_valid[gi]),
                .timestep_done  (core_done[gi]),
                .spike_out_valid(core_spike_valid[gi]),
                .spike_out_id   (core_spike_id[gi*NEURON_BITS +: NEURON_BITS]),
                .spike_out_payload(core_spike_payload[gi*8 +: 8]),
                .state_out      (),
                .total_spikes   (),
                .timestep_count (),
                .core_idle      (core_idle_bus[gi])
            );

            spike_fifo #(.ID_WIDTH(CAP_WIDTH), .DEPTH(64), .PTR_BITS(6)) capture_fifo (
                .clk(clk), .rst_n(rst_n),
                .clear(cap_clear[gi]),
                .push(core_spike_valid[gi] && (mesh_state == SM_RUN_WAIT || core_running[gi])),
                .push_data({core_spike_id[gi*NEURON_BITS +: NEURON_BITS],
                            core_spike_payload[gi*8 +: 8]}),
                .pop(cap_pop[gi]),
                .pop_data(cap_data[gi*CAP_WIDTH +: CAP_WIDTH]),
                .empty(cap_empty[gi]),
                .full(), .count()
            );

            spike_fifo #(.ID_WIDTH(PCF_WIDTH), .DEPTH(8), .PTR_BITS(3)) pcif (
                .clk(clk), .rst_n(rst_n),
                .clear(pcif_clear[gi]),
                .push(pcif_push[gi]),
                .push_data(pcif_push_data),
                .pop(pcif_pop[gi]),
                .pop_data(pcif_data[gi*PCF_WIDTH +: PCF_WIDTH]),
                .empty(pcif_empty[gi]),
                .full(pcif_full[gi]),
                .count()
            );
        end
    endgenerate

    wire mesh_active = (mesh_state != SM_IDLE && mesh_state != SM_DVFS_WAIT);
    assign core_clock_en = mesh_active ? {NUM_CORES{1'b1}} : ~core_idle_bus;
    assign power_idle_hint = (mesh_state == SM_IDLE) && (&core_idle_bus);

    reg [7:0] e_spike_coeff;
    reg [7:0] e_synop_coeff;
    reg [7:0] e_cycle_coeff;
    wire [31:0] total_spike_count_this_ts = popcount(core_spike_valid_sync);
    reg [NUM_CORES-1:0] core_spike_valid_sync;
    always @(posedge clk) core_spike_valid_sync <= {NUM_CORES{1'b0}};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            energy_counter  <= 32'd0;
            e_spike_coeff   <= 8'd10;
            e_synop_coeff   <= 8'd1;
            e_cycle_coeff   <= 8'd1;
        end else begin
            if (mesh_active)
                energy_counter <= energy_counter + {24'd0, e_cycle_coeff};
            if (mesh_state == SM_DONE)
                energy_counter <= energy_counter + total_spikes * {24'd0, e_spike_coeff};
        end
    end

    function [31:0] popcount;
        input [NUM_CORES-1:0] bits;
        integer k;
    begin
        popcount = 0;
        for (k = 0; k < NUM_CORES; k = k + 1)
            popcount = popcount + bits[k];
    end
    endfunction

    reg                      first_inject_found;
    reg  [CORE_ID_BITS-1:0]  first_inject_core;
    integer pe_i;
    always @(*) begin
        first_inject_found = 0;
        first_inject_core  = 0;
        for (pe_i = 0; pe_i < NUM_CORES; pe_i = pe_i + 1) begin
            if (!first_inject_found && !core_running[pe_i] && !pcif_empty[pe_i]) begin
                first_inject_found = 1;
                first_inject_core  = pe_i[CORE_ID_BITS-1:0];
            end
        end
    end

    reg                      first_route_found;
    reg  [CORE_ID_BITS-1:0]  first_route_core;
    integer pe_j;
    always @(*) begin
        first_route_found = 0;
        first_route_core  = 0;
        for (pe_j = 0; pe_j < NUM_CORES; pe_j = pe_j + 1) begin
            if (!first_route_found && !cap_empty[pe_j]) begin
                first_route_found = 1;
                first_route_core  = pe_j[CORE_ID_BITS-1:0];
            end
        end
    end

    reg                      first_restart_found;
    reg  [CORE_ID_BITS-1:0]  first_restart_core;
    integer pe_k;
    always @(*) begin
        first_restart_found = 0;
        first_restart_core  = 0;
        for (pe_k = 0; pe_k < NUM_CORES; pe_k = pe_k + 1) begin
            if (!first_restart_found && core_needs_restart[pe_k] && !core_running[pe_k]) begin
                first_restart_found = 1;
                first_restart_core  = pe_k[CORE_ID_BITS-1:0];
            end
        end
    end

    wire quiescent = (core_running == 0) && (core_start_r == 0) &&
                     (core_needs_restart == 0) && (&pcif_empty) && (&cap_empty);

    reg [CORE_ID_BITS-1:0]  route_core_idx;
    reg [NEURON_BITS-1:0]   route_neuron;
    reg [7:0]               route_payload;
    reg [ROUTE_SLOT_BITS-1:0] route_slot;
    reg [GLOBAL_ROUTE_SLOT_BITS-1:0] global_slot;

    wire signed [31:0] route_weight_ext    = rt_weight;
    wire signed [31:0] route_payload_ext   = {24'd0, route_payload};
    wire signed [31:0] route_graded_product = route_weight_ext * route_payload_ext;
    wire signed [DATA_WIDTH-1:0] route_graded_current = route_graded_product >>> GRADE_SHIFT;

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
            rt_we          <= 0;
            rt_addr        <= 0;
            rt_wdata       <= 0;
            grt_we         <= 0;
            grt_addr       <= 0;
            inj_push       <= 0;
            inj_pop        <= 0;
            inj_clear      <= 0;
            cap_pop        <= 0;
            cap_clear      <= 0;
            pcif_push      <= 0;
            pcif_pop       <= 0;
            pcif_clear     <= 0;
            pcif_push_data <= 0;
            inject_core_idx <= 0;
            link_tx_push    <= 0;
            link_tx_core    <= 0;
            link_tx_neuron  <= 0;
            link_tx_payload <= 0;
            link_rx_pop     <= 0;
            dvfs_wait_cnt   <= 0;
        end else begin
            timestep_done <= 0;
            core_start_r  <= 0;
            rt_we         <= 0;
            grt_we        <= 0;
            inj_push      <= 0;
            inj_pop       <= 0;
            inj_clear     <= 0;
            cap_pop       <= 0;
            cap_clear     <= 0;
            pcif_push     <= 0;
            pcif_pop      <= 0;
            pcif_clear    <= 0;
            link_tx_push  <= 0;
            link_rx_pop   <= 0;

            total_spikes <= total_spikes + popcount(core_spike_valid);

            case (mesh_state)
                SM_IDLE: begin
                    if (async_enable && ext_valid) begin
                        pcif_push[ext_core] <= 1;
                        pcif_push_data <= {ext_neuron_id, ext_current};
                    end
                    if (start) begin
                        if (async_enable)
                            mesh_state <= SM_ASYNC_ACTIVE;
                        else if (CHIP_LINK_EN)
                            mesh_state <= SM_LINK_RX_DRAIN;
                        else
                            mesh_state <= SM_INJECT;
                    end
                end

                SM_INJECT: begin
                    if (inj_empty) begin
                        mesh_state <= SM_START;
                    end else begin
                        inj_pop <= 1;
                    end
                end

                SM_START: begin
                    core_start_r <= {NUM_CORES{1'b1}};
                    mesh_state   <= SM_RUN_WAIT;
                end

                SM_RUN_WAIT: begin
                    if (core_done_latch == {NUM_CORES{1'b1}}) begin
                        route_core_idx <= 0;
                        mesh_state     <= SM_ROUTE_POP;
                    end
                end

                SM_ROUTE_POP: begin
                    if (cap_empty[route_core_idx]) begin
                        if (route_core_idx == NUM_CORES - 1) begin
                            mesh_state <= SM_DONE;
                        end else begin
                            route_core_idx <= route_core_idx + 1;
                        end
                    end else begin
                        cap_pop[route_core_idx] <= 1;
                        route_neuron  <= (cap_data >> (route_core_idx * CAP_WIDTH + 8));
                        route_payload <= (cap_data >> (route_core_idx * CAP_WIDTH));
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
                        inj_push <= 1;
                        if (graded_enable)
                            inj_push_data <= {rt_dest_core, rt_dest_nrn, route_graded_current};
                        else
                            inj_push_data <= {rt_dest_core, rt_dest_nrn, rt_weight};
                    end

                    if (route_slot < ROUTE_FANOUT - 1) begin
                        route_slot <= route_slot + 1;
                        mesh_state <= SM_ROUTE_ADDR;
                    end else begin

                        global_slot <= 0;
                        mesh_state  <= SM_GLOBAL_ROUTE_ADDR;
                    end
                end

                SM_GLOBAL_ROUTE_ADDR: begin
                    grt_addr   <= {route_core_idx, route_neuron, global_slot};
                    mesh_state <= SM_GLOBAL_ROUTE_WAIT;
                end

                SM_GLOBAL_ROUTE_WAIT: begin
                    mesh_state <= SM_GLOBAL_ROUTE_READ;
                end

                SM_GLOBAL_ROUTE_READ: begin
                    if (grt_valid) begin
                        if (CHIP_LINK_EN && grt_weight[DATA_WIDTH-1]) begin

                            if (!link_tx_full) begin
                                link_tx_push    <= 1;
                                link_tx_core    <= grt_dest_core;
                                link_tx_neuron  <= grt_dest_nrn;
                                link_tx_payload <= route_payload;
                            end
                        end else begin

                            inj_push <= 1;
                            if (graded_enable)
                                inj_push_data <= {grt_dest_core, grt_dest_nrn, grt_graded_current};
                            else
                                inj_push_data <= {grt_dest_core, grt_dest_nrn, grt_weight};
                        end
                    end

                    if (global_slot < GLOBAL_ROUTE_SLOTS - 1) begin
                        global_slot <= global_slot + 1;
                        mesh_state  <= SM_GLOBAL_ROUTE_ADDR;
                    end else begin
                        mesh_state <= SM_ROUTE_POP;
                    end
                end

                SM_LINK_RX_DRAIN: begin
                    if (link_rx_empty) begin
                        mesh_state <= SM_INJECT;
                    end else if (!inj_full) begin
                        link_rx_pop <= 1;
                        inj_push <= 1;
                        inj_push_data <= {link_rx_core, link_rx_neuron, link_rx_current};
                        mesh_state <= SM_LINK_RX_WAIT;
                    end
                end

                SM_LINK_RX_WAIT: begin

                    mesh_state <= SM_LINK_RX_DRAIN;
                end

                SM_DONE: begin
                    cap_clear      <= {NUM_CORES{1'b1}};
                    timestep_count <= timestep_count + 1;
                    if (dvfs_stall > 0) begin
                        dvfs_wait_cnt <= dvfs_stall;
                        mesh_state    <= SM_DVFS_WAIT;
                    end else begin
                        timestep_done <= 1;
                        mesh_state    <= SM_IDLE;
                    end
                end

                SM_DVFS_WAIT: begin
                    if (dvfs_wait_cnt <= 1) begin
                        timestep_done <= 1;
                        mesh_state    <= SM_IDLE;
                    end else begin
                        dvfs_wait_cnt <= dvfs_wait_cnt - 1;
                    end
                end

                SM_ASYNC_ACTIVE: begin
                    if (quiescent) begin
                        mesh_state <= SM_ASYNC_DONE;
                    end else if (first_inject_found) begin
                        inject_core_idx <= first_inject_core;
                        mesh_state <= SM_ASYNC_INJECT;
                    end else if (first_route_found) begin
                        route_core_idx <= first_route_core;
                        mesh_state <= SM_ASYNC_ROUTE_POP;
                    end else if (first_restart_found) begin
                        core_start_r <= ({{(NUM_CORES-1){1'b0}}, 1'b1} << first_restart_core);
                    end
                end

                SM_ASYNC_INJECT: begin
                    if (pcif_empty[inject_core_idx]) begin
                        core_start_r <= ({{(NUM_CORES-1){1'b0}}, 1'b1} << inject_core_idx);
                        mesh_state <= SM_ASYNC_ACTIVE;
                    end else begin
                        pcif_pop[inject_core_idx] <= 1;
                    end
                end

                SM_ASYNC_ROUTE_POP: begin
                    if (cap_empty[route_core_idx]) begin
                        mesh_state <= SM_ASYNC_ACTIVE;
                    end else begin
                        cap_pop[route_core_idx] <= 1;
                        route_neuron  <= (cap_data >> (route_core_idx * CAP_WIDTH + 8));
                        route_payload <= (cap_data >> (route_core_idx * CAP_WIDTH));
                        route_slot    <= 0;
                        mesh_state    <= SM_ASYNC_ROUTE_ADDR;
                    end
                end

                SM_ASYNC_ROUTE_ADDR: begin
                    rt_addr    <= {route_core_idx, route_neuron, route_slot};
                    mesh_state <= SM_ASYNC_ROUTE_WAIT;
                end

                SM_ASYNC_ROUTE_WAIT: begin
                    mesh_state <= SM_ASYNC_ROUTE_READ;
                end

                SM_ASYNC_ROUTE_READ: begin
                    if (rt_valid && !pcif_full[rt_dest_core]) begin
                        pcif_push[rt_dest_core] <= 1;
                        if (graded_enable)
                            pcif_push_data <= {rt_dest_nrn, route_graded_current};
                        else
                            pcif_push_data <= {rt_dest_nrn, rt_weight};
                    end

                    if (route_slot < ROUTE_FANOUT - 1) begin
                        route_slot <= route_slot + 1;
                        mesh_state <= SM_ASYNC_ROUTE_ADDR;
                    end else begin
                        mesh_state <= SM_ASYNC_ROUTE_POP;
                    end
                end

                SM_ASYNC_DONE: begin
                    pcif_clear     <= {NUM_CORES{1'b1}};
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
