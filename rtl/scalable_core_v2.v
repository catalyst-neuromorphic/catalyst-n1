module scalable_core_v2 #(
    parameter NUM_NEURONS      = 1024,
    parameter NEURON_BITS      = 10,
    parameter DATA_WIDTH       = 16,
    parameter POOL_DEPTH       = 131072,
    parameter POOL_ADDR_BITS   = 17,
    parameter COUNT_BITS       = 12,
    parameter REV_FANIN        = 32,
    parameter REV_SLOT_BITS    = 5,
    parameter THRESHOLD        = 16'sd1000,
    parameter LEAK_RATE        = 16'sd3,
    parameter RESTING_POT      = 16'sd0,
    parameter REFRAC_CYCLES    = 4,
    parameter TRACE_MAX        = 8'd100,
    parameter TRACE_DECAY      = 8'd3,
    parameter LEARN_SHIFT      = 3,
    parameter GRADE_SHIFT      = 7,
    parameter COMPARTMENT_BITS = 2,
    parameter signed [DATA_WIDTH-1:0] DEND_THRESHOLD = 16'sd0,
    parameter signed [DATA_WIDTH-1:0] WEIGHT_MAX = 16'sd2000,
    parameter signed [DATA_WIDTH-1:0] WEIGHT_MIN = 16'sd0,
    parameter REWARD_SHIFT      = 7,
    parameter ELIG_DECAY_SHIFT  = 3,
    parameter signed [DATA_WIDTH-1:0] ELIG_MAX = 16'sd1000,
    parameter [15:0] NOISE_LFSR_SEED = 16'hACE1,
    parameter [3:0]  TAU1_DEFAULT    = 4'd3,
    parameter [3:0]  TAU2_DEFAULT    = 4'd4,
    parameter DELAY_BITS           = 6,
    parameter DELAY_ENTRIES_PER_TS = 64,
    parameter DELAY_ENTRY_BITS     = 6,
    parameter NEURON_WIDTH         = 24
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    start,
    input  wire                    learn_enable,
    input  wire                    graded_enable,
    input  wire                    dendritic_enable,
    input  wire                    threefactor_enable,
    input  wire                    noise_enable,
    input  wire                    skip_idle_enable,
    input  wire                    scale_u_enable,
    input  wire signed [DATA_WIDTH-1:0] reward_value,

    input  wire                        ext_valid,
    input  wire [NEURON_BITS-1:0]      ext_neuron_id,
    input  wire signed [DATA_WIDTH-1:0] ext_current,

    input  wire                         pool_we,
    input  wire [POOL_ADDR_BITS-1:0]   pool_addr_in,
    input  wire [NEURON_BITS-1:0]      pool_src_in,
    input  wire [NEURON_BITS-1:0]      pool_target_in,
    input  wire signed [DATA_WIDTH-1:0] pool_weight_in,
    input  wire [COMPARTMENT_BITS-1:0] pool_comp_in,

    input  wire                         index_we,
    input  wire [NEURON_BITS-1:0]      index_neuron_in,
    input  wire [POOL_ADDR_BITS-1:0]   index_base_in,
    input  wire [COUNT_BITS-1:0]       index_count_in,
    input  wire [1:0]                  index_format_in,

    input  wire                         delay_we,
    input  wire [POOL_ADDR_BITS-1:0]   delay_addr_in,
    input  wire [DELAY_BITS-1:0]       delay_value_in,

    input  wire                         ucode_prog_we,
    input  wire [7:0]                   ucode_prog_addr,
    input  wire [31:0]                  ucode_prog_data,

    input  wire                        prog_param_we,
    input  wire [NEURON_BITS-1:0]      prog_param_neuron,
    input  wire [4:0]                  prog_param_id,
    input  wire signed [DATA_WIDTH-1:0] prog_param_value,

    input  wire                        probe_read,
    input  wire [NEURON_BITS-1:0]      probe_neuron,
    input  wire [4:0]                  probe_state_id,
    input  wire [POOL_ADDR_BITS-1:0]   probe_pool_addr,
    output reg  signed [DATA_WIDTH-1:0] probe_data,
    output reg                         probe_valid,

    output reg                     timestep_done,
    output reg                     spike_out_valid,
    output reg  [NEURON_BITS-1:0]  spike_out_id,
    output reg  [7:0]              spike_out_payload,
    output wire [5:0]              state_out,
    output reg  [31:0]             total_spikes,
    output reg  [31:0]             timestep_count,

    output wire                    core_idle
);

    localparam S_IDLE              = 6'd0;
    localparam S_DELIVER_POP       = 6'd1;
    localparam S_DELIVER_IDX_WAIT  = 6'd2;
    localparam S_DELIVER_IDX_READ  = 6'd3;
    localparam S_DELIVER_POOL_WAIT = 6'd4;
    localparam S_DELIVER_ADDR      = 6'd5;
    localparam S_DELIVER_ACC_WAIT  = 6'd6;
    localparam S_DELIVER_ACC       = 6'd7;
    localparam S_DELIVER_NEXT      = 6'd8;
    localparam S_UPDATE_INIT       = 6'd9;
    localparam S_UPDATE_READ       = 6'd10;
    localparam S_UPDATE_CALC       = 6'd11;
    localparam S_UPDATE_WRITE      = 6'd12;
    localparam S_LEARN_MC_SCAN     = 6'd13;
    localparam S_LEARN_MC_IDX_WAIT = 6'd14;
    localparam S_LEARN_MC_IDX_READ = 6'd15;
    localparam S_LEARN_MC_SETUP    = 6'd16;
    localparam S_LEARN_MC_WAIT1    = 6'd17;
    localparam S_LEARN_MC_LOAD     = 6'd18;
    localparam S_LEARN_MC_WAIT2    = 6'd19;
    localparam S_LEARN_MC_REGLD    = 6'd20;
    localparam S_LEARN_MC_FETCH    = 6'd21;
    localparam S_DONE              = 6'd22;
    localparam S_LEARN_MC_EXEC     = 6'd23;
    localparam S_LEARN_MC_NEXT     = 6'd24;
    localparam S_ELIG_MC           = 6'd25;
    localparam S_DELAY_DRAIN_INIT  = 6'd26;
    localparam S_DELAY_DRAIN_QWAIT = 6'd27;
    localparam S_DELAY_DRAIN_CAP   = 6'd28;
    localparam S_DELAY_DRAIN_AWAIT = 6'd29;
    localparam S_DELAY_DRAIN_ACC   = 6'd30;

    localparam S_UPDATE_PARENT_ADDR = 6'd31;
    localparam S_UPDATE_PARENT_WAIT = 6'd32;
    localparam S_UPDATE_PARENT_ACC  = 6'd33;

    localparam S_DELIVER_AXTYPE     = 6'd34;

    function signed [NEURON_WIDTH-1:0] raz_div4096;
        input signed [NEURON_WIDTH+11:0] product;
        reg signed [NEURON_WIDTH-1:0] truncated;
        reg has_frac;
        begin
            truncated = product[NEURON_WIDTH+11:12];
            has_frac  = |product[11:0];
            if (has_frac)
                raz_div4096 = truncated + (product[NEURON_WIDTH+11] ? -1 : 1);
            else
                raz_div4096 = truncated;
        end
    endfunction

    reg [5:0] state;
    assign state_out = state;

    reg was_idle;
    reg any_spike_this_ts;
    assign core_idle = was_idle;

    wire signed [DATA_WIDTH-1:0] probe_nrn_rdata;
    wire [7:0]                   probe_ref_rdata;
    wire signed [NEURON_WIDTH-1:0] probe_acc_rdata;
    wire signed [DATA_WIDTH-1:0] probe_wt_rdata;
    wire signed [DATA_WIDTH-1:0] probe_elig_rdata;
    wire [7:0]                   probe_trace1_rdata;
    wire [7:0]                   probe_trace2_rdata;
    wire signed [DATA_WIDTH-1:0] probe_dend1_rdata;
    wire signed [DATA_WIDTH-1:0] probe_dend2_rdata;
    wire signed [DATA_WIDTH-1:0] probe_dend3_rdata;

    reg [31:0] perf_spike_count;
    reg [31:0] perf_active_cycles;
    reg [31:0] perf_synaptic_ops;
    wire [31:0] perf_power_estimate = (perf_spike_count << 3) +
                                       (perf_synaptic_ops << 1) + perf_active_cycles;

    reg        trace_fifo_enable;
    reg [31:0] trace_fifo_mem [0:63];
    reg [6:0]  trace_wr_ptr, trace_rd_ptr;
    wire [6:0] trace_count_val = trace_wr_ptr - trace_rd_ptr;
    wire       trace_fifo_full  = (trace_count_val >= 7'd64);
    wire       trace_fifo_empty = (trace_wr_ptr == trace_rd_ptr);
    reg [31:0] trace_last_popped;

    reg probe_active_r;
    reg [4:0] probe_sid_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            probe_active_r <= 0;
            probe_sid_r    <= 0;
            probe_valid    <= 0;
            probe_data     <= 0;
        end else begin
            probe_active_r <= probe_read && (state == S_IDLE);
            probe_sid_r    <= probe_state_id;
            probe_valid    <= probe_active_r;
            if (probe_active_r) begin
                case (probe_sid_r)
                    5'd0:  probe_data <= probe_nrn_rdata;
                    5'd1:  probe_data <= param_thr_rdata[DATA_WIDTH-1:0];
                    5'd2:  probe_data <= {{(DATA_WIDTH-8){1'b0}}, probe_trace1_rdata};
                    5'd3:  probe_data <= {{(DATA_WIDTH-8){1'b0}}, probe_trace2_rdata};
                    5'd4:  probe_data <= {{(DATA_WIDTH-4){1'b0}}, probe_ref_rdata};
                    5'd5:  probe_data <= probe_acc_rdata[DATA_WIDTH-1:0];
                    5'd6:  probe_data <= probe_dend1_rdata;
                    5'd7:  probe_data <= probe_dend2_rdata;
                    5'd8:  probe_data <= probe_dend3_rdata;
                    5'd9:  probe_data <= param_leak_rdata;
                    5'd10: probe_data <= param_rest_rdata;
                    5'd11: probe_data <= probe_wt_rdata;
                    5'd12: probe_data <= probe_elig_rdata;
                    5'd13: probe_data <= probe_cur_full[DATA_WIDTH-1:0];
                    5'd14: probe_data <= perf_spike_count[15:0];
                    5'd15: probe_data <= perf_spike_count[31:16];
                    5'd16: probe_data <= perf_active_cycles[15:0];
                    5'd17: probe_data <= perf_active_cycles[31:16];
                    5'd18: probe_data <= perf_synaptic_ops[15:0];
                    5'd19: probe_data <= perf_synaptic_ops[31:16];
                    5'd20: probe_data <= perf_power_estimate[15:0];
                    5'd21: probe_data <= perf_power_estimate[31:16];
                    5'd22: probe_data <= trace_fifo_empty ? 16'hFFFF :
                                         trace_fifo_mem[trace_rd_ptr[5:0]][15:0];
                    5'd23: probe_data <= trace_last_popped[31:16];
                    5'd24: probe_data <= {9'd0, trace_count_val};
                    default: probe_data <= 16'sd0;
                endcase
            end
        end
    end

    reg                    nrn_we;
    reg  [NEURON_BITS-1:0] nrn_addr;
    reg  signed [NEURON_WIDTH-1:0] nrn_wdata;
    wire signed [NEURON_WIDTH-1:0] nrn_rdata;

    wire signed [NEURON_WIDTH-1:0] probe_nrn_full;
    sram #(.DATA_WIDTH(NEURON_WIDTH), .ADDR_WIDTH(NEURON_BITS)) neuron_mem (
        .clk(clk), .we_a(nrn_we), .addr_a(nrn_addr),
        .wdata_a(nrn_wdata), .rdata_a(nrn_rdata),
        .addr_b(probe_neuron), .rdata_b(probe_nrn_full)
    );
    assign probe_nrn_rdata = probe_nrn_full[DATA_WIDTH-1:0];

    reg                    cur_we;
    reg  [NEURON_BITS-1:0] cur_addr;
    reg  signed [NEURON_WIDTH-1:0] cur_wdata;
    wire signed [NEURON_WIDTH-1:0] cur_rdata;
    wire signed [NEURON_WIDTH-1:0] probe_cur_full;

    sram #(.DATA_WIDTH(NEURON_WIDTH), .ADDR_WIDTH(NEURON_BITS)) current_mem (
        .clk(clk), .we_a(cur_we), .addr_a(cur_addr),
        .wdata_a(cur_wdata), .rdata_a(cur_rdata),
        .addr_b(probe_neuron), .rdata_b(probe_cur_full)
    );

    reg                    ref_we;
    reg  [NEURON_BITS-1:0] ref_addr;
    reg  [7:0]             ref_wdata;
    wire [7:0]             ref_rdata;

    sram #(.DATA_WIDTH(8), .ADDR_WIDTH(NEURON_BITS)) refrac_mem (
        .clk(clk), .we_a(ref_we), .addr_a(ref_addr),
        .wdata_a(ref_wdata), .rdata_a(ref_rdata),
        .addr_b(probe_neuron), .rdata_b(probe_ref_rdata)
    );

    reg                    acc_we;
    reg  [NEURON_BITS-1:0] acc_addr;
    reg  signed [NEURON_WIDTH-1:0] acc_wdata;
    wire signed [NEURON_WIDTH-1:0] acc_rdata;

    sram #(.DATA_WIDTH(NEURON_WIDTH), .ADDR_WIDTH(NEURON_BITS)) acc_mem (
        .clk(clk), .we_a(acc_we), .addr_a(acc_addr),
        .wdata_a(acc_wdata), .rdata_a(acc_rdata),
        .addr_b(probe_neuron), .rdata_b(probe_acc_rdata)
    );

    localparam INDEX_WIDTH = 2 + POOL_ADDR_BITS + COUNT_BITS;

    localparam FMT_SPARSE = 2'd0;
    localparam FMT_DENSE  = 2'd1;
    localparam FMT_POP    = 2'd2;

    reg  [NEURON_BITS-1:0]   index_rd_addr;
    wire [INDEX_WIDTH-1:0]   index_rdata;

    wire                     index_we_mux  = (state == S_IDLE) ? index_we : 1'b0;
    wire [NEURON_BITS-1:0]   index_addr_mux = (state == S_IDLE) ? index_neuron_in : index_rd_addr;
    wire [INDEX_WIDTH-1:0]   index_wdata_mux = {index_format_in, index_base_in, index_count_in};

    sram #(.DATA_WIDTH(INDEX_WIDTH), .ADDR_WIDTH(NEURON_BITS)) index_mem (
        .clk(clk), .we_a(index_we_mux), .addr_a(index_addr_mux),
        .wdata_a(index_wdata_mux), .rdata_a(index_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    reg  [POOL_ADDR_BITS-1:0] pool_addr_r;
    wire [NEURON_BITS-1:0]    pool_tgt_rdata;

    wire                      pool_tgt_we_mux  = (state == S_IDLE) ? pool_we : 1'b0;
    wire [POOL_ADDR_BITS-1:0] pool_tgt_addr_mux = (state == S_IDLE) ? pool_addr_in : pool_addr_r;

    sram #(.DATA_WIDTH(NEURON_BITS), .ADDR_WIDTH(POOL_ADDR_BITS)) pool_target_mem (
        .clk(clk), .we_a(pool_tgt_we_mux), .addr_a(pool_tgt_addr_mux),
        .wdata_a((state == S_IDLE) ? pool_target_in : {NEURON_BITS{1'b0}}),
        .rdata_a(pool_tgt_rdata),
        .addr_b({POOL_ADDR_BITS{1'b0}}), .rdata_b()
    );

    reg                         pool_wt_we_r;
    reg  [POOL_ADDR_BITS-1:0]  pool_wt_wr_addr;
    reg  signed [DATA_WIDTH-1:0] pool_wt_wr_data;
    wire signed [DATA_WIDTH-1:0] pool_wt_rdata;

    wire                        pool_wt_we_mux = (state == S_IDLE) ? pool_we : pool_wt_we_r;
    wire [POOL_ADDR_BITS-1:0]   pool_wt_addr_mux = (state == S_IDLE) ? pool_addr_in :
        (pool_wt_we_r ? pool_wt_wr_addr : pool_addr_r);
    wire signed [DATA_WIDTH-1:0] pool_wt_wdata_mux = (state == S_IDLE) ? pool_weight_in : pool_wt_wr_data;

    sram #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(POOL_ADDR_BITS)) pool_weight_mem (
        .clk(clk), .we_a(pool_wt_we_mux), .addr_a(pool_wt_addr_mux),
        .wdata_a(pool_wt_wdata_mux), .rdata_a(pool_wt_rdata),
        .addr_b(probe_pool_addr), .rdata_b(probe_wt_rdata)
    );

    wire [COMPARTMENT_BITS-1:0] pool_comp_rdata;

    wire                        pool_comp_we_mux = (state == S_IDLE) ? pool_we : 1'b0;
    wire [POOL_ADDR_BITS-1:0]   pool_comp_addr_mux = (state == S_IDLE) ? pool_addr_in : pool_addr_r;

    sram #(.DATA_WIDTH(COMPARTMENT_BITS), .ADDR_WIDTH(POOL_ADDR_BITS)) pool_comp_mem (
        .clk(clk), .we_a(pool_comp_we_mux), .addr_a(pool_comp_addr_mux),
        .wdata_a((state == S_IDLE) ? pool_comp_in : {COMPARTMENT_BITS{1'b0}}),
        .rdata_a(pool_comp_rdata),
        .addr_b({POOL_ADDR_BITS{1'b0}}), .rdata_b()
    );

    reg                         elig_we;
    reg  [POOL_ADDR_BITS-1:0]  elig_addr;
    reg  signed [DATA_WIDTH-1:0] elig_wdata;
    wire signed [DATA_WIDTH-1:0] elig_rdata;

    sram #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(POOL_ADDR_BITS)) elig_mem (
        .clk(clk), .we_a(elig_we), .addr_a(elig_addr),
        .wdata_a(elig_wdata), .rdata_a(elig_rdata),
        .addr_b(probe_pool_addr), .rdata_b(probe_elig_rdata)
    );

    localparam UCODE_DEPTH     = 256;
    localparam UCODE_ADDR_BITS = 8;
    localparam UCODE_WIDTH     = 32;

    localparam [15:0] MC_WEIGHT_MIN   = WEIGHT_MIN;
    localparam [15:0] MC_WEIGHT_MAX   = WEIGHT_MAX;
    localparam [15:0] MC_ELIG_MAX     = ELIG_MAX;
    localparam [15:0] MC_NEG_ELIG_MAX = -ELIG_MAX;

    reg  [UCODE_ADDR_BITS-1:0] mc_pc;
    wire [UCODE_WIDTH-1:0]     ucode_rdata;

    wire mc_ucode_we = (state == S_IDLE) ? ucode_prog_we : 1'b0;
    wire [UCODE_ADDR_BITS-1:0] mc_ucode_addr = (state == S_IDLE) ? ucode_prog_addr : mc_pc;

    sram #(.DATA_WIDTH(UCODE_WIDTH), .ADDR_WIDTH(UCODE_ADDR_BITS)) ucode_mem (
        .clk(clk), .we_a(mc_ucode_we), .addr_a(mc_ucode_addr),
        .wdata_a(ucode_prog_data), .rdata_a(ucode_rdata),
        .addr_b({UCODE_ADDR_BITS{1'b0}}), .rdata_b()
    );

    reg signed [DATA_WIDTH-1:0] mc_regs [0:15];
    reg [1:0] elig_phase;

    localparam DELAY_QUEUE_ADDR_W = DELAY_BITS + DELAY_ENTRY_BITS;
    localparam DELAY_QUEUE_ENTRY_W = NEURON_BITS + DATA_WIDTH + COMPARTMENT_BITS;

    wire [DELAY_BITS-1:0] pool_delay_rdata;

    reg pool_delay_we_learn;
    reg [POOL_ADDR_BITS-1:0] pool_delay_learn_addr;
    reg [5:0] pool_delay_learn_data;

    wire                       pool_delay_we_mux  = (state == S_IDLE) ? delay_we : pool_delay_we_learn;
    wire [POOL_ADDR_BITS-1:0]  pool_delay_addr_mux = (state == S_IDLE) ? delay_addr_in :
        (pool_delay_we_learn ? pool_delay_learn_addr : pool_addr_r);
    wire [5:0] pool_delay_wdata_mux = (state == S_IDLE) ? delay_value_in : pool_delay_learn_data;

    sram #(.DATA_WIDTH(DELAY_BITS), .ADDR_WIDTH(POOL_ADDR_BITS)) pool_delay_mem (
        .clk(clk), .we_a(pool_delay_we_mux), .addr_a(pool_delay_addr_mux),
        .wdata_a(pool_delay_wdata_mux),
        .rdata_a(pool_delay_rdata),
        .addr_b({POOL_ADDR_BITS{1'b0}}), .rdata_b()
    );

    wire signed [DATA_WIDTH-1:0] pool_tag_rdata;
    reg                          pool_tag_we_r;
    reg  [POOL_ADDR_BITS-1:0]   pool_tag_wr_addr;
    reg  signed [DATA_WIDTH-1:0] pool_tag_wr_data;

    wire pool_tag_we_mux = (state == S_IDLE) ? 1'b0 : pool_tag_we_r;
    wire [POOL_ADDR_BITS-1:0] pool_tag_addr_mux = (state == S_IDLE) ? pool_addr_r :
        (pool_tag_we_r ? pool_tag_wr_addr : pool_addr_r);

    sram #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(POOL_ADDR_BITS)) pool_tag_mem (
        .clk(clk), .we_a(pool_tag_we_mux), .addr_a(pool_tag_addr_mux),
        .wdata_a(pool_tag_wr_data), .rdata_a(pool_tag_rdata),
        .addr_b({POOL_ADDR_BITS{1'b0}}), .rdata_b()
    );

    reg                         dq_we;
    reg  [DELAY_QUEUE_ADDR_W-1:0] dq_addr;
    reg  [DELAY_QUEUE_ENTRY_W-1:0] dq_wdata;
    wire [DELAY_QUEUE_ENTRY_W-1:0] dq_rdata;

    sram #(.DATA_WIDTH(DELAY_QUEUE_ENTRY_W), .ADDR_WIDTH(DELAY_QUEUE_ADDR_W)) delay_queue_mem (
        .clk(clk), .we_a(dq_we), .addr_a(dq_addr),
        .wdata_a(dq_wdata), .rdata_a(dq_rdata),
        .addr_b({DELAY_QUEUE_ADDR_W{1'b0}}), .rdata_b()
    );

    reg [DELAY_ENTRY_BITS:0] delay_count [0:(1 << DELAY_BITS)-1];

    reg [DELAY_BITS-1:0]       current_ts_mod64;
    reg [DELAY_ENTRY_BITS:0]   drain_cnt;
    reg [DELAY_ENTRY_BITS-1:0] drain_idx;
    reg [NEURON_BITS-1:0]          dq_cap_target;
    reg signed [DATA_WIDTH-1:0]    dq_cap_current;
    reg [COMPARTMENT_BITS-1:0]     dq_cap_comp;

    wire [DELAY_BITS-1:0] delivery_ts = current_ts_mod64 + pool_delay_rdata;
    wire signed [DATA_WIDTH-1:0] delivered_current = graded_enable ? graded_current : saved_weight;

    integer dci;
    initial begin
        for (dci = 0; dci < (1 << DELAY_BITS); dci = dci + 1)
            delay_count[dci] = 0;
    end

    reg                         dend_acc_1_we;
    reg  [NEURON_BITS-1:0]     dend_acc_1_addr;
    reg  signed [DATA_WIDTH-1:0] dend_acc_1_wdata;
    wire signed [DATA_WIDTH-1:0] dend_acc_1_rdata;

    sram #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(NEURON_BITS)) dend_acc_1_mem (
        .clk(clk), .we_a(dend_acc_1_we), .addr_a(dend_acc_1_addr),
        .wdata_a(dend_acc_1_wdata), .rdata_a(dend_acc_1_rdata),
        .addr_b(probe_neuron), .rdata_b(probe_dend1_rdata)
    );

    reg                         dend_acc_2_we;
    reg  [NEURON_BITS-1:0]     dend_acc_2_addr;
    reg  signed [DATA_WIDTH-1:0] dend_acc_2_wdata;
    wire signed [DATA_WIDTH-1:0] dend_acc_2_rdata;

    sram #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(NEURON_BITS)) dend_acc_2_mem (
        .clk(clk), .we_a(dend_acc_2_we), .addr_a(dend_acc_2_addr),
        .wdata_a(dend_acc_2_wdata), .rdata_a(dend_acc_2_rdata),
        .addr_b(probe_neuron), .rdata_b(probe_dend2_rdata)
    );

    reg                         dend_acc_3_we;
    reg  [NEURON_BITS-1:0]     dend_acc_3_addr;
    reg  signed [DATA_WIDTH-1:0] dend_acc_3_wdata;
    wire signed [DATA_WIDTH-1:0] dend_acc_3_rdata;

    sram #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(NEURON_BITS)) dend_acc_3_mem (
        .clk(clk), .we_a(dend_acc_3_we), .addr_a(dend_acc_3_addr),
        .wdata_a(dend_acc_3_wdata), .rdata_a(dend_acc_3_rdata),
        .addr_b(probe_neuron), .rdata_b(probe_dend3_rdata)
    );

    reg                    trace_we;
    reg  [NEURON_BITS-1:0] trace_addr;
    reg  [7:0]             trace_wdata;
    wire [7:0]             trace_rdata;

    sram #(.DATA_WIDTH(8), .ADDR_WIDTH(NEURON_BITS)) trace_mem (
        .clk(clk), .we_a(trace_we), .addr_a(trace_addr),
        .wdata_a(trace_wdata), .rdata_a(trace_rdata),
        .addr_b(probe_neuron), .rdata_b(probe_trace1_rdata)
    );

    reg                    trace2_we;
    reg  [NEURON_BITS-1:0] trace2_addr;
    reg  [7:0]             trace2_wdata;
    wire [7:0]             trace2_rdata;

    sram #(.DATA_WIDTH(8), .ADDR_WIDTH(NEURON_BITS)) trace2_mem (
        .clk(clk), .we_a(trace2_we), .addr_a(trace2_addr),
        .wdata_a(trace2_wdata), .rdata_a(trace2_rdata),
        .addr_b(probe_neuron), .rdata_b(probe_trace2_rdata)
    );

    reg                    x2_trace_we;
    reg  [NEURON_BITS-1:0] x2_trace_addr;
    reg  [7:0]             x2_trace_wdata;
    wire [7:0]             x2_trace_rdata;

    sram #(.DATA_WIDTH(8), .ADDR_WIDTH(NEURON_BITS)) x2_trace_mem (
        .clk(clk), .we_a(x2_trace_we), .addr_a(x2_trace_addr),
        .wdata_a(x2_trace_wdata), .rdata_a(x2_trace_rdata),
        .addr_b(probe_neuron), .rdata_b()
    );

    reg                    y2_trace_we;
    reg  [NEURON_BITS-1:0] y2_trace_addr;
    reg  [7:0]             y2_trace_wdata;
    wire [7:0]             y2_trace_rdata;

    sram #(.DATA_WIDTH(8), .ADDR_WIDTH(NEURON_BITS)) y2_trace_mem (
        .clk(clk), .we_a(y2_trace_we), .addr_a(y2_trace_addr),
        .wdata_a(y2_trace_wdata), .rdata_a(y2_trace_rdata),
        .addr_b(probe_neuron), .rdata_b()
    );

    reg                    y3_trace_we;
    reg  [NEURON_BITS-1:0] y3_trace_addr;
    reg  [7:0]             y3_trace_wdata;
    wire [7:0]             y3_trace_rdata;

    sram #(.DATA_WIDTH(8), .ADDR_WIDTH(NEURON_BITS)) y3_trace_mem (
        .clk(clk), .we_a(y3_trace_we), .addr_a(y3_trace_addr),
        .wdata_a(y3_trace_wdata), .rdata_a(y3_trace_rdata),
        .addr_b(probe_neuron), .rdata_b()
    );

    localparam REV_DATA_W  = 1 + NEURON_BITS + POOL_ADDR_BITS;
    localparam REV_ADDR_W  = NEURON_BITS + REV_SLOT_BITS;

    reg  [REV_ADDR_W-1:0]   rev_addr;
    wire [REV_DATA_W-1:0]   rev_rdata;

    reg [REV_SLOT_BITS-1:0] rev_count [0:NUM_NEURONS-1];

    wire                     rev_we_mux  = (state == S_IDLE) ? pool_we : 1'b0;
    wire [REV_ADDR_W-1:0]   rev_addr_mux = (state == S_IDLE) ?
        {pool_target_in, rev_count[pool_target_in]} : rev_addr;
    wire [REV_DATA_W-1:0]   rev_wdata_mux = (state == S_IDLE) ?
        {1'b1, pool_src_in, pool_addr_in} : {REV_DATA_W{1'b0}};

    sram #(.DATA_WIDTH(REV_DATA_W), .ADDR_WIDTH(REV_ADDR_W)) rev_conn_mem (
        .clk(clk), .we_a(rev_we_mux), .addr_a(rev_addr_mux),
        .wdata_a(rev_wdata_mux), .rdata_a(rev_rdata),
        .addr_b({REV_ADDR_W{1'b0}}), .rdata_b()
    );

    integer rci;
    initial begin
        for (rci = 0; rci < NUM_NEURONS; rci = rci + 1)
            rev_count[rci] = 0;
    end

    wire [NEURON_BITS-1:0] param_sram_addr =
        (state == S_IDLE) ? prog_param_neuron : proc_neuron[NEURON_BITS-1:0];

    wire param_thr_we  = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd0);
    reg homeo_thr_we;
    reg signed [NEURON_WIDTH-1:0] homeo_thr_wdata;
    wire thr_we_final = param_thr_we || homeo_thr_we;
    wire signed [NEURON_WIDTH-1:0] thr_wdata_final = homeo_thr_we ? homeo_thr_wdata : $signed(prog_param_value);
    wire signed [NEURON_WIDTH-1:0] param_thr_rdata;
    sram #(.DATA_WIDTH(NEURON_WIDTH), .ADDR_WIDTH(NEURON_BITS)) threshold_mem (
        .clk(clk), .we_a(thr_we_final), .addr_a(param_sram_addr),
        .wdata_a(thr_wdata_final), .rdata_a(param_thr_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    wire param_leak_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd1);
    wire signed [DATA_WIDTH-1:0] param_leak_rdata;
    sram #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(NEURON_BITS)) leak_mem (
        .clk(clk), .we_a(param_leak_we), .addr_a(param_sram_addr),
        .wdata_a(prog_param_value), .rdata_a(param_leak_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    wire param_rest_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd2);
    wire signed [DATA_WIDTH-1:0] param_rest_rdata;
    sram #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(NEURON_BITS)) rest_mem (
        .clk(clk), .we_a(param_rest_we), .addr_a(param_sram_addr),
        .wdata_a(prog_param_value), .rdata_a(param_rest_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    wire param_refrac_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd3);
    wire [15:0] param_refrac_rdata;
    sram #(.DATA_WIDTH(16), .ADDR_WIDTH(NEURON_BITS)) refrac_cfg_mem (
        .clk(clk), .we_a(param_refrac_we), .addr_a(param_sram_addr),
        .wdata_a(prog_param_value[15:0]), .rdata_a(param_refrac_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );
    wire       refrac_mode_abs = param_refrac_rdata[8];
    wire       refrac_mode_rel = param_refrac_rdata[9];

    wire param_dend_thr_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd4);
    wire signed [DATA_WIDTH-1:0] param_dend_thr_rdata;
    sram #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(NEURON_BITS)) dend_thr_mem (
        .clk(clk), .we_a(param_dend_thr_we), .addr_a(param_sram_addr),
        .wdata_a(prog_param_value), .rdata_a(param_dend_thr_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    wire param_noise_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd5);
    wire [11:0] param_noise_rdata;
    sram #(.DATA_WIDTH(12), .ADDR_WIDTH(NEURON_BITS)) noise_cfg_mem (
        .clk(clk), .we_a(param_noise_we), .addr_a(param_sram_addr),
        .wdata_a(prog_param_value[11:0]), .rdata_a(param_noise_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    wire param_noise_target_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd29);
    wire [1:0] param_noise_target_rdata;
    sram #(.DATA_WIDTH(2), .ADDR_WIDTH(NEURON_BITS)) noise_target_mem (
        .clk(clk), .we_a(param_noise_target_we), .addr_a(param_sram_addr),
        .wdata_a(prog_param_value[1:0]), .rdata_a(param_noise_target_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    wire param_vmin_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd30);
    wire signed [DATA_WIDTH-1:0] param_vmin_rdata;
    sram #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(NEURON_BITS)) vmin_mem (
        .clk(clk), .we_a(param_vmin_we), .addr_a(param_sram_addr),
        .wdata_a(prog_param_value), .rdata_a(param_vmin_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    wire param_vmax_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd31);
    wire signed [DATA_WIDTH-1:0] param_vmax_rdata;
    sram #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(NEURON_BITS)) vmax_mem (
        .clk(clk), .we_a(param_vmax_we), .addr_a(param_sram_addr),
        .wdata_a(prog_param_value), .rdata_a(param_vmax_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    wire param_tau1_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd6);
    wire [3:0] param_tau1_rdata;
    sram #(.DATA_WIDTH(4), .ADDR_WIDTH(NEURON_BITS)) tau1_cfg_mem (
        .clk(clk), .we_a(param_tau1_we), .addr_a(param_sram_addr),
        .wdata_a(prog_param_value[3:0]), .rdata_a(param_tau1_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    wire param_tau2_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd7);
    wire [3:0] param_tau2_rdata;
    sram #(.DATA_WIDTH(4), .ADDR_WIDTH(NEURON_BITS)) tau2_cfg_mem (
        .clk(clk), .we_a(param_tau2_we), .addr_a(param_sram_addr),
        .wdata_a(prog_param_value[3:0]), .rdata_a(param_tau2_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    wire param_tau_x2_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd19);
    wire [3:0] param_tau_x2_rdata;
    sram #(.DATA_WIDTH(4), .ADDR_WIDTH(NEURON_BITS)) tau_x2_cfg_mem (
        .clk(clk), .we_a(param_tau_x2_we), .addr_a(param_sram_addr),
        .wdata_a(prog_param_value[3:0]), .rdata_a(param_tau_x2_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    wire param_tau_y2_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd20);
    wire [3:0] param_tau_y2_rdata;
    sram #(.DATA_WIDTH(4), .ADDR_WIDTH(NEURON_BITS)) tau_y2_cfg_mem (
        .clk(clk), .we_a(param_tau_y2_we), .addr_a(param_sram_addr),
        .wdata_a(prog_param_value[3:0]), .rdata_a(param_tau_y2_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    wire param_tau_y3_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd21);
    wire [3:0] param_tau_y3_rdata;
    sram #(.DATA_WIDTH(4), .ADDR_WIDTH(NEURON_BITS)) tau_y3_cfg_mem (
        .clk(clk), .we_a(param_tau_y3_we), .addr_a(param_sram_addr),
        .wdata_a(prog_param_value[3:0]), .rdata_a(param_tau_y3_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    wire param_dend_thr1_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd8);
    wire signed [DATA_WIDTH-1:0] dend_thr_1_rdata;
    sram #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(NEURON_BITS)) dend_thr_1_mem (
        .clk(clk), .we_a(param_dend_thr1_we), .addr_a(param_sram_addr),
        .wdata_a(prog_param_value), .rdata_a(dend_thr_1_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    wire param_dend_thr2_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd9);
    wire signed [DATA_WIDTH-1:0] dend_thr_2_rdata;
    sram #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(NEURON_BITS)) dend_thr_2_mem (
        .clk(clk), .we_a(param_dend_thr2_we), .addr_a(param_sram_addr),
        .wdata_a(prog_param_value), .rdata_a(dend_thr_2_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    wire param_dend_thr3_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd10);
    wire signed [DATA_WIDTH-1:0] dend_thr_3_rdata;
    sram #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(NEURON_BITS)) dend_thr_3_mem (
        .clk(clk), .we_a(param_dend_thr3_we), .addr_a(param_sram_addr),
        .wdata_a(prog_param_value), .rdata_a(dend_thr_3_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    wire param_dend_parent_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd15);
    wire [5:0] dend_parent_rdata;
    sram #(.DATA_WIDTH(6), .ADDR_WIDTH(NEURON_BITS)) dend_parent_mem (
        .clk(clk), .we_a(param_dend_parent_we), .addr_a(param_sram_addr),
        .wdata_a(prog_param_value[5:0]), .rdata_a(dend_parent_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    wire parent_ptr_param_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd22);
    wire [NEURON_BITS-1:0] parent_ptr_rdata;
    sram #(.DATA_WIDTH(NEURON_BITS), .ADDR_WIDTH(NEURON_BITS),
           .INIT_VALUE({NEURON_BITS{1'b1}})) parent_ptr_mem (
        .clk(clk), .we_a(parent_ptr_param_we), .addr_a(param_sram_addr),
        .wdata_a(prog_param_value[NEURON_BITS-1:0]), .rdata_a(parent_ptr_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    wire joinop_param_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd23);
    wire [3:0] joinop_full_rdata;
    wire [1:0] joinop_rdata = joinop_full_rdata[1:0];
    wire [1:0] stackout_mode = joinop_full_rdata[3:2];
    sram #(.DATA_WIDTH(4), .ADDR_WIDTH(NEURON_BITS)) joinop_mem (
        .clk(clk), .we_a(joinop_param_we), .addr_a(param_sram_addr),
        .wdata_a(prog_param_value[3:0]), .rdata_a(joinop_full_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    wire is_root_param_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd24);
    wire is_root_rdata;
    sram #(.DATA_WIDTH(1), .ADDR_WIDTH(NEURON_BITS),
           .INIT_VALUE(1'b1)) is_root_mem (
        .clk(clk), .we_a(is_root_param_we), .addr_a(param_sram_addr),
        .wdata_a(prog_param_value[0]), .rdata_a(is_root_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    wire axon_type_param_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd25);
    wire [4:0] axon_type_rdata;
    reg [NEURON_BITS-1:0] axtype_rd_addr;
    sram #(.DATA_WIDTH(5), .ADDR_WIDTH(NEURON_BITS)) axon_type_mem (
        .clk(clk), .we_a(axon_type_param_we), .addr_a(param_sram_addr),
        .wdata_a(prog_param_value[4:0]), .rdata_a(),
        .addr_b(axtype_rd_addr), .rdata_b(axon_type_rdata)
    );

    wire axon_cfg_param_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd26);
    reg [11:0] axon_cfg_regs [0:31];
    wire [11:0] axon_cfg_rdata = axon_cfg_regs[axon_type_rdata];
    always @(posedge clk) begin
        if (axon_cfg_param_we)
            axon_cfg_regs[param_sram_addr[4:0]] <= prog_param_value[11:0];
    end

    wire param_trace_en_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd27);
    wire param_perf_reset_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd28);

    reg [7:0] epoch_interval;
    reg [7:0] epoch_counter;
    reg [3:0] num_updates;
    reg [3:0] update_pass;
    wire param_epoch_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd11);

    reg signed [DATA_WIDTH-1:0] reward_trace;
    reg [3:0] reward_tau;
    wire param_reward_tau_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd12);

    reg                    spike_ts_we;
    reg  [NEURON_BITS-1:0] spike_ts_addr;
    reg  [7:0]             spike_ts_wdata;
    wire [7:0]             spike_ts_rdata;
    sram #(.DATA_WIDTH(8), .ADDR_WIDTH(NEURON_BITS)) spike_ts_mem (
        .clk(clk), .we_a(spike_ts_we), .addr_a(spike_ts_addr),
        .wdata_a(spike_ts_wdata), .rdata_a(spike_ts_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );
    reg [7:0] timestep_within_epoch;

    wire signed [DATA_WIDTH-1:0] rt_decay_raw = reward_trace >>> reward_tau;
    wire signed [DATA_WIDTH-1:0] rt_decayed =
        (reward_trace == 0) ? 16'sd0 :
        (reward_trace > 0 && rt_decay_raw == 0) ? (reward_trace - 16'sd1) :
        (reward_trace < 0 && rt_decay_raw == 0) ? (reward_trace + 16'sd1) :
        (reward_trace - rt_decay_raw);

    wire param_homeo_target_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd13);
    wire [7:0] homeo_target_rdata;
    sram #(.DATA_WIDTH(8), .ADDR_WIDTH(NEURON_BITS)) homeo_target_mem (
        .clk(clk), .we_a(param_homeo_target_we), .addr_a(param_sram_addr),
        .wdata_a(prog_param_value[7:0]), .rdata_a(homeo_target_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    wire param_homeo_eta_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd14);
    wire [7:0] homeo_eta_rdata;
    sram #(.DATA_WIDTH(8), .ADDR_WIDTH(NEURON_BITS)) homeo_eta_mem (
        .clk(clk), .we_a(param_homeo_eta_we), .addr_a(param_sram_addr),
        .wdata_a(prog_param_value[7:0]), .rdata_a(homeo_eta_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    wire param_decay_v_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd16);
    wire [11:0] decay_v_rdata;
    sram #(.DATA_WIDTH(12), .ADDR_WIDTH(NEURON_BITS)) decay_v_mem (
        .clk(clk), .we_a(param_decay_v_we), .addr_a(param_sram_addr),
        .wdata_a(prog_param_value[11:0]), .rdata_a(decay_v_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    wire param_decay_u_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd17);
    wire [11:0] decay_u_rdata;
    sram #(.DATA_WIDTH(12), .ADDR_WIDTH(NEURON_BITS)) decay_u_mem (
        .clk(clk), .we_a(param_decay_u_we), .addr_a(param_sram_addr),
        .wdata_a(prog_param_value[11:0]), .rdata_a(decay_u_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    wire param_bias_cfg_we = (state == S_IDLE) && prog_param_we && (prog_param_id == 5'd18);
    wire [15:0] bias_cfg_rdata;
    sram #(.DATA_WIDTH(16), .ADDR_WIDTH(NEURON_BITS)) bias_cfg_mem (
        .clk(clk), .we_a(param_bias_cfg_we), .addr_a(param_sram_addr),
        .wdata_a(prog_param_value[15:0]), .rdata_a(bias_cfg_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    wire signed [12:0] bias_mant = $signed(bias_cfg_rdata[15:3]);
    wire [2:0] bias_exp  = bias_cfg_rdata[2:0];
    wire signed [NEURON_WIDTH-1:0] bias_scaled =
        ($signed({{(NEURON_WIDTH-13){bias_mant[12]}}, bias_mant}) << bias_exp);
    wire cuba_enabled = (decay_v_rdata != 12'd0) || (decay_u_rdata != 12'd0) || (bias_cfg_rdata != 16'd0);

    wire signed [NEURON_WIDTH+11:0] v_decay_product = nrn_rdata * $signed({1'b0, decay_v_rdata});
    wire signed [NEURON_WIDTH-1:0]  v_decay_step = (decay_v_rdata == 12'd0) ? {NEURON_WIDTH{1'b0}} :
                                                    raz_div4096(v_decay_product);

    wire signed [NEURON_WIDTH+11:0] u_decay_product = cur_rdata * $signed({1'b0, decay_u_rdata});
    wire signed [NEURON_WIDTH-1:0]  u_decay_step = (decay_u_rdata == 12'd0) ? {NEURON_WIDTH{1'b0}} :
                                                    raz_div4096(u_decay_product);

    reg                    spike_cnt_we;
    reg  [NEURON_BITS-1:0] spike_cnt_addr;
    reg  [7:0]             spike_cnt_wdata;
    wire [7:0]             spike_cnt_rdata;
    sram #(.DATA_WIDTH(8), .ADDR_WIDTH(NEURON_BITS)) spike_count_mem (
        .clk(clk), .we_a(spike_cnt_we), .addr_a(spike_cnt_addr),
        .wdata_a(spike_cnt_wdata), .rdata_a(spike_cnt_rdata),
        .addr_b({NEURON_BITS{1'b0}}), .rdata_b()
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            epoch_interval <= 8'd1;
            reward_tau     <= 4'd4;
            num_updates    <= 4'd1;
        end else begin
            if (param_epoch_we) begin
                epoch_interval <= prog_param_value[7:0];
                num_updates <= (prog_param_value[15:12] == 4'd0) ? 4'd1 : prog_param_value[15:12];
            end
            if (param_reward_tau_we) reward_tau <= prog_param_value[3:0];
        end
    end

    reg [15:0] lfsr;
    wire lfsr_feedback = lfsr[0];
    wire [15:0] lfsr_next = {lfsr_feedback, lfsr[15:1]} ^
                             (lfsr_feedback ? 16'hB400 : 16'h0000);

    wire [3:0] noise_mant = param_noise_rdata[3:0];
    wire [4:0] noise_exp  = param_noise_rdata[8:4];
    wire [31:0] noise_mask_wide = ({28'b0, noise_mant} << noise_exp);
    wire [DATA_WIDTH-1:0] noise_mask = (|noise_mask_wide[31:DATA_WIDTH]) ?
        {DATA_WIDTH{1'b1}} : noise_mask_wide[DATA_WIDTH-1:0];
    wire signed [DATA_WIDTH-1:0] noise_value =
        $signed({1'b0, lfsr[DATA_WIDTH-2:0] & noise_mask[DATA_WIDTH-2:0]}) -
        $signed({1'b0, noise_mask[DATA_WIDTH-1:1]});
    wire signed [NEURON_WIDTH-1:0] effective_threshold =
        (noise_enable && param_noise_target_rdata == 2'd0) ? (param_thr_rdata + $signed(noise_value)) : param_thr_rdata;
    wire signed [NEURON_WIDTH-1:0] noise_v_offset =
        (noise_enable && param_noise_target_rdata == 2'd1) ?
        $signed({{(NEURON_WIDTH-DATA_WIDTH){noise_value[DATA_WIDTH-1]}}, noise_value}) : {NEURON_WIDTH{1'b0}};
    wire signed [NEURON_WIDTH-1:0] noise_u_offset =
        (noise_enable && param_noise_target_rdata == 2'd2) ?
        $signed({{(NEURON_WIDTH-DATA_WIDTH){noise_value[DATA_WIDTH-1]}}, noise_value}) : {NEURON_WIDTH{1'b0}};

    wire signed [NEURON_WIDTH-1:0] vmin_ext = $signed({{(NEURON_WIDTH-DATA_WIDTH){param_vmin_rdata[DATA_WIDTH-1]}}, param_vmin_rdata});
    wire signed [NEURON_WIDTH-1:0] vmax_ext = $signed({{(NEURON_WIDTH-DATA_WIDTH){param_vmax_rdata[DATA_WIDTH-1]}}, param_vmax_rdata});

    wire [7:0] tau1_mask = (param_tau1_rdata == 4'd0) ? 8'd0 : ((8'd1 << param_tau1_rdata) - 8'd1);
    wire [7:0] trace1_frac = trace_rdata & tau1_mask;
    wire trace1_stoch_up = (param_tau1_rdata != 4'd0) && (trace1_frac != 8'd0) &&
                           ((lfsr[7:0] & tau1_mask) < trace1_frac);
    wire [7:0] trace1_decay_step = (trace_rdata >> param_tau1_rdata) + {7'd0, trace1_stoch_up};
    wire [7:0] trace1_decay_val = (trace_rdata == 8'd0) ? 8'd0 :
        (trace1_decay_step == 8'd0) ? (trace_rdata - 8'd1) :
        (trace_rdata - trace1_decay_step);

    wire [7:0] tau2_mask = (param_tau2_rdata == 4'd0) ? 8'd0 : ((8'd1 << param_tau2_rdata) - 8'd1);
    wire [7:0] trace2_frac = trace2_rdata & tau2_mask;
    wire trace2_stoch_up = (param_tau2_rdata != 4'd0) && (trace2_frac != 8'd0) &&
                           ((lfsr[15:8] & tau2_mask) < trace2_frac);
    wire [7:0] trace2_decay_step = (trace2_rdata >> param_tau2_rdata) + {7'd0, trace2_stoch_up};
    wire [7:0] trace2_decay_val = (trace2_rdata == 8'd0) ? 8'd0 :
        (trace2_decay_step == 8'd0) ? (trace2_rdata - 8'd1) :
        (trace2_rdata - trace2_decay_step);

    wire [7:0] taux2_mask = (param_tau_x2_rdata == 4'd0) ? 8'd0 : ((8'd1 << param_tau_x2_rdata) - 8'd1);
    wire [7:0] x2_frac = x2_trace_rdata & taux2_mask;
    wire x2_stoch_up = (param_tau_x2_rdata != 4'd0) && (x2_frac != 8'd0) &&
                       ((lfsr[7:0] ^ lfsr[15:8] & taux2_mask) < x2_frac);
    wire [7:0] x2_decay_step = (x2_trace_rdata >> param_tau_x2_rdata) + {7'd0, x2_stoch_up};
    wire [7:0] x2_decay_val = (x2_trace_rdata == 8'd0) ? 8'd0 :
        (x2_decay_step == 8'd0) ? (x2_trace_rdata - 8'd1) :
        (x2_trace_rdata - x2_decay_step);

    wire [7:0] tauy2_mask = (param_tau_y2_rdata == 4'd0) ? 8'd0 : ((8'd1 << param_tau_y2_rdata) - 8'd1);
    wire [7:0] y2_frac = y2_trace_rdata & tauy2_mask;
    wire y2_stoch_up = (param_tau_y2_rdata != 4'd0) && (y2_frac != 8'd0) &&
                       ({lfsr[3:0], lfsr[15:12]} & tauy2_mask) < y2_frac;
    wire [7:0] y2_decay_step = (y2_trace_rdata >> param_tau_y2_rdata) + {7'd0, y2_stoch_up};
    wire [7:0] y2_decay_val = (y2_trace_rdata == 8'd0) ? 8'd0 :
        (y2_decay_step == 8'd0) ? (y2_trace_rdata - 8'd1) :
        (y2_trace_rdata - y2_decay_step);

    wire [7:0] tauy3_mask = (param_tau_y3_rdata == 4'd0) ? 8'd0 : ((8'd1 << param_tau_y3_rdata) - 8'd1);
    wire [7:0] y3_frac = y3_trace_rdata & tauy3_mask;
    wire y3_stoch_up = (param_tau_y3_rdata != 4'd0) && (y3_frac != 8'd0) &&
                       ({lfsr[11:8], lfsr[7:4]} & tauy3_mask) < y3_frac;
    wire [7:0] y3_decay_step = (y3_trace_rdata >> param_tau_y3_rdata) + {7'd0, y3_stoch_up};
    wire [7:0] y3_decay_val = (y3_trace_rdata == 8'd0) ? 8'd0 :
        (y3_decay_step == 8'd0) ? (y3_trace_rdata - 8'd1) :
        (y3_trace_rdata - y3_decay_step);

    integer pi;
    initial begin
        for (pi = 0; pi < 32; pi = pi + 1) begin
            axon_cfg_regs[pi] = 12'd0;
        end
    end

    localparam FIFO_WIDTH = NEURON_BITS + 8;
    reg fifo_sel;

    reg                    fifo_a_push, fifo_a_pop, fifo_a_clear;
    reg [FIFO_WIDTH-1:0]  fifo_a_push_data_reg;
    wire [FIFO_WIDTH-1:0] fifo_a_pop_data;
    wire                   fifo_a_empty, fifo_a_full;

    spike_fifo #(.ID_WIDTH(FIFO_WIDTH), .DEPTH(64), .PTR_BITS(6)) fifo_a (
        .clk(clk), .rst_n(rst_n), .clear(fifo_a_clear),
        .push(fifo_a_push), .push_data(fifo_a_push_data_reg),
        .pop(fifo_a_pop), .pop_data(fifo_a_pop_data),
        .empty(fifo_a_empty), .full(fifo_a_full), .count()
    );

    reg                    fifo_b_push, fifo_b_pop, fifo_b_clear;
    reg [FIFO_WIDTH-1:0]  fifo_b_push_data_reg;
    wire [FIFO_WIDTH-1:0] fifo_b_pop_data;
    wire                   fifo_b_empty, fifo_b_full;

    spike_fifo #(.ID_WIDTH(FIFO_WIDTH), .DEPTH(64), .PTR_BITS(6)) fifo_b (
        .clk(clk), .rst_n(rst_n), .clear(fifo_b_clear),
        .push(fifo_b_push), .push_data(fifo_b_push_data_reg),
        .pop(fifo_b_pop), .pop_data(fifo_b_pop_data),
        .empty(fifo_b_empty), .full(fifo_b_full), .count()
    );

    wire                   prev_fifo_empty = fifo_sel ? fifo_b_empty : fifo_a_empty;
    wire [FIFO_WIDTH-1:0]  prev_fifo_data  = fifo_sel ? fifo_b_pop_data : fifo_a_pop_data;
    wire                   curr_fifo_full  = fifo_sel ? fifo_a_full : fifo_b_full;

    reg [NEURON_BITS:0]            proc_neuron;
    reg [NEURON_BITS-1:0]          curr_spike_src;
    reg [7:0]                      curr_spike_payload;
    reg [POOL_ADDR_BITS-1:0]       curr_base_addr;
    reg [COUNT_BITS-1:0]           curr_count;
    reg [COUNT_BITS-1:0]           conn_idx;
    reg signed [NEURON_WIDTH-1:0]  proc_potential;
    reg signed [NEURON_WIDTH-1:0]  proc_current;
    reg [7:0]                      proc_refrac;
    reg signed [DATA_WIDTH-1:0]    proc_input;

    reg [NEURON_BITS-1:0]          saved_target;
    reg signed [DATA_WIDTH-1:0]    saved_weight;
    reg [COMPARTMENT_BITS-1:0]     saved_comp;

    reg                            proc_spiked_this_neuron;
    reg signed [NEURON_WIDTH-1:0]  spike_contribution;
    reg [NEURON_BITS-1:0]          saved_parent_ptr;

    reg [1:0]                      curr_format;
    reg [NEURON_BITS-1:0]          base_target;
    reg signed [DATA_WIDTH-1:0]    shared_weight;
    reg [COMPARTMENT_BITS-1:0]     shared_comp;

    reg                            pack_active;
    reg [3:0]                      pack_shift;
    reg [3:0]                      pack_nwb;

    reg [POOL_ADDR_BITS:0]         pool_used_count;
    reg [POOL_ADDR_BITS:0]         elig_scan_addr;

    reg                            learn_mode;
    reg [NEURON_BITS:0]            learn_neuron;
    reg [COUNT_BITS-1:0]           learn_slot;
    reg [POOL_ADDR_BITS-1:0]       learn_base_addr;
    reg [COUNT_BITS-1:0]           learn_count;
    reg                            learn_rev_valid;
    reg [NEURON_BITS-1:0]          learn_rev_src;
    reg [POOL_ADDR_BITS-1:0]       learn_rev_pool_addr;
    reg [NUM_NEURONS-1:0]          spike_bitmap;

    wire [3:0]  mc_opcode = ucode_rdata[31:28];
    wire [3:0]  mc_dst    = ucode_rdata[27:24];
    wire [3:0]  mc_src_a  = ucode_rdata[23:20];
    wire [3:0]  mc_src_b  = ucode_rdata[19:16];
    wire [2:0]  mc_shift  = ucode_rdata[15:13];
    wire signed [15:0] mc_imm = ucode_rdata[15:0];

    wire signed [DATA_WIDTH-1:0] mc_op_a = mc_regs[mc_src_a];
    wire signed [DATA_WIDTH-1:0] mc_op_b = mc_regs[mc_src_b];
    wire signed [31:0] mc_mul_raw = mc_op_a * mc_op_b;

    reg signed [DATA_WIDTH-1:0] mc_alu_result;
    always @(*) begin
        case (mc_opcode)
            4'd1:    mc_alu_result = mc_op_a + mc_op_b;
            4'd2:    mc_alu_result = mc_op_a - mc_op_b;
            4'd3:    mc_alu_result = mc_mul_raw >>> mc_shift;
            4'd4:    mc_alu_result = mc_op_a >>> mc_shift;
            4'd5:    mc_alu_result = mc_op_a << mc_shift;
            4'd6:    mc_alu_result = (mc_op_a > mc_op_b) ? mc_op_a : mc_op_b;
            4'd7:    mc_alu_result = (mc_op_a < mc_op_b) ? mc_op_a : mc_op_b;
            4'd8:    mc_alu_result = mc_imm;
            default: mc_alu_result = 16'sd0;
        endcase
    end

    wire [POOL_ADDR_BITS-1:0] learn_wr_addr =
        (learn_mode == 0) ? (learn_base_addr + learn_slot) : learn_rev_pool_addr;

    wire signed [31:0] reward_product = $signed(elig_rdata) * $signed(reward_trace);
    wire signed [DATA_WIDTH-1:0] reward_delta = reward_product >>> REWARD_SHIFT;
    wire signed [DATA_WIDTH-1:0] elig_new_wt_raw = pool_wt_rdata + reward_delta;
    wire signed [DATA_WIDTH-1:0] elig_new_wt =
        (elig_new_wt_raw > WEIGHT_MAX) ? WEIGHT_MAX :
        (elig_new_wt_raw < WEIGHT_MIN) ? WEIGHT_MIN :
        elig_new_wt_raw;

    wire signed [DATA_WIDTH-1:0] elig_decay_step = elig_rdata >>> ELIG_DECAY_SHIFT;
    wire signed [DATA_WIDTH-1:0] elig_decayed =
        (elig_rdata > 0 && elig_decay_step == 0) ? elig_rdata - 16'sd1 :
        elig_rdata - elig_decay_step;

    wire [1:0] dend_parent1 = dend_parent_rdata[1:0];
    wire [1:0] dend_parent2 = dend_parent_rdata[3:2];
    wire [1:0] dend_parent3 = dend_parent_rdata[5:4];

    wire signed [DATA_WIDTH-1:0] tree_out3 =
        (dend_acc_3_rdata > dend_thr_3_rdata) ? (dend_acc_3_rdata - dend_thr_3_rdata) : 16'sd0;

    wire signed [DATA_WIDTH-1:0] tree_in2 = dend_acc_2_rdata +
        ((dend_parent3 == 2'd2) ? tree_out3 : 16'sd0);
    wire signed [DATA_WIDTH-1:0] tree_out2 =
        (tree_in2 > dend_thr_2_rdata) ? (tree_in2 - dend_thr_2_rdata) : 16'sd0;

    wire signed [DATA_WIDTH-1:0] tree_in1 = dend_acc_1_rdata +
        ((dend_parent2 == 2'd1) ? tree_out2 : 16'sd0) +
        ((dend_parent3 == 2'd1) ? tree_out3 : 16'sd0);
    wire signed [DATA_WIDTH-1:0] tree_out1 =
        (tree_in1 > dend_thr_1_rdata) ? (tree_in1 - dend_thr_1_rdata) : 16'sd0;

    wire signed [DATA_WIDTH-1:0] total_dend =
        ((dend_parent1 == 2'd0) ? tree_out1 : 16'sd0) +
        ((dend_parent2 == 2'd0) ? tree_out2 : 16'sd0) +
        ((dend_parent3 == 2'd0) ? tree_out3 : 16'sd0);

    wire signed [NEURON_WIDTH-1:0] total_input = dendritic_enable ?
        (acc_rdata + $signed(total_dend)) : acc_rdata;

    wire signed [NEURON_WIDTH+11:0] scale_u_product = total_input * $signed({1'b0, decay_u_rdata});
    wire signed [NEURON_WIDTH-1:0]  scaled_total_input = scale_u_enable ?
        raz_div4096(scale_u_product) : total_input;

    wire signed [NEURON_WIDTH-1:0] spike_excess = $signed(nrn_rdata[DATA_WIDTH-1:0]) + total_input - $signed(param_leak_rdata) - effective_threshold;
    wire [7:0] spike_payload_val = (spike_excess > 16'sd255) ? 8'd255 :
                                   (spike_excess < 16'sd1)   ? 8'd1   : spike_excess[7:0];

    wire signed [31:0] graded_weight_ext  = saved_weight;
    wire signed [31:0] graded_payload_ext = {24'd0, curr_spike_payload};
    wire signed [31:0] graded_product     = graded_weight_ext * graded_payload_ext;
    wire signed [DATA_WIDTH-1:0] graded_current = graded_product >>> GRADE_SHIFT;

    wire [NEURON_BITS-1:0] deliver_target =
        (curr_format == FMT_SPARSE) ? pool_tgt_rdata :
        (conn_idx == 0)             ? pool_tgt_rdata :
        (base_target + conn_idx);

    wire signed [DATA_WIDTH-1:0] deliver_weight =
        (curr_format == FMT_POP && conn_idx != 0) ? shared_weight : pool_wt_rdata;

    wire [COMPARTMENT_BITS-1:0] deliver_comp =
        (curr_format == FMT_POP && conn_idx != 0) ? shared_comp : pool_comp_rdata;

    reg                        ext_pending;
    reg [NEURON_BITS-1:0]      ext_buf_id;
    reg signed [DATA_WIDTH-1:0] ext_buf_current;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            ext_pending <= 0;
        else if (ext_valid) begin
            ext_pending    <= 1;
            ext_buf_id     <= ext_neuron_id;
            ext_buf_current <= ext_current;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            fifo_sel        <= 0;
            timestep_done   <= 0;
            spike_out_valid <= 0;
            spike_out_payload <= 0;
            total_spikes    <= 0;
            timestep_count  <= 0;
            proc_neuron     <= 0;
            conn_idx        <= 0;
            curr_spike_payload <= 0;
            nrn_we <= 0; ref_we <= 0; acc_we <= 0; cur_we <= 0;
            pool_wt_we_r <= 0; trace_we <= 0; trace2_we <= 0;
            x2_trace_we <= 0; y2_trace_we <= 0; y3_trace_we <= 0;
            pool_tag_we_r <= 0; pool_delay_we_learn <= 0;
            dend_acc_1_we <= 0; dend_acc_2_we <= 0; dend_acc_3_we <= 0;
            fifo_a_push <= 0; fifo_a_pop <= 0; fifo_a_clear <= 0;
            fifo_b_push <= 0; fifo_b_pop <= 0; fifo_b_clear <= 0;
            proc_current    <= 0;
            spike_bitmap    <= 0;
            learn_mode      <= 0;
            learn_neuron    <= 0;
            learn_slot      <= 0;
            learn_rev_valid <= 0;
            learn_rev_src   <= 0;
            learn_rev_pool_addr <= 0;
            rev_addr        <= 0;
            saved_comp      <= 0;
            curr_format     <= 0;
            base_target     <= 0;
            shared_weight   <= 0;
            shared_comp     <= 0;
            pack_active     <= 0;
            pack_shift      <= 0;
            pack_nwb        <= 0;
            elig_we         <= 0;
            elig_addr       <= 0;
            elig_wdata      <= 0;
            elig_scan_addr  <= 0;
            pool_used_count <= 0;
            lfsr            <= NOISE_LFSR_SEED;
            mc_pc           <= 0;
            elig_phase      <= 0;
            mc_regs[0] <= 0; mc_regs[1] <= 0; mc_regs[2] <= 0; mc_regs[3] <= 0;
            mc_regs[4] <= 0; mc_regs[5] <= 0; mc_regs[6] <= 0; mc_regs[7] <= 0;
            mc_regs[8] <= 0; mc_regs[9] <= 0; mc_regs[10] <= 0; mc_regs[11] <= 0;
            mc_regs[12] <= 0; mc_regs[13] <= 0; mc_regs[14] <= 0; mc_regs[15] <= 0;
            pool_addr_r     <= 0;
            pool_wt_wr_addr <= 0;
            pool_wt_wr_data <= 0;
            index_rd_addr   <= 0;
            curr_base_addr  <= 0;
            curr_count      <= 0;
            learn_base_addr <= 0;
            learn_count     <= 0;
            dq_we           <= 0;
            dq_addr         <= 0;
            dq_wdata        <= 0;
            current_ts_mod64 <= 0;
            drain_cnt       <= 0;
            drain_idx       <= 0;
            dq_cap_target   <= 0;
            dq_cap_current  <= 0;
            dq_cap_comp     <= 0;
            proc_spiked_this_neuron <= 0;
            spike_contribution      <= 0;
            saved_parent_ptr        <= 0;
            was_idle        <= 1;
            any_spike_this_ts <= 0;
            epoch_counter   <= 0;
            reward_trace    <= 0;
            spike_cnt_we    <= 0;
            spike_cnt_addr  <= 0;
            spike_cnt_wdata <= 0;
            homeo_thr_we    <= 0;
            homeo_thr_wdata <= 0;
            axtype_rd_addr  <= 0;
            spike_ts_we     <= 0;
            spike_ts_addr   <= 0;
            spike_ts_wdata  <= 0;
            update_pass     <= 0;
            timestep_within_epoch <= 0;
            perf_spike_count   <= 0;
            perf_active_cycles <= 0;
            perf_synaptic_ops  <= 0;
            trace_fifo_enable  <= 0;
            trace_wr_ptr       <= 0;
            trace_rd_ptr       <= 0;
            trace_last_popped  <= 0;
        end else begin
            nrn_we <= 0; ref_we <= 0; acc_we <= 0; cur_we <= 0;
            pool_wt_we_r <= 0; trace_we <= 0; trace2_we <= 0; elig_we <= 0;
            x2_trace_we <= 0; y2_trace_we <= 0; y3_trace_we <= 0;
            pool_tag_we_r <= 0; pool_delay_we_learn <= 0;
            dq_we <= 0;
            dend_acc_1_we <= 0; dend_acc_2_we <= 0; dend_acc_3_we <= 0;
            spike_cnt_we <= 0; homeo_thr_we <= 0; spike_ts_we <= 0;
            timestep_done <= 0;
            spike_out_valid <= 0;
            fifo_a_push <= 0; fifo_a_pop <= 0; fifo_a_clear <= 0;
            fifo_b_push <= 0; fifo_b_pop <= 0; fifo_b_clear <= 0;

            if (state != S_IDLE)
                perf_active_cycles <= perf_active_cycles + 1;

            if (param_trace_en_we)
                trace_fifo_enable <= prog_param_value[0];
            if (param_perf_reset_we) begin
                perf_spike_count   <= 0;
                perf_active_cycles <= 0;
                perf_synaptic_ops  <= 0;
            end

            if (probe_active_r && probe_sid_r == 5'd22 && !trace_fifo_empty) begin
                trace_last_popped <= trace_fifo_mem[trace_rd_ptr[5:0]];
                trace_rd_ptr <= trace_rd_ptr + 1;
            end

            if (state == S_IDLE && pool_we) begin
                rev_count[pool_target_in] <= rev_count[pool_target_in] + 1;
                if ({1'b0, pool_addr_in} + 1 > pool_used_count)
                    pool_used_count <= {1'b0, pool_addr_in} + 1;
            end

            case (state)
                S_IDLE: begin
                    if (ext_valid) begin
                        acc_we    <= 1;
                        acc_addr  <= ext_neuron_id;
                        acc_wdata <= ext_current;
                    end
                    if (start) begin
                        any_spike_this_ts <= 0;
                        update_pass <= 0;
                        state <= S_DELAY_DRAIN_INIT;
                    end
                end

                S_DELIVER_POP: begin
                    if (prev_fifo_empty) begin
                        state       <= S_UPDATE_INIT;
                        proc_neuron <= 0;
                    end else begin
                        curr_spike_src     <= prev_fifo_data[FIFO_WIDTH-1:8];
                        curr_spike_payload <= prev_fifo_data[7:0];
                        if (fifo_sel)
                            fifo_b_pop <= 1;
                        else
                            fifo_a_pop <= 1;
                        index_rd_addr <= prev_fifo_data[FIFO_WIDTH-1:8];
                        axtype_rd_addr <= prev_fifo_data[FIFO_WIDTH-1:8];
                        state <= S_DELIVER_IDX_WAIT;
                    end
                end

                S_DELIVER_IDX_WAIT: begin
                    state <= S_DELIVER_IDX_READ;
                end

                S_DELIVER_IDX_READ: begin
                    curr_format    <= index_rdata[INDEX_WIDTH-1 -: 2];
                    curr_base_addr <= index_rdata[COUNT_BITS +: POOL_ADDR_BITS];
                    curr_count     <= index_rdata[COUNT_BITS-1:0];
                    conn_idx       <= 0;
                    if (index_rdata[INDEX_WIDTH-1 -: 2] == FMT_DENSE &&
                        axon_cfg_rdata[0] == 1'b1) begin
                        pack_active <= 1;
                        pack_nwb    <= axon_cfg_rdata[11:8];
                        case (axon_cfg_rdata[11:8])
                            4'd1:    pack_shift <= 4'd4;
                            4'd2:    pack_shift <= 4'd3;
                            4'd4:    pack_shift <= 4'd2;
                            4'd8:    pack_shift <= 4'd1;
                            default: pack_active <= 0;
                        endcase
                    end else begin
                        pack_active <= 0;
                    end
                    if (index_rdata[COUNT_BITS-1:0] == 0) begin
                        state <= S_DELIVER_POP;
                    end else begin
                        pool_addr_r <= index_rdata[COUNT_BITS +: POOL_ADDR_BITS];
                        state <= S_DELIVER_POOL_WAIT;
                    end
                end

                S_DELIVER_POOL_WAIT: begin
                    state <= S_DELIVER_ADDR;
                end

                S_DELIVER_ADDR: begin
                    saved_target <= deliver_target;
                    saved_comp   <= deliver_comp;
                    if (pack_active) begin : pack_extract
                        reg [3:0] p_sub;
                        reg [6:0] p_off;
                        case (pack_shift)
                            4'd4: p_sub = conn_idx[3:0];
                            4'd3: p_sub = conn_idx[2:0];
                            4'd2: p_sub = conn_idx[1:0];
                            4'd1: p_sub = conn_idx[0:0];
                            default: p_sub = 0;
                        endcase
                        p_off = p_sub * pack_nwb;
                        saved_weight <= (deliver_weight >> p_off);
                    end else begin
                        saved_weight <= deliver_weight;
                    end
                    acc_addr        <= deliver_target;
                    dend_acc_1_addr <= deliver_target;
                    dend_acc_2_addr <= deliver_target;
                    dend_acc_3_addr <= deliver_target;
                    axtype_rd_addr <= deliver_target;
                    if (conn_idx == 0 && curr_format != FMT_SPARSE)
                        base_target <= pool_tgt_rdata;
                    if (conn_idx == 0 && curr_format == FMT_POP) begin
                        shared_weight <= pool_wt_rdata;
                        shared_comp   <= pool_comp_rdata;
                    end
                    state           <= S_DELIVER_ACC_WAIT;
                end

                S_DELIVER_ACC_WAIT: begin
                    state <= S_DELIVER_AXTYPE;
                end

                S_DELIVER_AXTYPE: begin
                    if (axon_cfg_rdata[11:8] != 4'd0) begin
                        begin: axtype_decompress
                            reg [3:0] nwb;
                            reg signed [3:0] wexp_s;
                            reg is_exc;
                            reg is_mixed;
                            reg signed [DATA_WIDTH-1:0] raw, shifted;
                            reg sign_bit;
                            reg signed [DATA_WIDTH-1:0] magnitude;
                            nwb = axon_cfg_rdata[11:8];
                            wexp_s = $signed(axon_cfg_rdata[7:4]);
                            is_exc = axon_cfg_rdata[2];
                            is_mixed = axon_cfg_rdata[1];
                            case (nwb)
                                4'd1:  raw = saved_weight & 16'h0001;
                                4'd2:  raw = saved_weight & 16'h0003;
                                4'd3:  raw = saved_weight & 16'h0007;
                                4'd4:  raw = saved_weight & 16'h000F;
                                4'd5:  raw = saved_weight & 16'h001F;
                                4'd6:  raw = saved_weight & 16'h003F;
                                4'd7:  raw = saved_weight & 16'h007F;
                                4'd8:  raw = saved_weight & 16'h00FF;
                                4'd9:  raw = saved_weight & 16'h01FF;
                                default: raw = saved_weight;
                            endcase

                            if (is_mixed && nwb > 1) begin
                                sign_bit = raw[nwb-1];
                                case (nwb)
                                    4'd2:  magnitude = raw & 16'h0001;
                                    4'd3:  magnitude = raw & 16'h0003;
                                    4'd4:  magnitude = raw & 16'h0007;
                                    4'd5:  magnitude = raw & 16'h000F;
                                    4'd6:  magnitude = raw & 16'h001F;
                                    4'd7:  magnitude = raw & 16'h003F;
                                    4'd8:  magnitude = raw & 16'h007F;
                                    4'd9:  magnitude = raw & 16'h00FF;
                                    default: magnitude = raw;
                                endcase
                                if (wexp_s >= 0)
                                    shifted = magnitude << wexp_s;
                                else
                                    shifted = magnitude >>> (-wexp_s);
                                saved_weight <= sign_bit ? (-shifted) : shifted;
                            end else begin
                                if (wexp_s >= 0)
                                    shifted = raw << wexp_s;
                                else
                                    shifted = raw >>> (-wexp_s);
                                saved_weight <= is_exc ? (-shifted) : shifted;
                            end
                        end
                    end
                    state <= S_DELIVER_ACC;
                end

                S_DELIVER_ACC: begin
                    perf_synaptic_ops <= perf_synaptic_ops + 1;
                    if (pool_delay_rdata != 0 &&
                        delay_count[delivery_ts] < DELAY_ENTRIES_PER_TS) begin
                        dq_we    <= 1;
                        dq_addr  <= {delivery_ts, delay_count[delivery_ts][DELAY_ENTRY_BITS-1:0]};
                        dq_wdata <= {saved_target, delivered_current, saved_comp};
                        delay_count[delivery_ts] <= delay_count[delivery_ts] + 1;
                    end else begin
                        case (saved_comp)
                            2'd0: begin
                                acc_we    <= 1;
                                acc_addr  <= saved_target;
                                acc_wdata <= graded_enable ?
                                    (acc_rdata + graded_current) :
                                    (acc_rdata + saved_weight);
                            end
                            2'd1: begin
                                dend_acc_1_we    <= 1;
                                dend_acc_1_addr  <= saved_target;
                                dend_acc_1_wdata <= graded_enable ?
                                    (dend_acc_1_rdata + graded_current) :
                                    (dend_acc_1_rdata + saved_weight);
                            end
                            2'd2: begin
                                dend_acc_2_we    <= 1;
                                dend_acc_2_addr  <= saved_target;
                                dend_acc_2_wdata <= graded_enable ?
                                    (dend_acc_2_rdata + graded_current) :
                                    (dend_acc_2_rdata + saved_weight);
                            end
                            2'd3: begin
                                dend_acc_3_we    <= 1;
                                dend_acc_3_addr  <= saved_target;
                                dend_acc_3_wdata <= graded_enable ?
                                    (dend_acc_3_rdata + graded_current) :
                                    (dend_acc_3_rdata + saved_weight);
                            end
                        endcase
                    end
                    state <= S_DELIVER_NEXT;
                end

                S_DELIVER_NEXT: begin
                    if (conn_idx < curr_count - 1) begin
                        conn_idx <= conn_idx + 1;
                        if (curr_format == FMT_POP) begin
                            state <= S_DELIVER_ADDR;
                        end else begin
                            if (pack_active)
                                pool_addr_r <= curr_base_addr + ((conn_idx + 1) >> pack_shift);
                            else
                                pool_addr_r <= pool_addr_r + 1;
                            state       <= S_DELIVER_POOL_WAIT;
                        end
                    end else begin
                        state <= S_DELIVER_POP;
                    end
                end

                S_UPDATE_INIT: begin
                    nrn_addr        <= proc_neuron[NEURON_BITS-1:0];
                    cur_addr        <= proc_neuron[NEURON_BITS-1:0];
                    ref_addr        <= proc_neuron[NEURON_BITS-1:0];
                    acc_addr        <= proc_neuron[NEURON_BITS-1:0];
                    trace_addr      <= proc_neuron[NEURON_BITS-1:0];
                    trace2_addr     <= proc_neuron[NEURON_BITS-1:0];
                    x2_trace_addr   <= proc_neuron[NEURON_BITS-1:0];
                    y2_trace_addr   <= proc_neuron[NEURON_BITS-1:0];
                    y3_trace_addr   <= proc_neuron[NEURON_BITS-1:0];
                    dend_acc_1_addr <= proc_neuron[NEURON_BITS-1:0];
                    dend_acc_2_addr <= proc_neuron[NEURON_BITS-1:0];
                    dend_acc_3_addr <= proc_neuron[NEURON_BITS-1:0];
                    spike_cnt_addr  <= proc_neuron[NEURON_BITS-1:0];
                    state           <= S_UPDATE_READ;
                end

                S_UPDATE_READ: begin
                    nrn_addr        <= proc_neuron[NEURON_BITS-1:0];
                    cur_addr        <= proc_neuron[NEURON_BITS-1:0];
                    ref_addr        <= proc_neuron[NEURON_BITS-1:0];
                    acc_addr        <= proc_neuron[NEURON_BITS-1:0];
                    trace_addr      <= proc_neuron[NEURON_BITS-1:0];
                    trace2_addr     <= proc_neuron[NEURON_BITS-1:0];
                    x2_trace_addr   <= proc_neuron[NEURON_BITS-1:0];
                    y2_trace_addr   <= proc_neuron[NEURON_BITS-1:0];
                    y3_trace_addr   <= proc_neuron[NEURON_BITS-1:0];
                    dend_acc_1_addr <= proc_neuron[NEURON_BITS-1:0];
                    dend_acc_2_addr <= proc_neuron[NEURON_BITS-1:0];
                    dend_acc_3_addr <= proc_neuron[NEURON_BITS-1:0];
                    spike_cnt_addr  <= proc_neuron[NEURON_BITS-1:0];
                    state           <= S_UPDATE_CALC;
                end

                S_UPDATE_CALC: begin
                    proc_refrac   <= ref_rdata;
                    proc_input    <= total_input;
                    proc_spiked_this_neuron <= 0;

                    lfsr <= lfsr_next;

                    if (cuba_enabled) begin
                        proc_current <= cur_rdata - u_decay_step + scaled_total_input + noise_u_offset;
                        if (ref_rdata > 0) begin
                            proc_refrac <= ref_rdata - 1;
                            if (refrac_mode_rel) begin
                                proc_potential <= nrn_rdata - v_decay_step - bias_scaled + noise_v_offset;
                            end else begin
                                proc_potential <= $signed({{(NEURON_WIDTH-DATA_WIDTH){param_rest_rdata[DATA_WIDTH-1]}}, param_rest_rdata});
                            end
                            trace_wdata  <= trace1_decay_val;
                            trace2_wdata <= trace2_decay_val;
                            x2_trace_wdata <= x2_decay_val;
                            y2_trace_wdata <= y2_decay_val;
                            y3_trace_wdata <= y3_decay_val;
                        end else begin
                            proc_potential <= nrn_rdata - v_decay_step + cur_rdata + bias_scaled + noise_v_offset;
                            if (nrn_rdata - v_decay_step + cur_rdata + bias_scaled + noise_v_offset >= effective_threshold) begin
                                proc_potential <= $signed({{(NEURON_WIDTH-DATA_WIDTH){param_rest_rdata[DATA_WIDTH-1]}}, param_rest_rdata});
                                proc_refrac    <= param_refrac_rdata[7:0];
                                trace_wdata    <= TRACE_MAX;
                                trace2_wdata   <= TRACE_MAX;
                                x2_trace_wdata <= TRACE_MAX;
                                y2_trace_wdata <= TRACE_MAX;
                                y3_trace_wdata <= TRACE_MAX;
                                spike_bitmap[proc_neuron[NEURON_BITS-1:0]] <= 1;
                                any_spike_this_ts <= 1;
                                proc_spiked_this_neuron <= 1;
                                spike_ts_we    <= 1;
                                spike_ts_addr  <= proc_neuron[NEURON_BITS-1:0];
                                spike_ts_wdata <= timestep_within_epoch;
                                case (stackout_mode)
                                    2'd0: spike_contribution <= effective_threshold;
                                    2'd1: spike_contribution <= nrn_rdata;
                                    2'd2: spike_contribution <= cur_rdata;
                                    2'd3: spike_contribution <= acc_rdata;
                                endcase
                                if (is_root_rdata) begin
                                    if (fifo_sel) begin
                                        fifo_a_push          <= 1;
                                        fifo_a_push_data_reg <= {proc_neuron[NEURON_BITS-1:0], spike_payload_val};
                                    end else begin
                                        fifo_b_push          <= 1;
                                        fifo_b_push_data_reg <= {proc_neuron[NEURON_BITS-1:0], spike_payload_val};
                                    end
                                    spike_out_valid   <= 1;
                                    spike_out_id      <= proc_neuron[NEURON_BITS-1:0];
                                    spike_out_payload <= spike_payload_val;
                                    total_spikes      <= total_spikes + 1;
                                    perf_spike_count  <= perf_spike_count + 1;
                                    if (trace_fifo_enable && !trace_fifo_full)
                                        trace_fifo_mem[trace_wr_ptr[5:0]] <= {timestep_count[15:0], {(16-NEURON_BITS){1'b0}}, proc_neuron[NEURON_BITS-1:0]};
                                    if (trace_fifo_enable && !trace_fifo_full)
                                        trace_wr_ptr <= trace_wr_ptr + 1;
                                end
                            end else begin
                                trace_wdata  <= trace1_decay_val;
                                trace2_wdata <= trace2_decay_val;
                                x2_trace_wdata <= x2_decay_val;
                                y2_trace_wdata <= y2_decay_val;
                                y3_trace_wdata <= y3_decay_val;
                            end
                        end
                    end else begin
                        proc_current <= {NEURON_WIDTH{1'b0}};
                        if (ref_rdata > 0) begin
                            proc_potential <= $signed({{(NEURON_WIDTH-DATA_WIDTH){param_rest_rdata[DATA_WIDTH-1]}}, param_rest_rdata});
                            proc_refrac   <= ref_rdata - 1;
                            trace_wdata   <= trace1_decay_val;
                            trace2_wdata  <= trace2_decay_val;
                            x2_trace_wdata <= x2_decay_val;
                            y2_trace_wdata <= y2_decay_val;
                            y3_trace_wdata <= y3_decay_val;
                        end else if ($signed(nrn_rdata[DATA_WIDTH-1:0]) + total_input - param_leak_rdata >= effective_threshold) begin
                            proc_potential <= $signed({{(NEURON_WIDTH-DATA_WIDTH){param_rest_rdata[DATA_WIDTH-1]}}, param_rest_rdata});
                            proc_refrac   <= param_refrac_rdata[7:0];
                            trace_wdata   <= TRACE_MAX;
                            trace2_wdata  <= TRACE_MAX;
                            x2_trace_wdata <= TRACE_MAX;
                            y2_trace_wdata <= TRACE_MAX;
                            y3_trace_wdata <= TRACE_MAX;
                            spike_bitmap[proc_neuron[NEURON_BITS-1:0]] <= 1;
                            any_spike_this_ts <= 1;
                            proc_spiked_this_neuron <= 1;
                            spike_ts_we    <= 1;
                            spike_ts_addr  <= proc_neuron[NEURON_BITS-1:0];
                            spike_ts_wdata <= timestep_within_epoch;
                            spike_contribution <= effective_threshold;
                            if (is_root_rdata) begin
                                if (fifo_sel) begin
                                    fifo_a_push          <= 1;
                                    fifo_a_push_data_reg <= {proc_neuron[NEURON_BITS-1:0], spike_payload_val};
                                end else begin
                                    fifo_b_push          <= 1;
                                    fifo_b_push_data_reg <= {proc_neuron[NEURON_BITS-1:0], spike_payload_val};
                                end
                                spike_out_valid   <= 1;
                                spike_out_id      <= proc_neuron[NEURON_BITS-1:0];
                                spike_out_payload <= spike_payload_val;
                                total_spikes      <= total_spikes + 1;
                                perf_spike_count  <= perf_spike_count + 1;
                                if (trace_fifo_enable && !trace_fifo_full)
                                    trace_fifo_mem[trace_wr_ptr[5:0]] <= {timestep_count[15:0], {(16-NEURON_BITS){1'b0}}, proc_neuron[NEURON_BITS-1:0]};
                                if (trace_fifo_enable && !trace_fifo_full)
                                    trace_wr_ptr <= trace_wr_ptr + 1;
                            end
                        end else if ($signed(nrn_rdata[DATA_WIDTH-1:0]) + total_input > param_leak_rdata) begin
                            proc_potential <= $signed({{(NEURON_WIDTH-DATA_WIDTH){1'b0}}, $signed(nrn_rdata[DATA_WIDTH-1:0]) + total_input - param_leak_rdata});
                            trace_wdata   <= trace1_decay_val;
                            trace2_wdata  <= trace2_decay_val;
                            x2_trace_wdata <= x2_decay_val;
                            y2_trace_wdata <= y2_decay_val;
                            y3_trace_wdata <= y3_decay_val;
                        end else begin
                            proc_potential <= $signed({{(NEURON_WIDTH-DATA_WIDTH){param_rest_rdata[DATA_WIDTH-1]}}, param_rest_rdata});
                            trace_wdata   <= trace1_decay_val;
                            trace2_wdata  <= trace2_decay_val;
                            x2_trace_wdata <= x2_decay_val;
                            y2_trace_wdata <= y2_decay_val;
                            y3_trace_wdata <= y3_decay_val;
                        end
                    end

                    if (epoch_counter == epoch_interval - 1 && homeo_target_rdata > 0) begin
                        if (spike_cnt_rdata > homeo_target_rdata) begin
                            homeo_thr_we <= 1;
                            homeo_thr_wdata <= (param_thr_rdata + $signed({8'd0, homeo_eta_rdata}) > THRESHOLD * 4)
                                ? THRESHOLD * 4
                                : param_thr_rdata + $signed({8'd0, homeo_eta_rdata});
                        end else if (spike_cnt_rdata < homeo_target_rdata) begin
                            homeo_thr_we <= 1;
                            homeo_thr_wdata <= (param_thr_rdata - $signed({8'd0, homeo_eta_rdata}) < THRESHOLD / 4)
                                ? THRESHOLD / 4
                                : param_thr_rdata - $signed({8'd0, homeo_eta_rdata});
                        end
                    end

                    saved_parent_ptr <= parent_ptr_rdata;

                    state <= S_UPDATE_WRITE;
                end

                S_UPDATE_PARENT_ADDR: begin
                    acc_addr <= saved_parent_ptr;
                    state <= S_UPDATE_PARENT_WAIT;
                end

                S_UPDATE_PARENT_WAIT: begin
                    state <= S_UPDATE_PARENT_ACC;
                end

                S_UPDATE_PARENT_ACC: begin
                    acc_we   <= 1;
                    acc_addr <= saved_parent_ptr;
                    case (joinop_rdata)
                        2'd0:
                            acc_wdata <= acc_rdata + spike_contribution;
                        2'd1: begin
                            if (spike_contribution[NEURON_WIDTH-1] ?
                                (-spike_contribution > (acc_rdata[NEURON_WIDTH-1] ? -acc_rdata : acc_rdata)) :
                                (spike_contribution > (acc_rdata[NEURON_WIDTH-1] ? -acc_rdata : acc_rdata)))
                                acc_wdata <= spike_contribution;
                            else
                                acc_wdata <= acc_rdata;
                        end
                        2'd2:
                            acc_wdata <= acc_rdata | spike_contribution;
                        2'd3:
                            acc_wdata <= acc_rdata;
                    endcase
                    if (proc_neuron < NUM_NEURONS - 1) begin
                        proc_neuron <= proc_neuron + 1;
                        state       <= S_UPDATE_INIT;
                    end else if (update_pass < num_updates - 1) begin
                        update_pass <= update_pass + 1;
                        proc_neuron <= 0;
                        state       <= S_UPDATE_INIT;
                    end else begin
                        if (skip_idle_enable && !any_spike_this_ts) begin
                            state <= S_DONE;
                        end else if (learn_enable && epoch_counter == 0) begin
                            learn_neuron <= 0;
                            learn_mode   <= 0;
                            state        <= S_LEARN_MC_SCAN;
                        end else if (threefactor_enable && epoch_counter == 0) begin
                            elig_scan_addr <= 0;
                            elig_phase     <= 0;
                            state <= S_ELIG_MC;
                        end else begin
                            state <= S_DONE;
                        end
                    end
                end

                S_UPDATE_WRITE: begin
                    nrn_we    <= 1;
                    nrn_addr  <= proc_neuron[NEURON_BITS-1:0];
                    nrn_wdata <= (proc_potential < vmin_ext) ? vmin_ext :
                                 (proc_potential > vmax_ext) ? vmax_ext : proc_potential;

                    cur_we    <= 1;
                    cur_addr  <= proc_neuron[NEURON_BITS-1:0];
                    cur_wdata <= proc_current;

                    ref_we    <= 1;
                    ref_addr  <= proc_neuron[NEURON_BITS-1:0];
                    ref_wdata <= proc_refrac;

                    acc_we    <= 1;
                    acc_addr  <= proc_neuron[NEURON_BITS-1:0];
                    acc_wdata <= 0;

                    dend_acc_1_we    <= 1;
                    dend_acc_1_addr  <= proc_neuron[NEURON_BITS-1:0];
                    dend_acc_1_wdata <= 0;

                    dend_acc_2_we    <= 1;
                    dend_acc_2_addr  <= proc_neuron[NEURON_BITS-1:0];
                    dend_acc_2_wdata <= 0;

                    dend_acc_3_we    <= 1;
                    dend_acc_3_addr  <= proc_neuron[NEURON_BITS-1:0];
                    dend_acc_3_wdata <= 0;

                    trace_we    <= 1;
                    trace_addr  <= proc_neuron[NEURON_BITS-1:0];
                    trace2_we   <= 1;
                    trace2_addr <= proc_neuron[NEURON_BITS-1:0];
                    x2_trace_we   <= 1;
                    x2_trace_addr <= proc_neuron[NEURON_BITS-1:0];
                    y2_trace_we   <= 1;
                    y2_trace_addr <= proc_neuron[NEURON_BITS-1:0];
                    y3_trace_we   <= 1;
                    y3_trace_addr <= proc_neuron[NEURON_BITS-1:0];

                    spike_cnt_addr <= proc_neuron[NEURON_BITS-1:0];
                    if (epoch_counter == epoch_interval - 1) begin
                        spike_cnt_we    <= 1;
                        spike_cnt_wdata <= spike_bitmap[proc_neuron[NEURON_BITS-1:0]] ? 8'd1 : 8'd0;
                    end else if (spike_bitmap[proc_neuron[NEURON_BITS-1:0]]) begin
                        spike_cnt_we    <= 1;
                        spike_cnt_wdata <= spike_cnt_rdata + 8'd1;
                    end

                    if (proc_spiked_this_neuron && saved_parent_ptr != {NEURON_BITS{1'b1}}) begin
                        state <= S_UPDATE_PARENT_ADDR;
                    end else if (proc_neuron < NUM_NEURONS - 1) begin
                        proc_neuron <= proc_neuron + 1;
                        state       <= S_UPDATE_INIT;
                    end else if (update_pass < num_updates - 1) begin
                        update_pass <= update_pass + 1;
                        proc_neuron <= 0;
                        state       <= S_UPDATE_INIT;
                    end else begin
                        if (skip_idle_enable && !any_spike_this_ts) begin
                            state <= S_DONE;
                        end else if (learn_enable && epoch_counter == 0) begin
                            learn_neuron <= 0;
                            learn_mode   <= 0;
                            state        <= S_LEARN_MC_SCAN;
                        end else if (threefactor_enable && epoch_counter == 0) begin
                            elig_scan_addr <= 0;
                            elig_phase     <= 0;
                            state <= S_ELIG_MC;
                        end else begin
                            state <= S_DONE;
                        end
                    end
                end

                S_LEARN_MC_SCAN: begin
                    if (learn_neuron == NUM_NEURONS) begin
                        if (learn_mode == 0) begin
                            learn_mode   <= 1;
                            learn_neuron <= 0;
                        end else begin
                            if (threefactor_enable) begin
                                elig_scan_addr <= 0;
                                elig_phase     <= 0;
                                state <= S_ELIG_MC;
                            end else begin
                                state <= S_DONE;
                            end
                        end
                    end else if (spike_bitmap[learn_neuron[NEURON_BITS-1:0]]) begin
                        learn_slot <= 0;
                        if (learn_mode == 0) begin
                            index_rd_addr <= learn_neuron[NEURON_BITS-1:0];
                            state <= S_LEARN_MC_IDX_WAIT;
                        end else begin
                            state <= S_LEARN_MC_SETUP;
                        end
                    end else begin
                        learn_neuron <= learn_neuron + 1;
                    end
                end

                S_LEARN_MC_IDX_WAIT: begin
                    state <= S_LEARN_MC_IDX_READ;
                end

                S_LEARN_MC_IDX_READ: begin
                    learn_base_addr <= index_rdata[COUNT_BITS +: POOL_ADDR_BITS];
                    learn_count     <= index_rdata[COUNT_BITS-1:0];
                    if (index_rdata[COUNT_BITS-1:0] == 0 ||
                        index_rdata[INDEX_WIDTH-1 -: 2] != FMT_SPARSE) begin
                        learn_neuron <= learn_neuron + 1;
                        state <= S_LEARN_MC_SCAN;
                    end else begin
                        state <= S_LEARN_MC_SETUP;
                    end
                end

                S_LEARN_MC_SETUP: begin
                    if (learn_mode == 0) begin
                        pool_addr_r <= learn_base_addr + learn_slot;
                        elig_addr   <= learn_base_addr + learn_slot;
                    end else begin
                        rev_addr <= {learn_neuron[NEURON_BITS-1:0], learn_slot[REV_SLOT_BITS-1:0]};
                    end
                    state <= S_LEARN_MC_WAIT1;
                end

                S_LEARN_MC_WAIT1: begin
                    state <= S_LEARN_MC_LOAD;
                end

                S_LEARN_MC_LOAD: begin
                    if (learn_mode == 0) begin
                        trace_addr    <= pool_tgt_rdata;
                        trace2_addr   <= pool_tgt_rdata;
                        x2_trace_addr <= pool_tgt_rdata;
                        y2_trace_addr <= pool_tgt_rdata;
                        y3_trace_addr <= pool_tgt_rdata;
                        spike_ts_addr <= pool_tgt_rdata;
                    end else begin
                        learn_rev_valid     <= rev_rdata[REV_DATA_W-1];
                        learn_rev_src       <= rev_rdata[POOL_ADDR_BITS +: NEURON_BITS];
                        learn_rev_pool_addr <= rev_rdata[POOL_ADDR_BITS-1:0];
                        trace_addr          <= rev_rdata[POOL_ADDR_BITS +: NEURON_BITS];
                        trace2_addr         <= rev_rdata[POOL_ADDR_BITS +: NEURON_BITS];
                        x2_trace_addr       <= rev_rdata[POOL_ADDR_BITS +: NEURON_BITS];
                        y2_trace_addr       <= rev_rdata[POOL_ADDR_BITS +: NEURON_BITS];
                        y3_trace_addr       <= rev_rdata[POOL_ADDR_BITS +: NEURON_BITS];
                        pool_addr_r         <= rev_rdata[POOL_ADDR_BITS-1:0];
                        elig_addr           <= rev_rdata[POOL_ADDR_BITS-1:0];
                        spike_ts_addr       <= rev_rdata[POOL_ADDR_BITS +: NEURON_BITS];
                    end
                    state <= S_LEARN_MC_WAIT2;
                end

                S_LEARN_MC_WAIT2: begin
                    state <= S_LEARN_MC_REGLD;
                end

                S_LEARN_MC_REGLD: begin
                    if (learn_mode == 1 && !learn_rev_valid) begin
                        state <= S_LEARN_MC_NEXT;
                    end else begin
                        mc_regs[0]  <= $signed({8'd0, trace_rdata});
                        mc_regs[1]  <= $signed({8'd0, x2_trace_rdata});
                        mc_regs[2]  <= $signed({8'd0, trace2_rdata});
                        mc_regs[3]  <= $signed({8'd0, y2_trace_rdata});
                        mc_regs[4]  <= $signed({8'd0, y3_trace_rdata});
                        mc_regs[5]  <= pool_wt_rdata;
                        mc_regs[6]  <= $signed({10'd0, pool_delay_rdata});
                        mc_regs[7]  <= pool_tag_rdata;
                        mc_regs[8]  <= elig_rdata;
                        mc_regs[9]  <= reward_trace;
                        mc_regs[10] <= $signed({8'd0, spike_ts_rdata});
                        mc_regs[11] <= 16'sd0;
                        mc_regs[12] <= 16'sd0;
                        mc_regs[13] <= 16'sd0;
                        mc_regs[14] <= 16'sd0;
                        mc_regs[15] <= 16'sd0;
                        mc_pc <= {threefactor_enable, learn_mode, 6'd0};
                        state <= S_LEARN_MC_FETCH;
                    end
                end

                S_LEARN_MC_FETCH: begin
                    state <= S_LEARN_MC_EXEC;
                end

                S_LEARN_MC_EXEC: begin
                    case (mc_opcode)
                        4'd0: begin
                            mc_pc <= mc_pc + 1;
                            state <= S_LEARN_MC_FETCH;
                        end
                        4'd1, 4'd2, 4'd3, 4'd4, 4'd5, 4'd6, 4'd7, 4'd8: begin
                            mc_regs[mc_dst] <= mc_alu_result;
                            mc_pc <= mc_pc + 1;
                            state <= S_LEARN_MC_FETCH;
                        end
                        4'd9: begin
                            pool_wt_we_r    <= 1;
                            pool_wt_wr_addr <= learn_wr_addr;
                            pool_wt_wr_data <= mc_regs[5] + {{15{1'b0}}, lfsr[0]};
                            mc_pc <= mc_pc + 1;
                            state <= S_LEARN_MC_FETCH;
                        end
                        4'd10: begin
                            elig_we    <= 1;
                            elig_addr  <= learn_wr_addr;
                            elig_wdata <= mc_regs[8];
                            mc_pc <= mc_pc + 1;
                            state <= S_LEARN_MC_FETCH;
                        end
                        4'd11: begin
                            mc_pc <= (mc_regs[mc_src_a] == 0) ? (mc_pc + 2) : (mc_pc + 1);
                            state <= S_LEARN_MC_FETCH;
                        end
                        4'd12: begin
                            mc_pc <= (mc_regs[mc_src_a] != 0) ? (mc_pc + 2) : (mc_pc + 1);
                            state <= S_LEARN_MC_FETCH;
                        end
                        4'd13: begin
                            state <= S_LEARN_MC_NEXT;
                        end
                        4'd14: begin
                            pool_delay_we_learn   <= 1;
                            pool_delay_learn_addr <= learn_wr_addr;
                            pool_delay_learn_data <= mc_regs[6][5:0];
                            mc_pc <= mc_pc + 1;
                            state <= S_LEARN_MC_FETCH;
                        end
                        4'd15: begin
                            pool_tag_we_r    <= 1;
                            pool_tag_wr_addr <= learn_wr_addr;
                            pool_tag_wr_data <= mc_regs[7];
                            mc_pc <= mc_pc + 1;
                            state <= S_LEARN_MC_FETCH;
                        end
                        default: begin
                            mc_pc <= mc_pc + 1;
                            state <= S_LEARN_MC_FETCH;
                        end
                    endcase
                end

                S_LEARN_MC_NEXT: begin
                    if (learn_mode == 0) begin
                        if (learn_slot < learn_count - 1) begin
                            learn_slot <= learn_slot + 1;
                            state      <= S_LEARN_MC_SETUP;
                        end else begin
                            learn_neuron <= learn_neuron + 1;
                            state        <= S_LEARN_MC_SCAN;
                        end
                    end else begin
                        if (learn_slot < REV_FANIN - 1) begin
                            learn_slot <= learn_slot + 1;
                            state      <= S_LEARN_MC_SETUP;
                        end else begin
                            learn_neuron <= learn_neuron + 1;
                            state        <= S_LEARN_MC_SCAN;
                        end
                    end
                end

                S_ELIG_MC: begin
                    case (elig_phase)
                        2'd0: begin
                            if (elig_scan_addr >= pool_used_count) begin
                                state <= S_DONE;
                            end else begin
                                pool_addr_r <= elig_scan_addr[POOL_ADDR_BITS-1:0];
                                elig_addr   <= elig_scan_addr[POOL_ADDR_BITS-1:0];
                                elig_phase  <= 2'd1;
                            end
                        end
                        2'd1: begin
                            elig_phase <= 2'd2;
                        end
                        2'd2: begin
                            if (reward_trace != 0) begin
                                pool_wt_we_r    <= 1;
                                pool_wt_wr_addr <= elig_scan_addr[POOL_ADDR_BITS-1:0];
                                pool_wt_wr_data <= elig_new_wt;
                            end
                            elig_we    <= 1;
                            elig_wdata <= elig_decayed;
                            elig_scan_addr <= elig_scan_addr + 1;
                            elig_phase     <= 2'd0;
                        end
                        default: elig_phase <= 2'd0;
                    endcase
                end

                S_DELAY_DRAIN_INIT: begin
                    drain_cnt <= delay_count[current_ts_mod64];
                    drain_idx <= 0;
                    if (delay_count[current_ts_mod64] == 0) begin
                        state <= S_DELIVER_POP;
                    end else begin
                        dq_addr <= {current_ts_mod64, {DELAY_ENTRY_BITS{1'b0}}};
                        state <= S_DELAY_DRAIN_QWAIT;
                    end
                end

                S_DELAY_DRAIN_QWAIT: begin
                    state <= S_DELAY_DRAIN_CAP;
                end

                S_DELAY_DRAIN_CAP: begin
                    dq_cap_target  <= dq_rdata[DELAY_QUEUE_ENTRY_W-1 -: NEURON_BITS];
                    dq_cap_current <= dq_rdata[COMPARTMENT_BITS +: DATA_WIDTH];
                    dq_cap_comp    <= dq_rdata[COMPARTMENT_BITS-1:0];
                    acc_addr        <= dq_rdata[DELAY_QUEUE_ENTRY_W-1 -: NEURON_BITS];
                    dend_acc_1_addr <= dq_rdata[DELAY_QUEUE_ENTRY_W-1 -: NEURON_BITS];
                    dend_acc_2_addr <= dq_rdata[DELAY_QUEUE_ENTRY_W-1 -: NEURON_BITS];
                    dend_acc_3_addr <= dq_rdata[DELAY_QUEUE_ENTRY_W-1 -: NEURON_BITS];
                    state <= S_DELAY_DRAIN_AWAIT;
                end

                S_DELAY_DRAIN_AWAIT: begin
                    state <= S_DELAY_DRAIN_ACC;
                end

                S_DELAY_DRAIN_ACC: begin
                    case (dq_cap_comp)
                        2'd0: begin
                            acc_we    <= 1;
                            acc_addr  <= dq_cap_target;
                            acc_wdata <= acc_rdata + dq_cap_current;
                        end
                        2'd1: begin
                            dend_acc_1_we    <= 1;
                            dend_acc_1_addr  <= dq_cap_target;
                            dend_acc_1_wdata <= dend_acc_1_rdata + dq_cap_current;
                        end
                        2'd2: begin
                            dend_acc_2_we    <= 1;
                            dend_acc_2_addr  <= dq_cap_target;
                            dend_acc_2_wdata <= dend_acc_2_rdata + dq_cap_current;
                        end
                        2'd3: begin
                            dend_acc_3_we    <= 1;
                            dend_acc_3_addr  <= dq_cap_target;
                            dend_acc_3_wdata <= dend_acc_3_rdata + dq_cap_current;
                        end
                    endcase
                    if (drain_idx < drain_cnt - 1) begin
                        drain_idx <= drain_idx + 1;
                        dq_addr   <= {current_ts_mod64, drain_idx + {{(DELAY_ENTRY_BITS-1){1'b0}}, 1'b1}};
                        state     <= S_DELAY_DRAIN_QWAIT;
                    end else begin
                        delay_count[current_ts_mod64] <= 0;
                        state <= S_DELIVER_POP;
                    end
                end

                S_DONE: begin
                    fifo_sel <= ~fifo_sel;
                    if (fifo_sel)
                        fifo_b_clear <= 1;
                    else
                        fifo_a_clear <= 1;

                    timestep_done    <= 1;
                    timestep_count   <= timestep_count + 1;
                    current_ts_mod64 <= current_ts_mod64 + 1;
                    proc_neuron      <= 0;
                    spike_bitmap     <= 0;

                    epoch_counter <= (epoch_counter >= epoch_interval - 1) ? 8'd0 : epoch_counter + 8'd1;

                    timestep_within_epoch <= (epoch_counter >= epoch_interval - 1) ?
                        8'd0 : timestep_within_epoch + 8'd1;

                    was_idle <= ~any_spike_this_ts;

                    reward_trace <= rt_decayed + reward_value;

                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

`ifdef SIMULATION
    integer sim_init_i;
    initial begin
        for (sim_init_i = 0; sim_init_i < NUM_NEURONS; sim_init_i = sim_init_i + 1) begin
            is_root_mem.mem[sim_init_i] = 1'b1;
        end
        for (sim_init_i = 0; sim_init_i < NUM_NEURONS; sim_init_i = sim_init_i + 1) begin
            threshold_mem.mem[sim_init_i] = THRESHOLD;
            leak_mem.mem[sim_init_i]      = LEAK_RATE;
            rest_mem.mem[sim_init_i]      = RESTING_POT;
            refrac_cfg_mem.mem[sim_init_i] = REFRAC_CYCLES;
            vmin_mem.mem[sim_init_i]      = 16'sh8000;
            vmax_mem.mem[sim_init_i]      = 16'sh7FFF;
            tau1_cfg_mem.mem[sim_init_i]  = TAU1_DEFAULT;
            tau2_cfg_mem.mem[sim_init_i]  = TAU2_DEFAULT;
            parent_ptr_mem.mem[sim_init_i] = {NEURON_BITS{1'b1}};
        end
        ucode_mem.mem[0]   = 32'hC000_0000;
        ucode_mem.mem[1]   = 32'hD000_0000;
        ucode_mem.mem[2]   = 32'h4B00_6000;
        ucode_mem.mem[3]   = 32'h255B_0000;
        ucode_mem.mem[4]   = 32'h8B00_0000;
        ucode_mem.mem[5]   = 32'h655B_0000;
        ucode_mem.mem[6]   = 32'h8B00_07D0;
        ucode_mem.mem[7]   = 32'h755B_0000;
        ucode_mem.mem[8]   = 32'h9000_0000;
        ucode_mem.mem[9]   = 32'hD000_0000;
        ucode_mem.mem[64]  = 32'hC000_0000;
        ucode_mem.mem[65]  = 32'hD000_0000;
        ucode_mem.mem[66]  = 32'h4B00_6000;
        ucode_mem.mem[67]  = 32'h155B_0000;
        ucode_mem.mem[68]  = 32'h8B00_0000;
        ucode_mem.mem[69]  = 32'h655B_0000;
        ucode_mem.mem[70]  = 32'h8B00_07D0;
        ucode_mem.mem[71]  = 32'h755B_0000;
        ucode_mem.mem[72]  = 32'h9000_0000;
        ucode_mem.mem[73]  = 32'hD000_0000;
        ucode_mem.mem[128] = 32'hC000_0000;
        ucode_mem.mem[129] = 32'hD000_0000;
        ucode_mem.mem[130] = 32'h4B00_6000;
        ucode_mem.mem[131] = 32'h288B_0000;
        ucode_mem.mem[132] = 32'h8B00_FC18;
        ucode_mem.mem[133] = 32'h688B_0000;
        ucode_mem.mem[134] = 32'h8B00_03E8;
        ucode_mem.mem[135] = 32'h788B_0000;
        ucode_mem.mem[136] = 32'hA000_0000;
        ucode_mem.mem[137] = 32'hD000_0000;
        ucode_mem.mem[192] = 32'hC000_0000;
        ucode_mem.mem[193] = 32'hD000_0000;
        ucode_mem.mem[194] = 32'h4B00_6000;
        ucode_mem.mem[195] = 32'h188B_0000;
        ucode_mem.mem[196] = 32'h8B00_FC18;
        ucode_mem.mem[197] = 32'h688B_0000;
        ucode_mem.mem[198] = 32'h8B00_03E8;
        ucode_mem.mem[199] = 32'h788B_0000;
        ucode_mem.mem[200] = 32'hA000_0000;
        ucode_mem.mem[201] = 32'hD000_0000;
    end
`endif

endmodule
