module neuromorphic_top #(
    parameter CLK_FREQ       = 100_000_000,
    parameter BAUD           = 115200,
    parameter NUM_CORES      = 128,
    parameter CORE_ID_BITS   = 12,
    parameter NUM_NEURONS    = 1024,
    parameter NEURON_BITS    = 10,
    parameter DATA_WIDTH     = 16,
    parameter POOL_DEPTH     = 131072,
    parameter POOL_ADDR_BITS = 17,
    parameter COUNT_BITS     = 12,
    parameter REV_FANIN      = 32,
    parameter REV_SLOT_BITS  = 5,
    parameter THRESHOLD      = 16'sd1000,
    parameter LEAK_RATE      = 16'sd3,
    parameter REFRAC_CYCLES  = 3,
    parameter ROUTE_FANOUT   = 8,
    parameter ROUTE_SLOT_BITS = 3,
    parameter GLOBAL_ROUTE_SLOTS     = 4,
    parameter GLOBAL_ROUTE_SLOT_BITS = 2,

    parameter CHIP_LINK_EN = 0,
    parameter NOC_MODE = 0,
    parameter MESH_X   = 2,
    parameter MESH_Y   = 2,

    parameter BYPASS_UART = 0
)(
    input  wire clk,
    input  wire rst_n,
    input  wire uart_rxd,
    output wire uart_txd,

    output wire [7:0]  link_tx_data,
    output wire        link_tx_valid,
    input  wire        link_tx_ready,
    input  wire [7:0]  link_rx_data,
    input  wire        link_rx_valid,
    output wire        link_rx_ready,

    input  wire [7:0]  rx_data_ext,
    input  wire        rx_valid_ext,
    output wire [7:0]  tx_data_ext,
    output wire        tx_valid_ext,
    input  wire        tx_ready_ext
);

    wire [7:0] rx_data;
    wire       rx_valid;
    wire [7:0] tx_data;
    wire       tx_valid;
    wire       tx_ready;

    generate
        if (BYPASS_UART == 0) begin : gen_uart
            uart_rx #(
                .CLK_FREQ (CLK_FREQ),
                .BAUD     (BAUD)
            ) u_uart_rx (
                .clk   (clk),
                .rst_n (rst_n),
                .rx    (uart_rxd),
                .data  (rx_data),
                .valid (rx_valid)
            );

            uart_tx #(
                .CLK_FREQ (CLK_FREQ),
                .BAUD     (BAUD)
            ) u_uart_tx (
                .clk   (clk),
                .rst_n (rst_n),
                .data  (tx_data),
                .valid (tx_valid),
                .tx    (uart_txd),
                .ready (tx_ready)
            );
        end else begin : gen_bypass
            assign rx_data  = rx_data_ext;
            assign rx_valid = rx_valid_ext;
            assign tx_ready = tx_ready_ext;
            assign uart_txd = 1'b1;
        end
    endgenerate

    assign tx_data_ext  = tx_data;
    assign tx_valid_ext = tx_valid;

    wire        hi_mesh_start;

    wire                              hi_prog_pool_we;
    wire [CORE_ID_BITS-1:0]         hi_prog_pool_core;
    wire [POOL_ADDR_BITS-1:0]       hi_prog_pool_addr;
    wire [NEURON_BITS-1:0]          hi_prog_pool_src;
    wire [NEURON_BITS-1:0]          hi_prog_pool_target;
    wire signed [DATA_WIDTH-1:0]    hi_prog_pool_weight;
    wire [1:0]                      hi_prog_pool_comp;

    wire                              hi_prog_index_we;
    wire [CORE_ID_BITS-1:0]         hi_prog_index_core;
    wire [NEURON_BITS-1:0]          hi_prog_index_neuron;
    wire [POOL_ADDR_BITS-1:0]       hi_prog_index_base;
    wire [COUNT_BITS-1:0]           hi_prog_index_count;
    wire [1:0]                      hi_prog_index_format;

    wire        hi_prog_route_we;
    wire [CORE_ID_BITS-1:0]    hi_prog_route_src_core;
    wire [NEURON_BITS-1:0]     hi_prog_route_src_neuron;
    wire [ROUTE_SLOT_BITS-1:0] hi_prog_route_slot;
    wire [CORE_ID_BITS-1:0]    hi_prog_route_dest_core;
    wire [NEURON_BITS-1:0]     hi_prog_route_dest_neuron;
    wire signed [DATA_WIDTH-1:0] hi_prog_route_weight;

    wire                              hi_prog_global_route_we;
    wire [CORE_ID_BITS-1:0]         hi_prog_global_route_src_core;
    wire [NEURON_BITS-1:0]          hi_prog_global_route_src_neuron;
    wire [GLOBAL_ROUTE_SLOT_BITS-1:0] hi_prog_global_route_slot;
    wire [CORE_ID_BITS-1:0]         hi_prog_global_route_dest_core;
    wire [NEURON_BITS-1:0]          hi_prog_global_route_dest_neuron;
    wire signed [DATA_WIDTH-1:0]    hi_prog_global_route_weight;

    wire        hi_ext_valid;
    wire [CORE_ID_BITS-1:0]    hi_ext_core;
    wire [NEURON_BITS-1:0]     hi_ext_neuron_id;
    wire signed [DATA_WIDTH-1:0] hi_ext_current;

    wire        hi_learn_enable;
    wire        hi_graded_enable;
    wire        hi_dendritic_enable;
    wire        hi_async_enable;
    wire        hi_threefactor_enable;
    wire        hi_noise_enable;
    wire        hi_skip_idle_enable;
    wire        hi_scale_u_enable;
    wire signed [DATA_WIDTH-1:0] hi_reward_value;

    wire                              hi_prog_delay_we;
    wire [CORE_ID_BITS-1:0]         hi_prog_delay_core;
    wire [POOL_ADDR_BITS-1:0]       hi_prog_delay_addr;
    wire [5:0]                      hi_prog_delay_value;

    wire                              hi_prog_ucode_we;
    wire [CORE_ID_BITS-1:0]         hi_prog_ucode_core;
    wire [7:0]                      hi_prog_ucode_addr;
    wire [31:0]                     hi_prog_ucode_data;

    wire        hi_prog_param_we;
    wire [CORE_ID_BITS-1:0]    hi_prog_param_core;
    wire [NEURON_BITS-1:0]     hi_prog_param_neuron;
    wire [4:0]                 hi_prog_param_id;
    wire signed [DATA_WIDTH-1:0] hi_prog_param_value;

    wire                              hi_probe_read;
    wire [CORE_ID_BITS-1:0]         hi_probe_core;
    wire [NEURON_BITS-1:0]          hi_probe_neuron;
    wire [4:0]                      hi_probe_state_id;
    wire [POOL_ADDR_BITS-1:0]       hi_probe_pool_addr;
    wire signed [DATA_WIDTH-1:0]    mesh_probe_data;
    wire                            mesh_probe_valid;

    wire [7:0]  hi_dvfs_stall;

    wire        mesh_timestep_done;
    wire [5:0]  mesh_state;
    wire [31:0] mesh_total_spikes;
    wire [31:0] mesh_timestep_count;

    host_interface #(
        .NUM_CORES      (NUM_CORES),
        .CORE_ID_BITS   (CORE_ID_BITS),
        .NUM_NEURONS    (NUM_NEURONS),
        .NEURON_BITS    (NEURON_BITS),
        .DATA_WIDTH     (DATA_WIDTH),
        .POOL_ADDR_BITS (POOL_ADDR_BITS),
        .COUNT_BITS     (COUNT_BITS),
        .ROUTE_SLOT_BITS(ROUTE_SLOT_BITS),
        .GLOBAL_ROUTE_SLOT_BITS(GLOBAL_ROUTE_SLOT_BITS)
    ) u_host_if (
        .clk       (clk),
        .rst_n     (rst_n),
        .rx_data   (rx_data),
        .rx_valid  (rx_valid),
        .tx_data   (tx_data),
        .tx_valid  (tx_valid),
        .tx_ready  (tx_ready),

        .mesh_start              (hi_mesh_start),

        .mesh_prog_pool_we       (hi_prog_pool_we),
        .mesh_prog_pool_core     (hi_prog_pool_core),
        .mesh_prog_pool_addr     (hi_prog_pool_addr),
        .mesh_prog_pool_src      (hi_prog_pool_src),
        .mesh_prog_pool_target   (hi_prog_pool_target),
        .mesh_prog_pool_weight   (hi_prog_pool_weight),
        .mesh_prog_pool_comp     (hi_prog_pool_comp),

        .mesh_prog_index_we      (hi_prog_index_we),
        .mesh_prog_index_core    (hi_prog_index_core),
        .mesh_prog_index_neuron  (hi_prog_index_neuron),
        .mesh_prog_index_base    (hi_prog_index_base),
        .mesh_prog_index_count   (hi_prog_index_count),
        .mesh_prog_index_format  (hi_prog_index_format),

        .mesh_prog_route_we      (hi_prog_route_we),
        .mesh_prog_route_src_core   (hi_prog_route_src_core),
        .mesh_prog_route_src_neuron (hi_prog_route_src_neuron),
        .mesh_prog_route_slot       (hi_prog_route_slot),
        .mesh_prog_route_dest_core  (hi_prog_route_dest_core),
        .mesh_prog_route_dest_neuron(hi_prog_route_dest_neuron),
        .mesh_prog_route_weight     (hi_prog_route_weight),

        .mesh_prog_global_route_we          (hi_prog_global_route_we),
        .mesh_prog_global_route_src_core    (hi_prog_global_route_src_core),
        .mesh_prog_global_route_src_neuron  (hi_prog_global_route_src_neuron),
        .mesh_prog_global_route_slot        (hi_prog_global_route_slot),
        .mesh_prog_global_route_dest_core   (hi_prog_global_route_dest_core),
        .mesh_prog_global_route_dest_neuron (hi_prog_global_route_dest_neuron),
        .mesh_prog_global_route_weight      (hi_prog_global_route_weight),

        .mesh_ext_valid          (hi_ext_valid),
        .mesh_ext_core           (hi_ext_core),
        .mesh_ext_neuron_id      (hi_ext_neuron_id),
        .mesh_ext_current        (hi_ext_current),

        .mesh_learn_enable       (hi_learn_enable),
        .mesh_graded_enable      (hi_graded_enable),
        .mesh_dendritic_enable   (hi_dendritic_enable),
        .mesh_async_enable       (hi_async_enable),
        .mesh_threefactor_enable (hi_threefactor_enable),
        .mesh_noise_enable       (hi_noise_enable),
        .mesh_skip_idle_enable   (hi_skip_idle_enable),
        .mesh_scale_u_enable     (hi_scale_u_enable),
        .mesh_reward_value       (hi_reward_value),

        .mesh_prog_delay_we      (hi_prog_delay_we),
        .mesh_prog_delay_core    (hi_prog_delay_core),
        .mesh_prog_delay_addr    (hi_prog_delay_addr),
        .mesh_prog_delay_value   (hi_prog_delay_value),

        .mesh_prog_ucode_we     (hi_prog_ucode_we),
        .mesh_prog_ucode_core   (hi_prog_ucode_core),
        .mesh_prog_ucode_addr   (hi_prog_ucode_addr),
        .mesh_prog_ucode_data   (hi_prog_ucode_data),

        .mesh_prog_param_we      (hi_prog_param_we),
        .mesh_prog_param_core    (hi_prog_param_core),
        .mesh_prog_param_neuron  (hi_prog_param_neuron),
        .mesh_prog_param_id      (hi_prog_param_id),
        .mesh_prog_param_value   (hi_prog_param_value),

        .mesh_probe_read     (hi_probe_read),
        .mesh_probe_core     (hi_probe_core),
        .mesh_probe_neuron   (hi_probe_neuron),
        .mesh_probe_state_id (hi_probe_state_id),
        .mesh_probe_pool_addr(hi_probe_pool_addr),
        .mesh_probe_data     (mesh_probe_data),
        .mesh_probe_valid    (mesh_probe_valid),

        .mesh_dvfs_stall     (hi_dvfs_stall),

        .mesh_timestep_done  (mesh_timestep_done),
        .mesh_state          (mesh_state),
        .mesh_total_spikes   (mesh_total_spikes),
        .mesh_timestep_count (mesh_timestep_count)
    );

    wire        mesh_link_tx_push;
    wire [CORE_ID_BITS-1:0] mesh_link_tx_core;
    wire [NEURON_BITS-1:0]  mesh_link_tx_neuron;
    wire [7:0]              mesh_link_tx_payload;
    wire                    mesh_link_tx_full;
    wire [CORE_ID_BITS-1:0] mesh_link_rx_core;
    wire [NEURON_BITS-1:0]  mesh_link_rx_neuron;
    wire signed [DATA_WIDTH-1:0] mesh_link_rx_current;
    wire                    mesh_link_rx_pop;
    wire                    mesh_link_rx_empty;

    wire [NUM_CORES-1:0]             spike_valid_bus;
    wire [NUM_CORES*NEURON_BITS-1:0] spike_id_bus;

    generate
        if (NOC_MODE == 1) begin : gen_async_noc
            async_noc_mesh #(
                .NUM_CORES      (NUM_CORES),
                .CORE_ID_BITS   (CORE_ID_BITS),
                .NUM_NEURONS    (NUM_NEURONS),
                .NEURON_BITS    (NEURON_BITS),
                .DATA_WIDTH     (DATA_WIDTH),
                .POOL_DEPTH     (POOL_DEPTH),
                .POOL_ADDR_BITS (POOL_ADDR_BITS),
                .COUNT_BITS     (COUNT_BITS),
                .REV_FANIN      (REV_FANIN),
                .REV_SLOT_BITS  (REV_SLOT_BITS),
                .THRESHOLD      (THRESHOLD),
                .LEAK_RATE      (LEAK_RATE),
                .REFRAC_CYCLES  (REFRAC_CYCLES),
                .ROUTE_FANOUT   (ROUTE_FANOUT),
                .ROUTE_SLOT_BITS(ROUTE_SLOT_BITS),
                .GLOBAL_ROUTE_SLOTS    (GLOBAL_ROUTE_SLOTS),
                .GLOBAL_ROUTE_SLOT_BITS(GLOBAL_ROUTE_SLOT_BITS),
                .MESH_X         (MESH_X),
                .MESH_Y         (MESH_Y)
            ) u_mesh (
                .clk               (clk),
                .rst_n             (rst_n),
                .start             (hi_mesh_start),
                .learn_enable      (hi_learn_enable),
                .graded_enable     (hi_graded_enable),
                .dendritic_enable  (hi_dendritic_enable),
                .async_enable      (hi_async_enable),
                .threefactor_enable(hi_threefactor_enable),
                .noise_enable      (hi_noise_enable),
                .skip_idle_enable  (hi_skip_idle_enable),
                .scale_u_enable    (hi_scale_u_enable),
                .reward_value      (hi_reward_value),
                .prog_pool_we      (hi_prog_pool_we),
                .prog_pool_core    (hi_prog_pool_core),
                .prog_pool_addr    (hi_prog_pool_addr),
                .prog_pool_src     (hi_prog_pool_src),
                .prog_pool_target  (hi_prog_pool_target),
                .prog_pool_weight  (hi_prog_pool_weight),
                .prog_pool_comp    (hi_prog_pool_comp),
                .prog_index_we     (hi_prog_index_we),
                .prog_index_core   (hi_prog_index_core),
                .prog_index_neuron (hi_prog_index_neuron),
                .prog_index_base   (hi_prog_index_base),
                .prog_index_count  (hi_prog_index_count),
                .prog_index_format (hi_prog_index_format),
                .prog_route_we         (hi_prog_route_we),
                .prog_route_src_core   (hi_prog_route_src_core),
                .prog_route_src_neuron (hi_prog_route_src_neuron),
                .prog_route_slot       (hi_prog_route_slot),
                .prog_route_dest_core  (hi_prog_route_dest_core),
                .prog_route_dest_neuron(hi_prog_route_dest_neuron),
                .prog_route_weight     (hi_prog_route_weight),
                .prog_global_route_we          (hi_prog_global_route_we),
                .prog_global_route_src_core    (hi_prog_global_route_src_core),
                .prog_global_route_src_neuron  (hi_prog_global_route_src_neuron),
                .prog_global_route_slot        (hi_prog_global_route_slot),
                .prog_global_route_dest_core   (hi_prog_global_route_dest_core),
                .prog_global_route_dest_neuron (hi_prog_global_route_dest_neuron),
                .prog_global_route_weight      (hi_prog_global_route_weight),
                .prog_delay_we     (hi_prog_delay_we),
                .prog_delay_core   (hi_prog_delay_core),
                .prog_delay_addr   (hi_prog_delay_addr),
                .prog_delay_value  (hi_prog_delay_value),
                .prog_ucode_we     (hi_prog_ucode_we),
                .prog_ucode_core   (hi_prog_ucode_core),
                .prog_ucode_addr   (hi_prog_ucode_addr),
                .prog_ucode_data   (hi_prog_ucode_data),
                .prog_param_we     (hi_prog_param_we),
                .prog_param_core   (hi_prog_param_core),
                .prog_param_neuron (hi_prog_param_neuron),
                .prog_param_id     (hi_prog_param_id),
                .prog_param_value  (hi_prog_param_value),
                .probe_read        (hi_probe_read),
                .probe_core        (hi_probe_core),
                .probe_neuron      (hi_probe_neuron),
                .probe_state_id    (hi_probe_state_id),
                .probe_pool_addr   (hi_probe_pool_addr),
                .probe_data        (mesh_probe_data),
                .probe_valid       (mesh_probe_valid),
                .ext_valid         (hi_ext_valid),
                .ext_core          (hi_ext_core),
                .ext_neuron_id     (hi_ext_neuron_id),
                .ext_current       (hi_ext_current),
                .timestep_done     (mesh_timestep_done),
                .spike_valid_bus   (spike_valid_bus),
                .spike_id_bus      (spike_id_bus),
                .mesh_state_out    (mesh_state),
                .total_spikes      (mesh_total_spikes),
                .timestep_count    (mesh_timestep_count),
                .core_idle_bus     (),
                .core_clock_en   (),
                .energy_counter  (),
                .power_idle_hint (),
                .link_tx_push    (mesh_link_tx_push),
                .link_tx_core    (mesh_link_tx_core),
                .link_tx_neuron  (mesh_link_tx_neuron),
                .link_tx_payload (mesh_link_tx_payload),
                .link_tx_full    (mesh_link_tx_full),
                .link_rx_core    (mesh_link_rx_core),
                .link_rx_neuron  (mesh_link_rx_neuron),
                .link_rx_current (mesh_link_rx_current),
                .link_rx_pop     (mesh_link_rx_pop),
                .link_rx_empty   (mesh_link_rx_empty)
            );
        end else begin : gen_barrier_mesh
            neuromorphic_mesh #(
                .NUM_CORES      (NUM_CORES),
                .CORE_ID_BITS   (CORE_ID_BITS),
                .NUM_NEURONS    (NUM_NEURONS),
                .NEURON_BITS    (NEURON_BITS),
                .DATA_WIDTH     (DATA_WIDTH),
                .POOL_DEPTH     (POOL_DEPTH),
                .POOL_ADDR_BITS (POOL_ADDR_BITS),
                .COUNT_BITS     (COUNT_BITS),
                .REV_FANIN      (REV_FANIN),
                .REV_SLOT_BITS  (REV_SLOT_BITS),
                .THRESHOLD      (THRESHOLD),
                .LEAK_RATE      (LEAK_RATE),
                .REFRAC_CYCLES  (REFRAC_CYCLES),
                .ROUTE_FANOUT   (ROUTE_FANOUT),
                .ROUTE_SLOT_BITS(ROUTE_SLOT_BITS),
                .GLOBAL_ROUTE_SLOTS    (GLOBAL_ROUTE_SLOTS),
                .GLOBAL_ROUTE_SLOT_BITS(GLOBAL_ROUTE_SLOT_BITS),
                .CHIP_LINK_EN          (CHIP_LINK_EN)
            ) u_mesh (
                .clk               (clk),
                .rst_n             (rst_n),
                .start             (hi_mesh_start),
                .dvfs_stall        (hi_dvfs_stall),
                .learn_enable      (hi_learn_enable),
                .graded_enable     (hi_graded_enable),
                .dendritic_enable  (hi_dendritic_enable),
                .async_enable      (hi_async_enable),
                .threefactor_enable(hi_threefactor_enable),
                .noise_enable      (hi_noise_enable),
                .skip_idle_enable  (hi_skip_idle_enable),
                .scale_u_enable    (hi_scale_u_enable),
                .reward_value      (hi_reward_value),
                .prog_pool_we      (hi_prog_pool_we),
                .prog_pool_core    (hi_prog_pool_core),
                .prog_pool_addr    (hi_prog_pool_addr),
                .prog_pool_src     (hi_prog_pool_src),
                .prog_pool_target  (hi_prog_pool_target),
                .prog_pool_weight  (hi_prog_pool_weight),
                .prog_pool_comp    (hi_prog_pool_comp),
                .prog_index_we     (hi_prog_index_we),
                .prog_index_core   (hi_prog_index_core),
                .prog_index_neuron (hi_prog_index_neuron),
                .prog_index_base   (hi_prog_index_base),
                .prog_index_count  (hi_prog_index_count),
                .prog_index_format (hi_prog_index_format),
                .prog_route_we         (hi_prog_route_we),
                .prog_route_src_core   (hi_prog_route_src_core),
                .prog_route_src_neuron (hi_prog_route_src_neuron),
                .prog_route_slot       (hi_prog_route_slot),
                .prog_route_dest_core  (hi_prog_route_dest_core),
                .prog_route_dest_neuron(hi_prog_route_dest_neuron),
                .prog_route_weight     (hi_prog_route_weight),
                .prog_global_route_we          (hi_prog_global_route_we),
                .prog_global_route_src_core    (hi_prog_global_route_src_core),
                .prog_global_route_src_neuron  (hi_prog_global_route_src_neuron),
                .prog_global_route_slot        (hi_prog_global_route_slot),
                .prog_global_route_dest_core   (hi_prog_global_route_dest_core),
                .prog_global_route_dest_neuron (hi_prog_global_route_dest_neuron),
                .prog_global_route_weight      (hi_prog_global_route_weight),
                .prog_delay_we     (hi_prog_delay_we),
                .prog_delay_core   (hi_prog_delay_core),
                .prog_delay_addr   (hi_prog_delay_addr),
                .prog_delay_value  (hi_prog_delay_value),
                .prog_ucode_we     (hi_prog_ucode_we),
                .prog_ucode_core   (hi_prog_ucode_core),
                .prog_ucode_addr   (hi_prog_ucode_addr),
                .prog_ucode_data   (hi_prog_ucode_data),
                .prog_param_we     (hi_prog_param_we),
                .prog_param_core   (hi_prog_param_core),
                .prog_param_neuron (hi_prog_param_neuron),
                .prog_param_id     (hi_prog_param_id),
                .prog_param_value  (hi_prog_param_value),
                .probe_read        (hi_probe_read),
                .probe_core        (hi_probe_core),
                .probe_neuron      (hi_probe_neuron),
                .probe_state_id    (hi_probe_state_id),
                .probe_pool_addr   (hi_probe_pool_addr),
                .probe_data        (mesh_probe_data),
                .probe_valid       (mesh_probe_valid),
                .ext_valid         (hi_ext_valid),
                .ext_core          (hi_ext_core),
                .ext_neuron_id     (hi_ext_neuron_id),
                .ext_current       (hi_ext_current),
                .timestep_done     (mesh_timestep_done),
                .spike_valid_bus   (spike_valid_bus),
                .spike_id_bus      (spike_id_bus),
                .mesh_state_out    (mesh_state),
                .total_spikes      (mesh_total_spikes),
                .timestep_count    (mesh_timestep_count),
                .core_idle_bus     (),
                .core_clock_en   (),
                .energy_counter  (),
                .power_idle_hint (),
                .link_tx_push    (mesh_link_tx_push),
                .link_tx_core    (mesh_link_tx_core),
                .link_tx_neuron  (mesh_link_tx_neuron),
                .link_tx_payload (mesh_link_tx_payload),
                .link_tx_full    (mesh_link_tx_full),
                .link_rx_core    (mesh_link_rx_core),
                .link_rx_neuron  (mesh_link_rx_neuron),
                .link_rx_current (mesh_link_rx_current),
                .link_rx_pop     (mesh_link_rx_pop),
                .link_rx_empty   (mesh_link_rx_empty)
            );
        end
    endgenerate

    generate
        if (CHIP_LINK_EN) begin : gen_chip_link
            chip_link #(
                .CORE_ID_BITS (CORE_ID_BITS),
                .NEURON_BITS  (NEURON_BITS),
                .DATA_WIDTH   (DATA_WIDTH),
                .TX_DEPTH     (256),
                .RX_DEPTH     (256)
            ) u_chip_link (
                .clk            (clk),
                .rst_n          (rst_n),
                .tx_push        (mesh_link_tx_push),
                .tx_core        (mesh_link_tx_core),
                .tx_neuron      (mesh_link_tx_neuron),
                .tx_payload     (mesh_link_tx_payload),
                .tx_full        (mesh_link_tx_full),
                .rx_core        (mesh_link_rx_core),
                .rx_neuron      (mesh_link_rx_neuron),
                .rx_current     (mesh_link_rx_current),
                .rx_pop         (mesh_link_rx_pop),
                .rx_empty       (mesh_link_rx_empty),
                .link_tx_data   (link_tx_data),
                .link_tx_valid  (link_tx_valid),
                .link_tx_ready  (link_tx_ready),
                .link_rx_data   (link_rx_data),
                .link_rx_valid  (link_rx_valid),
                .link_rx_ready  (link_rx_ready)
            );
        end else begin : gen_no_chip_link
            assign mesh_link_tx_full  = 1'b0;
            assign mesh_link_rx_core  = {CORE_ID_BITS{1'b0}};
            assign mesh_link_rx_neuron = {NEURON_BITS{1'b0}};
            assign mesh_link_rx_current = {DATA_WIDTH{1'b0}};
            assign mesh_link_rx_empty = 1'b1;
            assign link_tx_data  = 8'd0;
            assign link_tx_valid = 1'b0;
            assign link_rx_ready = 1'b0;
        end
    endgenerate

endmodule
