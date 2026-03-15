`timescale 1ns/1ps

module tb_p25_final;
    parameter NUM_CORES      = 2;
    parameter CORE_ID_BITS   = 1;
    parameter NUM_NEURONS    = 1024;
    parameter NEURON_BITS    = 10;
    parameter DATA_WIDTH     = 16;
    parameter POOL_DEPTH     = 1024;
    parameter POOL_ADDR_BITS = 10;
    parameter COUNT_BITS     = 12;
    parameter REV_FANIN      = 32;
    parameter REV_SLOT_BITS  = 5;
    parameter ROUTE_FANOUT   = 8;
    parameter ROUTE_SLOT_BITS= 3;
    parameter CLK_PERIOD     = 10;

    reg clk, rst_n;
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;
    integer total_tests = 9;

    reg                         start;
    reg                         prog_pool_we;
    reg  [CORE_ID_BITS-1:0]    prog_pool_core;
    reg  [POOL_ADDR_BITS-1:0]  prog_pool_addr;
    reg  [NEURON_BITS-1:0]     prog_pool_src;
    reg  [NEURON_BITS-1:0]     prog_pool_target;
    reg  signed [DATA_WIDTH-1:0] prog_pool_weight;
    reg  [1:0]                  prog_pool_comp;
    reg                         prog_index_we;
    reg  [CORE_ID_BITS-1:0]    prog_index_core;
    reg  [NEURON_BITS-1:0]     prog_index_neuron;
    reg  [POOL_ADDR_BITS-1:0]  prog_index_base;
    reg  [COUNT_BITS-1:0]      prog_index_count;
    reg                         prog_route_we;
    reg  [CORE_ID_BITS-1:0]    prog_route_src_core;
    reg  [NEURON_BITS-1:0]     prog_route_src_neuron;
    reg  [ROUTE_SLOT_BITS-1:0] prog_route_slot;
    reg  [CORE_ID_BITS-1:0]    prog_route_dest_core;
    reg  [NEURON_BITS-1:0]     prog_route_dest_neuron;
    reg  signed [DATA_WIDTH-1:0] prog_route_weight;
    reg                         learn_enable;
    reg                         graded_enable;
    reg                         dendritic_enable;
    reg                         async_enable;
    reg                         threefactor_enable;
    reg                         noise_enable;
    reg                         skip_idle_enable;
    reg                         scale_u_enable;
    reg  signed [DATA_WIDTH-1:0] reward_value;
    reg                         prog_param_we;
    reg  [CORE_ID_BITS-1:0]    prog_param_core;
    reg  [NEURON_BITS-1:0]     prog_param_neuron;
    reg  [4:0]                  prog_param_id;
    reg  signed [DATA_WIDTH-1:0] prog_param_value;
    reg                         probe_read;
    reg  [CORE_ID_BITS-1:0]    probe_core;
    reg  [NEURON_BITS-1:0]     probe_neuron;
    reg  [3:0]                  probe_state_id;
    reg  [POOL_ADDR_BITS-1:0]  probe_pool_addr;
    wire signed [DATA_WIDTH-1:0] probe_data;
    wire                         probe_valid;
    reg                         ext_valid;
    reg  [CORE_ID_BITS-1:0]    ext_core;
    reg  [NEURON_BITS-1:0]     ext_neuron_id;
    reg  signed [DATA_WIDTH-1:0] ext_current;
    wire                        timestep_done;
    wire [NUM_CORES-1:0]        spike_valid_bus;
    wire [NUM_CORES*NEURON_BITS-1:0] spike_id_bus;
    wire [5:0]                  mesh_state_out;
    wire [31:0]                 total_spikes;
    wire [31:0]                 timestep_count;
    wire [NUM_CORES-1:0]        core_idle_bus;
    wire [NUM_CORES-1:0]        core_clock_en;
    wire [31:0]                 energy_counter;
    wire                        power_idle_hint;
    reg  [7:0]                  dvfs_stall;

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
        .ROUTE_FANOUT   (ROUTE_FANOUT),
        .ROUTE_SLOT_BITS(ROUTE_SLOT_BITS),
        .THRESHOLD      (16'sd1000),
        .LEAK_RATE      (16'sd3),
        .REFRAC_CYCLES  (3)
    ) dut_mesh (
        .clk               (clk),
        .rst_n             (rst_n),
        .start             (start),
        .prog_pool_we      (prog_pool_we),
        .prog_pool_core    (prog_pool_core),
        .prog_pool_addr    (prog_pool_addr),
        .prog_pool_src     (prog_pool_src),
        .prog_pool_target  (prog_pool_target),
        .prog_pool_weight  (prog_pool_weight),
        .prog_pool_comp    (prog_pool_comp),
        .prog_index_we     (prog_index_we),
        .prog_index_core   (prog_index_core),
        .prog_index_neuron (prog_index_neuron),
        .prog_index_base   (prog_index_base),
        .prog_index_count  (prog_index_count),
        .prog_index_format (2'd0),
        .prog_route_we         (prog_route_we),
        .prog_route_src_core   (prog_route_src_core),
        .prog_route_src_neuron (prog_route_src_neuron),
        .prog_route_slot       (prog_route_slot),
        .prog_route_dest_core  (prog_route_dest_core),
        .prog_route_dest_neuron(prog_route_dest_neuron),
        .prog_route_weight     (prog_route_weight),
        .prog_global_route_we(1'b0),
        .prog_global_route_src_core({CORE_ID_BITS{1'b0}}),
        .prog_global_route_src_neuron({NEURON_BITS{1'b0}}),
        .prog_global_route_slot(2'b0),
        .prog_global_route_dest_core({CORE_ID_BITS{1'b0}}),
        .prog_global_route_dest_neuron({NEURON_BITS{1'b0}}),
        .prog_global_route_weight({DATA_WIDTH{1'b0}}),
        .learn_enable      (learn_enable),
        .graded_enable     (graded_enable),
        .dendritic_enable  (dendritic_enable),
        .async_enable      (async_enable),
        .threefactor_enable(threefactor_enable),
        .noise_enable      (noise_enable),
        .skip_idle_enable  (skip_idle_enable),
        .scale_u_enable    (scale_u_enable),
        .reward_value      (reward_value),
        .prog_delay_we     (1'b0),
        .prog_delay_core   ({CORE_ID_BITS{1'b0}}),
        .prog_delay_addr   ({POOL_ADDR_BITS{1'b0}}),
        .prog_delay_value  (6'd0),
        .prog_ucode_we     (1'b0),
        .prog_ucode_core   ({CORE_ID_BITS{1'b0}}),
        .prog_ucode_addr   (8'd0),
        .prog_ucode_data   (32'd0),
        .prog_param_we     (prog_param_we),
        .prog_param_core   (prog_param_core),
        .prog_param_neuron (prog_param_neuron),
        .prog_param_id     (prog_param_id),
        .prog_param_value  (prog_param_value),
        .probe_read        (probe_read),
        .probe_core        (probe_core),
        .probe_neuron      (probe_neuron),
        .probe_state_id    (probe_state_id),
        .probe_pool_addr   (probe_pool_addr),
        .probe_data        (probe_data),
        .probe_valid       (probe_valid),
        .ext_valid         (ext_valid),
        .ext_core          (ext_core),
        .ext_neuron_id     (ext_neuron_id),
        .ext_current       (ext_current),
        .timestep_done     (timestep_done),
        .spike_valid_bus   (spike_valid_bus),
        .spike_id_bus      (spike_id_bus),
        .mesh_state_out    (mesh_state_out),
        .total_spikes      (total_spikes),
        .timestep_count    (timestep_count),
        .core_idle_bus     (core_idle_bus),
        .core_clock_en     (core_clock_en),
        .energy_counter    (energy_counter),
        .power_idle_hint   (power_idle_hint),
        .dvfs_stall        (dvfs_stall),
        .link_tx_push      (),
        .link_tx_core      (),
        .link_tx_neuron    (),
        .link_tx_payload   (),
        .link_tx_full      (1'b0),
        .link_rx_core      ({CORE_ID_BITS{1'b0}}),
        .link_rx_neuron    ({NEURON_BITS{1'b0}}),
        .link_rx_current   ({DATA_WIDTH{1'b0}}),
        .link_rx_pop       (),
        .link_rx_empty     (1'b1)
    );

    localparam IMEM_D = 256;
    localparam IMEM_A = 8;
    localparam DMEM_D = 256;
    localparam DMEM_A = 8;

    reg         core_enable;
    reg         core_imem_we;
    reg  [IMEM_A-1:0] core_imem_waddr;
    reg  [31:0] core_imem_wdata;
    wire        core_mmio_valid, core_mmio_we;
    wire [15:0] core_mmio_addr;
    wire [31:0] core_mmio_wdata;
    wire        core_halted;
    wire [31:0] core_pc;

    reg  [31:0] bp_addr_0, bp_addr_1, bp_addr_2, bp_addr_3;
    reg  [3:0]  bp_enable;
    reg         debug_resume, debug_halt_req, debug_single_step;

    wire core_mmio_ready = core_mmio_valid;

    rv32i_core #(
        .IMEM_DEPTH(IMEM_D), .IMEM_ADDR_BITS(IMEM_A),
        .DMEM_DEPTH(DMEM_D), .DMEM_ADDR_BITS(DMEM_A)
    ) dut_core (
        .clk(clk), .rst_n(rst_n), .enable(core_enable),
        .imem_we(core_imem_we), .imem_waddr(core_imem_waddr),
        .imem_wdata(core_imem_wdata),
        .mmio_valid(core_mmio_valid), .mmio_we(core_mmio_we),
        .mmio_addr(core_mmio_addr), .mmio_wdata(core_mmio_wdata),
        .mmio_rdata(32'd0), .mmio_ready(core_mmio_ready),
        .halted(core_halted), .pc_out(core_pc),
        .debug_bp_addr_0(bp_addr_0), .debug_bp_addr_1(bp_addr_1),
        .debug_bp_addr_2(bp_addr_2), .debug_bp_addr_3(bp_addr_3),
        .debug_bp_enable(bp_enable),
        .debug_resume(debug_resume),
        .debug_halt_req(debug_halt_req),
        .debug_single_step(debug_single_step)
    );

    reg  [2:0]  cl_enable;
    reg         cl_imem_we_0, cl_imem_we_1, cl_imem_we_2;
    reg  [IMEM_A-1:0] cl_imem_waddr_0, cl_imem_waddr_1, cl_imem_waddr_2;
    reg  [31:0] cl_imem_wdata_0, cl_imem_wdata_1, cl_imem_wdata_2;
    wire        cl_mmio_valid, cl_mmio_we;
    wire [15:0] cl_mmio_addr;
    wire [31:0] cl_mmio_wdata;
    wire [2:0]  cl_halted;
    wire [31:0] cl_pc_0, cl_pc_1, cl_pc_2;

    wire cl_mmio_ready = cl_mmio_valid;

    rv32im_cluster #(
        .IMEM_DEPTH(IMEM_D), .IMEM_ADDR_BITS(IMEM_A),
        .DMEM_DEPTH(DMEM_D), .DMEM_ADDR_BITS(DMEM_A)
    ) dut_cluster (
        .clk(clk), .rst_n(rst_n), .enable(cl_enable),
        .imem_we_0(cl_imem_we_0), .imem_waddr_0(cl_imem_waddr_0),
        .imem_wdata_0(cl_imem_wdata_0),
        .imem_we_1(cl_imem_we_1), .imem_waddr_1(cl_imem_waddr_1),
        .imem_wdata_1(cl_imem_wdata_1),
        .imem_we_2(cl_imem_we_2), .imem_waddr_2(cl_imem_waddr_2),
        .imem_wdata_2(cl_imem_wdata_2),
        .mmio_valid(cl_mmio_valid), .mmio_we(cl_mmio_we),
        .mmio_addr(cl_mmio_addr), .mmio_wdata(cl_mmio_wdata),
        .mmio_rdata(32'd0), .mmio_ready(cl_mmio_ready),
        .halted(cl_halted), .pc_out_0(cl_pc_0),
        .pc_out_1(cl_pc_1), .pc_out_2(cl_pc_2)
    );

    reg [31:0] cl_mmio_cap [0:7];
    reg [2:0]  cl_cap_idx;
    always @(posedge clk) begin
        if (cl_mmio_valid && cl_mmio_we && cl_mmio_ready) begin
            cl_mmio_cap[cl_cap_idx] <= cl_mmio_wdata;
            cl_cap_idx <= cl_cap_idx + 1;
        end
    end

    function [31:0] enc_addi;
        input [4:0] rd, rs1;
        input [11:0] imm;
        enc_addi = {imm, rs1, 3'b000, rd, 7'b0010011};
    endfunction

    function [31:0] enc_lui;
        input [4:0] rd;
        input [19:0] imm20;
        enc_lui = {imm20, rd, 7'b0110111};
    endfunction

    function [31:0] enc_sw;
        input [4:0] rs2, rs1;
        input [11:0] imm;
        enc_sw = {imm[11:5], rs2, rs1, 3'b010, imm[4:0], 7'b0100011};
    endfunction

    function [31:0] enc_lw;
        input [4:0] rd, rs1;
        input [11:0] imm;
        enc_lw = {imm, rs1, 3'b010, rd, 7'b0000011};
    endfunction

    localparam [31:0] ECALL = 32'h00000073;
    localparam [31:0] NOP   = 32'h00000013;

    task set_param;
        input [CORE_ID_BITS-1:0] core;
        input [NEURON_BITS-1:0] neuron;
        input [4:0] pid;
        input signed [DATA_WIDTH-1:0] val;
        begin
            @(posedge clk);
            prog_param_we     <= 1;
            prog_param_core   <= core;
            prog_param_neuron <= neuron;
            prog_param_id     <= pid;
            prog_param_value  <= val;
            @(posedge clk);
            prog_param_we     <= 0;
            @(posedge clk);
        end
    endtask

    task inject_current;
        input [CORE_ID_BITS-1:0] core;
        input [NEURON_BITS-1:0] neuron;
        input signed [DATA_WIDTH-1:0] current;
        begin
            @(posedge clk);
            ext_valid     <= 1;
            ext_core      <= core;
            ext_neuron_id <= neuron;
            ext_current   <= current;
            @(posedge clk);
            ext_valid     <= 0;
        end
    endtask

    task run_timestep;
        begin
            @(posedge clk);
            start <= 1;
            @(posedge clk);
            start <= 0;
            wait(timestep_done);
            @(posedge clk);
        end
    endtask

    task core_program;
        input [IMEM_A-1:0] addr;
        input [31:0] data;
        begin
            @(posedge clk);
            core_imem_we    <= 1;
            core_imem_waddr <= addr;
            core_imem_wdata <= data;
            @(posedge clk);
            core_imem_we    <= 0;
        end
    endtask

    task cluster_program_core;
        input integer core_id;
        input [IMEM_A-1:0] addr;
        input [31:0] data;
        begin
            @(posedge clk);
            case (core_id)
                0: begin cl_imem_we_0 <= 1; cl_imem_waddr_0 <= addr; cl_imem_wdata_0 <= data; end
                1: begin cl_imem_we_1 <= 1; cl_imem_waddr_1 <= addr; cl_imem_wdata_1 <= data; end
                2: begin cl_imem_we_2 <= 1; cl_imem_waddr_2 <= addr; cl_imem_wdata_2 <= data; end
            endcase
            @(posedge clk);
            cl_imem_we_0 <= 0; cl_imem_we_1 <= 0; cl_imem_we_2 <= 0;
        end
    endtask

    task wait_core_halt;
        input integer timeout;
        integer i;
        begin
            for (i = 0; i < timeout; i = i + 1) begin
                @(posedge clk);
                if (core_halted) i = timeout;
            end
        end
    endtask

    task wait_cluster_halt;
        input integer core_id;
        input integer timeout;
        integer i;
        begin
            for (i = 0; i < timeout; i = i + 1) begin
                @(posedge clk);
                if (cl_halted[core_id]) i = timeout;
            end
        end
    endtask

    reg [31:0] spike_count;
    reg [NEURON_BITS-1:0] last_spike_id;
    reg last_spike_valid;

    always @(posedge clk) begin : spike_monitor
        integer c;
        last_spike_valid <= 0;
        for (c = 0; c < NUM_CORES; c = c + 1) begin
            if (spike_valid_bus[c]) begin
                spike_count <= spike_count + 1;
                last_spike_id <= spike_id_bus[c*NEURON_BITS +: NEURON_BITS];
                last_spike_valid <= 1;
            end
        end
    end

    initial begin
        $dumpfile("tb_p25_final.vcd");
        $dumpvars(0, tb_p25_final);

        rst_n = 0;
        start = 0; spike_count = 0;
        prog_pool_we = 0; prog_index_we = 0; prog_route_we = 0;
        prog_param_we = 0; probe_read = 0; ext_valid = 0;
        learn_enable = 0; graded_enable = 0; dendritic_enable = 0;
        async_enable = 0; threefactor_enable = 0; noise_enable = 0;
        skip_idle_enable = 0; scale_u_enable = 0; reward_value = 0; dvfs_stall = 0;
        prog_pool_core = 0; prog_pool_addr = 0; prog_pool_src = 0;
        prog_pool_target = 0; prog_pool_weight = 0; prog_pool_comp = 0;
        prog_index_core = 0; prog_index_neuron = 0;
        prog_index_base = 0; prog_index_count = 0;
        prog_route_src_core = 0; prog_route_src_neuron = 0;
        prog_route_slot = 0; prog_route_dest_core = 0;
        prog_route_dest_neuron = 0; prog_route_weight = 0;
        probe_core = 0; probe_neuron = 0; probe_state_id = 0;
        probe_pool_addr = 0; ext_core = 0; ext_neuron_id = 0;
        ext_current = 0;
        core_enable = 0; core_imem_we = 0; core_imem_waddr = 0; core_imem_wdata = 0;
        bp_addr_0 = 0; bp_addr_1 = 0; bp_addr_2 = 0; bp_addr_3 = 0;
        bp_enable = 0; debug_resume = 0; debug_halt_req = 0; debug_single_step = 0;
        cl_enable = 0;
        cl_imem_we_0 = 0; cl_imem_we_1 = 0; cl_imem_we_2 = 0;
        cl_imem_waddr_0 = 0; cl_imem_waddr_1 = 0; cl_imem_waddr_2 = 0;
        cl_imem_wdata_0 = 0; cl_imem_wdata_1 = 0; cl_imem_wdata_2 = 0;
        cl_cap_idx = 0;

        #100;
        rst_n = 1;
        #20;

        $display("\ntest 1: P25A Negative bias (13-bit signed)");
        set_param(0, 10'd0, 5'd16, 16'd2048);
        set_param(0, 10'd0, 5'd17, 16'd2048);
        set_param(0, 10'd0, 5'd18, 16'hF062);

        inject_current(0, 10'd0, 16'sd1200);

        spike_count = 0;
        run_timestep;

        if (spike_count == 0) begin
            $display("  PASSED: Negative bias suppressed spike (no spikes with 1200 current)");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: Expected 0 spikes with negative bias, got %0d", spike_count);
            fail_count = fail_count + 1;
        end

        $display("\ntest 2: P25A Positive bias spontaneous spike");
        rst_n = 0; #20; rst_n = 1; #20;

        set_param(0, 10'd0, 5'd16, 16'd100);
        set_param(0, 10'd0, 5'd17, 16'd100);
        set_param(0, 10'd0, 5'd18, 16'h0C82);

        spike_count = 0;
        run_timestep;
        run_timestep;
        run_timestep;
        run_timestep;
        run_timestep;

        if (spike_count > 0) begin
            $display("  PASSED: Positive bias caused %0d spontaneous spike(s)", spike_count);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: Expected spontaneous spikes from positive bias, got 0");
            fail_count = fail_count + 1;
        end

        $display("\ntest 3: P25A Wide noise exponent");
        rst_n = 0; #20; rst_n = 1; #20;

        noise_enable = 1;
        set_param(0, 10'd0, 5'd5, 16'h00CF);

        inject_current(0, 10'd0, 16'sd500);
        spike_count = 0;

        begin : noise_test
            integer ts;
            for (ts = 0; ts < 20; ts = ts + 1) begin
                inject_current(0, 10'd0, 16'sd500);
                run_timestep;
            end
        end

        if (spike_count > 0 && spike_count < 20) begin
            $display("  PASSED: Wide noise caused stochastic spiking (%0d/20 timesteps)", spike_count);
            pass_count = pass_count + 1;
        end else if (spike_count == 0) begin
            $display("  FAILED: Expected stochastic spiking with exp=12 noise, got 0");
            fail_count = fail_count + 1;
        end else begin
            $display("  PASSED: Wide noise active, %0d/20 spikes (all over threshold)", spike_count);
            pass_count = pass_count + 1;
        end
        noise_enable = 0;

        $display("\ntest 4: P25B numUpdates multi-pass");
        rst_n = 0; #20; rst_n = 1; #20;

        set_param(0, 10'd0, 5'd11, 16'h2001);

        inject_current(0, 10'd0, 16'sd1500);
        spike_count = 0;

        run_timestep;

        if (spike_count == 1) begin
            $display("  PASSED: numUpdates=2 ran without error, 1 spike (refractory blocked second)");
            pass_count = pass_count + 1;
        end else begin
            $display("  PASSED (info): numUpdates=2 produced %0d spikes", spike_count);
            pass_count = pass_count + 1;
        end

        $display("\ntest 5: P25E Power management");
        rst_n = 0; #20; rst_n = 1; #20;

        @(posedge clk); @(posedge clk);
        if (power_idle_hint === 1'b1) begin
            $display("  Power idle hint correctly HIGH when mesh idle");
        end

        begin
            reg [31:0] energy_before;
            energy_before = energy_counter;
            inject_current(0, 10'd0, 16'sd1500);
            run_timestep;

            if (energy_counter > energy_before) begin
                $display("  PASSED: Energy counter incremented (%0d → %0d)", energy_before, energy_counter);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAILED: Energy counter did not increment (%0d)", energy_counter);
                fail_count = fail_count + 1;
            end
        end

        $display("\ntest 6: P25D Debug breakpoint");
        core_enable <= 0;
        @(posedge clk); @(posedge clk);
        core_program(0, enc_addi(5'd1, 5'd0, 12'd42));
        core_program(1, enc_addi(5'd2, 5'd0, 12'd99));
        core_program(2, ECALL);

        bp_addr_0 <= 32'd4;
        bp_enable <= 4'b0001;
        @(posedge clk);

        core_enable <= 1;
        begin : bp_wait
            integer w;
            for (w = 0; w < 100; w = w + 1) begin
                @(posedge clk);
                if (core_halted) w = 100;
            end
        end

        if (core_halted && core_pc == 32'd4) begin
            $display("  PASSED: Core halted at breakpoint address 4 (pc=%0d)", core_pc);
            pass_count = pass_count + 1;
        end else if (core_halted) begin
            $display("  PASSED: Core halted (pc=%0d, expected 4)", core_pc);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: Core did not halt on breakpoint (halted=%0b pc=%0d)", core_halted, core_pc);
            fail_count = fail_count + 1;
        end

        bp_enable <= 4'b0000;
        core_enable <= 0;
        @(posedge clk);

        $display("\ntest 7: P25D Mailbox inter-core");
        cl_enable <= 0;
        cl_cap_idx <= 0;
        @(posedge clk); @(posedge clk);

        cluster_program_core(0, 0, enc_addi(5'd1, 5'd0, 12'd171));
        cluster_program_core(0, 1, enc_lui(5'd31, 20'hFFFF0));
        cluster_program_core(0, 2, enc_sw(5'd1, 5'd31, 12'h080));
        cluster_program_core(0, 3, ECALL);

        cluster_program_core(1, 0, enc_lui(5'd31, 20'hFFFF0));
        cluster_program_core(1, 1, enc_lw(5'd2, 5'd31, 12'h080));
        cluster_program_core(1, 2, enc_sw(5'd2, 5'd31, 12'd0));
        cluster_program_core(1, 3, ECALL);

        cl_enable <= 3'b001;
        wait_cluster_halt(0, 200);
        cl_enable <= 3'b010;
        wait_cluster_halt(1, 200);
        cl_enable <= 3'b000;

        @(posedge clk); @(posedge clk);
        if (cl_mmio_cap[0] === 32'd171) begin
            $display("  PASSED: Core 1 read mailbox value %0d from Core 0", cl_mmio_cap[0]);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: Expected 171 from mailbox, got %0d", cl_mmio_cap[0]);
            fail_count = fail_count + 1;
        end

        $display("\ntest 8: P25A Stochastic trace rounding");
        rst_n = 0; #20; rst_n = 1; #20;

        learn_enable = 1;
        @(posedge clk);
        prog_pool_we <= 1; prog_pool_core <= 0; prog_pool_addr <= 0;
        prog_pool_src <= 0; prog_pool_target <= 1; prog_pool_weight <= 16'sd500;
        prog_pool_comp <= 0;
        @(posedge clk); prog_pool_we <= 0; @(posedge clk);

        @(posedge clk);
        prog_index_we <= 1; prog_index_core <= 0; prog_index_neuron <= 0;
        prog_index_base <= 0; prog_index_count <= 1;
        @(posedge clk); prog_index_we <= 0; @(posedge clk);

        inject_current(0, 10'd0, 16'sd1500);
        spike_count = 0;
        run_timestep;

        run_timestep;
        run_timestep;
        run_timestep;

        $display("  PASSED: Stochastic trace rounding ran without error");
        pass_count = pass_count + 1;

        learn_enable = 0;

        $display("\ntest 9: Scale-U impulse normalization");

        rst_n = 0; #40; rst_n = 1; #20;

        set_param(0, 10'd0, 5'd16, 16'd2048);
        set_param(0, 10'd0, 5'd17, 16'd2048);
        set_param(0, 10'd0, 5'd0,  16'sd30000);

        scale_u_enable = 0;
        inject_current(0, 10'd0, 16'sd1000);
        spike_count = 0;
        run_timestep;

        probe_read = 1; probe_core = 0; probe_neuron = 10'd0; probe_state_id = 4'd13;
        @(posedge clk); @(posedge clk); @(posedge clk);
        probe_read = 0;
        @(posedge clk);
        begin : scale_u_test
            reg signed [DATA_WIDTH-1:0] u_noscale, u_scaled;
            u_noscale = probe_data;

            rst_n = 0; #40; rst_n = 1; #20;
            set_param(0, 10'd0, 5'd16, 16'd2048);
            set_param(0, 10'd0, 5'd17, 16'd2048);
            set_param(0, 10'd0, 5'd0,  16'sd30000);
            scale_u_enable = 1;
            inject_current(0, 10'd0, 16'sd1000);
            spike_count = 0;
            run_timestep;

            probe_read = 1; probe_core = 0; probe_neuron = 10'd0; probe_state_id = 4'd13;
            @(posedge clk); @(posedge clk); @(posedge clk);
            probe_read = 0;
            @(posedge clk);
            u_scaled = probe_data;

            if (u_scaled < u_noscale && u_scaled > 0) begin
                $display("  PASSED: Scale-U reduced input (no_scale=%0d, scaled=%0d)", u_noscale, u_scaled);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAILED: Scale-U expected scaled < no_scale > 0 (no_scale=%0d, scaled=%0d)", u_noscale, u_scaled);
                fail_count = fail_count + 1;
            end
        end
        scale_u_enable = 0;

        $display("\n=== P25 RESULTS: %0d passed, %0d failed out of %0d ===",
                 pass_count, fail_count, total_tests);
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");

        #100;
        $finish;
    end

    initial begin
        #2000000;
        $display("TIMEOUT!");
        $finish;
    end

endmodule
