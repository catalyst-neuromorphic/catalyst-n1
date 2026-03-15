`timescale 1ns/1ps

module tb_p23c_scale;

    reg clk, rst_n;
    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;

    localparam CIB = 4, NN = 16, NB = 4, DW = 16;
    localparam PD = 65536, PAB = 16, NC = 4;

    reg nm_start, nm_prog_pool_we, nm_prog_index_we, nm_prog_route_we;
    reg nm_prog_param_we, nm_ext_valid, nm_probe_read;
    reg [CIB-1:0] nm_prog_pool_core, nm_prog_index_core, nm_prog_route_src_core;
    reg [CIB-1:0] nm_prog_route_dest_core, nm_prog_param_core, nm_ext_core, nm_probe_core;
    reg [PAB-1:0] nm_prog_pool_addr, nm_prog_index_base, nm_probe_pool_addr;
    reg [NB-1:0]  nm_prog_pool_src, nm_prog_pool_target, nm_prog_index_neuron;
    reg [NB-1:0]  nm_prog_route_src_neuron, nm_prog_route_dest_neuron;
    reg [NB-1:0]  nm_prog_param_neuron, nm_ext_neuron_id, nm_probe_neuron;
    reg signed [DW-1:0] nm_prog_pool_weight, nm_prog_route_weight;
    reg signed [DW-1:0] nm_prog_param_value, nm_ext_current;
    reg [1:0]  nm_prog_pool_comp, nm_prog_index_format;
    reg [9:0]  nm_prog_index_count;
    reg [2:0]  nm_prog_route_slot;
    reg [4:0]  nm_prog_param_id, nm_probe_state_id;

    wire signed [DW-1:0] nm_probe_data;
    wire nm_probe_valid, nm_timestep_done;

    async_noc_mesh #(
        .NUM_CORES(NC), .CORE_ID_BITS(CIB),
        .NUM_NEURONS(NN), .NEURON_BITS(NB),
        .DATA_WIDTH(DW), .POOL_DEPTH(PD), .POOL_ADDR_BITS(PAB),
        .COUNT_BITS(10), .THRESHOLD(16'sd500),
        .LEAK_RATE(16'sd0), .REFRAC_CYCLES(0),
        .DUAL_NOC(1), .MESH_X(2), .MESH_Y(2)
    ) noc (
        .clk(clk), .rst_n(rst_n), .start(nm_start),
        .prog_pool_we(nm_prog_pool_we), .prog_pool_core(nm_prog_pool_core),
        .prog_pool_addr(nm_prog_pool_addr), .prog_pool_src(nm_prog_pool_src),
        .prog_pool_target(nm_prog_pool_target), .prog_pool_weight(nm_prog_pool_weight),
        .prog_pool_comp(nm_prog_pool_comp),
        .prog_index_we(nm_prog_index_we), .prog_index_core(nm_prog_index_core),
        .prog_index_neuron(nm_prog_index_neuron), .prog_index_base(nm_prog_index_base),
        .prog_index_count(nm_prog_index_count), .prog_index_format(nm_prog_index_format),
        .prog_route_we(nm_prog_route_we),
        .prog_route_src_core(nm_prog_route_src_core),
        .prog_route_src_neuron(nm_prog_route_src_neuron),
        .prog_route_slot(nm_prog_route_slot),
        .prog_route_dest_core(nm_prog_route_dest_core),
        .prog_route_dest_neuron(nm_prog_route_dest_neuron),
        .prog_route_weight(nm_prog_route_weight),
        .prog_global_route_we(1'b0),
        .prog_global_route_src_core(0), .prog_global_route_src_neuron(0),
        .prog_global_route_slot(0), .prog_global_route_dest_core(0),
        .prog_global_route_dest_neuron(0), .prog_global_route_weight(0),
        .learn_enable(1'b0), .graded_enable(1'b0), .dendritic_enable(1'b0),
        .async_enable(1'b0), .threefactor_enable(1'b0), .noise_enable(1'b0),
        .skip_idle_enable(1'b0), .scale_u_enable(1'b0), .reward_value(16'sd0),
        .prog_delay_we(1'b0), .prog_delay_core(0), .prog_delay_addr(0), .prog_delay_value(0),
        .prog_ucode_we(1'b0), .prog_ucode_core(0), .prog_ucode_addr(0), .prog_ucode_data(0),
        .prog_param_we(nm_prog_param_we), .prog_param_core(nm_prog_param_core),
        .prog_param_neuron(nm_prog_param_neuron), .prog_param_id(nm_prog_param_id),
        .prog_param_value(nm_prog_param_value),
        .ext_valid(nm_ext_valid), .ext_core(nm_ext_core),
        .ext_neuron_id(nm_ext_neuron_id), .ext_current(nm_ext_current),
        .probe_read(nm_probe_read), .probe_core(nm_probe_core),
        .probe_neuron(nm_probe_neuron), .probe_state_id(nm_probe_state_id),
        .probe_pool_addr(nm_probe_pool_addr),
        .probe_data(nm_probe_data), .probe_valid(nm_probe_valid),
        .timestep_done(nm_timestep_done),
        .spike_valid_bus(), .spike_id_bus(),
        .mesh_state_out(), .total_spikes(), .timestep_count(),
        .core_idle_bus(),
        .link_tx_push(), .link_tx_core(), .link_tx_neuron(), .link_tx_payload(),
        .link_tx_full(1'b0),
        .link_rx_core(0), .link_rx_neuron(0), .link_rx_current(0),
        .link_rx_pop(), .link_rx_empty(1'b1)
    );

    localparam MCR_CB = 14;

    reg mcr_tx_push, mcr_rx_pop;
    reg [MCR_CB-1:0] mcr_tx_dest;
    reg [6:0]  mcr_tx_core;
    reg [9:0]  mcr_tx_neuron;
    reg [7:0]  mcr_tx_payload;
    wire mcr_tx_full, mcr_rx_empty;
    wire [MCR_CB-1:0] mcr_rx_src;
    wire [6:0]  mcr_rx_core;
    wire [9:0]  mcr_rx_neuron;
    wire signed [15:0] mcr_rx_current;

    wire [7:0] mcr_link_data;
    wire       mcr_link_valid;

    multi_chip_router #(
        .NUM_LINKS(1), .CHIP_ID_BITS(MCR_CB),
        .CORE_ID_BITS(7), .NEURON_BITS(10),
        .DATA_WIDTH(16), .TX_DEPTH(16), .RX_DEPTH(16)
    ) mcr (
        .clk(clk), .rst_n(rst_n),
        .my_chip_id(14'd42),
        .tx_push(mcr_tx_push), .tx_dest_chip(mcr_tx_dest),
        .tx_core(mcr_tx_core), .tx_neuron(mcr_tx_neuron),
        .tx_payload(mcr_tx_payload), .tx_full(mcr_tx_full),
        .rx_src_chip(mcr_rx_src), .rx_core(mcr_rx_core),
        .rx_neuron(mcr_rx_neuron), .rx_current(mcr_rx_current),
        .rx_pop(mcr_rx_pop), .rx_empty(mcr_rx_empty),
        .link_tx_data(mcr_link_data), .link_tx_valid(mcr_link_valid),
        .link_tx_ready(1'b1),
        .link_rx_data(mcr_link_data),
        .link_rx_valid(mcr_link_valid),
        .link_rx_ready()
    );

    task clear_inputs;
    begin
        nm_start = 0; nm_prog_pool_we = 0; nm_prog_index_we = 0;
        nm_prog_route_we = 0; nm_prog_param_we = 0;
        nm_ext_valid = 0; nm_probe_read = 0;
        mcr_tx_push = 0; mcr_rx_pop = 0;
    end
    endtask

    task prog_param(input [CIB-1:0] core, input [NB-1:0] neuron,
                    input [4:0] pid, input signed [DW-1:0] val);
    begin
        @(posedge clk);
        nm_prog_param_we = 1; nm_prog_param_core = core;
        nm_prog_param_neuron = neuron; nm_prog_param_id = pid;
        nm_prog_param_value = val;
        @(posedge clk); nm_prog_param_we = 0;
    end
    endtask

    task prog_pool(input [CIB-1:0] core, input [PAB-1:0] addr,
                   input [NB-1:0] src, input [NB-1:0] target,
                   input signed [DW-1:0] weight);
    begin
        @(posedge clk);
        nm_prog_pool_we = 1; nm_prog_pool_core = core;
        nm_prog_pool_addr = addr; nm_prog_pool_src = src;
        nm_prog_pool_target = target; nm_prog_pool_weight = weight;
        nm_prog_pool_comp = 0;
        @(posedge clk); nm_prog_pool_we = 0;
    end
    endtask

    task prog_index(input [CIB-1:0] core, input [NB-1:0] neuron,
                    input [PAB-1:0] base, input [9:0] count);
    begin
        @(posedge clk);
        nm_prog_index_we = 1; nm_prog_index_core = core;
        nm_prog_index_neuron = neuron; nm_prog_index_base = base;
        nm_prog_index_count = count; nm_prog_index_format = 2'd0;
        @(posedge clk); nm_prog_index_we = 0;
    end
    endtask

    task prog_route(input [CIB-1:0] src_core, input [NB-1:0] src_nrn,
                    input [2:0] slot,
                    input [CIB-1:0] dst_core, input [NB-1:0] dst_nrn,
                    input signed [DW-1:0] weight);
    begin
        @(posedge clk);
        nm_prog_route_we = 1;
        nm_prog_route_src_core = src_core; nm_prog_route_src_neuron = src_nrn;
        nm_prog_route_slot = slot;
        nm_prog_route_dest_core = dst_core; nm_prog_route_dest_neuron = dst_nrn;
        nm_prog_route_weight = weight;
        @(posedge clk); nm_prog_route_we = 0;
    end
    endtask

    task inject(input [CIB-1:0] core, input [NB-1:0] neuron,
                input signed [DW-1:0] current);
    begin
        @(posedge clk);
        nm_ext_valid = 1; nm_ext_core = core;
        nm_ext_neuron_id = neuron; nm_ext_current = current;
        @(posedge clk); nm_ext_valid = 0;
    end
    endtask

    task run_timestep;
    begin
        @(posedge clk); nm_start = 1;
        @(posedge clk); nm_start = 0;
        wait(nm_timestep_done);
        repeat(5) @(posedge clk);
    end
    endtask

    task probe_check(input [CIB-1:0] core, input [NB-1:0] neuron,
                     input [4:0] sid, input signed [DW-1:0] expected,
                     input [255:0] label);
    begin
        @(posedge clk);
        nm_probe_read = 1; nm_probe_core = core;
        nm_probe_neuron = neuron; nm_probe_state_id = sid;
        nm_probe_pool_addr = 0;
        @(posedge clk); nm_probe_read = 0;
        repeat(3) @(posedge clk);
        if (nm_probe_data == expected) begin
            $display("PASSED: %0s (got %0d)", label, nm_probe_data);
            pass_count = pass_count + 1;
        end else begin
            $display("FAILED: %0s - expected %0d, got %0d", label, expected, nm_probe_data);
            fail_count = fail_count + 1;
        end
    end
    endtask

    initial begin
        $display("=== P23C Scale Parity Tests ===");
        rst_n = 0;
        clear_inputs;
        nm_prog_pool_core = 0; nm_prog_pool_addr = 0;
        nm_prog_pool_src = 0; nm_prog_pool_target = 0;
        nm_prog_pool_weight = 0; nm_prog_pool_comp = 0;
        nm_prog_index_core = 0; nm_prog_index_neuron = 0;
        nm_prog_index_base = 0; nm_prog_index_count = 0; nm_prog_index_format = 0;
        nm_prog_route_src_core = 0; nm_prog_route_src_neuron = 0;
        nm_prog_route_slot = 0; nm_prog_route_dest_core = 0;
        nm_prog_route_dest_neuron = 0; nm_prog_route_weight = 0;
        nm_prog_param_core = 0; nm_prog_param_neuron = 0;
        nm_prog_param_id = 0; nm_prog_param_value = 0;
        nm_ext_core = 0; nm_ext_neuron_id = 0; nm_ext_current = 0;
        nm_probe_core = 0; nm_probe_neuron = 0;
        nm_probe_state_id = 0; nm_probe_pool_addr = 0;
        mcr_tx_dest = 0; mcr_tx_core = 0; mcr_tx_neuron = 0; mcr_tx_payload = 0;

        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);

        prog_param(4'd0, 4'd0, 5'd0, 16'sd10);
        prog_param(4'd1, 4'd0, 5'd0, 16'sd10);

        prog_pool(4'd0, 16'd50000, 4'd0, 4'd1, 16'sd123);
        prog_index(4'd0, 4'd0, 16'd50000, 10'd1);

        prog_route(4'd0, 4'd0, 3'd0, 4'd3, 4'd2, 16'sd100);
        prog_route(4'd1, 4'd0, 3'd0, 4'd2, 4'd2, 16'sd200);

        repeat(5) @(posedge clk);

        inject(4'd0, 4'd0, 16'sd600);
        inject(4'd1, 4'd0, 16'sd600);
        repeat(3) @(posedge clk);
        run_timestep;

        run_timestep;

        probe_check(4'd0, 4'd1, 5'd0, 16'sd123, "T1: Pool depth 65K synapse@50000");

        probe_check(4'd3, 4'd2, 5'd0, 16'sd100, "T2: Dual NoC netA core0->core3");

        probe_check(4'd2, 4'd2, 5'd0, 16'sd200, "T3: Dual NoC netB core1->core2");

        @(posedge clk);
        mcr_tx_push = 1;
        mcr_tx_dest = 14'd12345;
        mcr_tx_core = 7'd99;
        mcr_tx_neuron = 10'd511;
        mcr_tx_payload = 8'd128;
        @(posedge clk); mcr_tx_push = 0;

        repeat(50) @(posedge clk);

        if (!mcr_rx_empty) begin
            if (mcr_rx_src == 14'd42 && mcr_rx_core == 7'd99 &&
                mcr_rx_neuron == 10'd511 && mcr_rx_current[7:0] == 8'd128) begin
                $display("PASSED: T4: Wide chip 14-bit loopback (src=%0d core=%0d nrn=%0d pay=%0d)",
                    mcr_rx_src, mcr_rx_core, mcr_rx_neuron, mcr_rx_current[7:0]);
                pass_count = pass_count + 1;
            end else begin
                $display("FAILED: T4: src=%0d(exp42) core=%0d(exp99) nrn=%0d(exp511) cur=%0d(exp128)",
                    mcr_rx_src, mcr_rx_core, mcr_rx_neuron, mcr_rx_current);
                fail_count = fail_count + 1;
            end
        end else begin
            $display("FAILED: T4: RX FIFO empty after loopback");
            fail_count = fail_count + 1;
        end

        $display("=== P23C RESULTS: %0d passed, %0d failed ===", pass_count, fail_count);
        if (fail_count == 0)
            $display("ALL P23C TESTS PASSED");
        $finish;
    end

    initial begin
        #5000000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
