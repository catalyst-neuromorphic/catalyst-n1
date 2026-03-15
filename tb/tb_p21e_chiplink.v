// ============================================================================
// Testbench: P21E - Multi-Chip Spike Interface
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

`timescale 1ps/1ps

module tb_p21e_chiplink;

    localparam NUM_CORES    = 4;
    localparam CORE_ID_BITS = 2;
    localparam NUM_NEURONS  = 1024;
    localparam NEURON_BITS  = 10;
    localparam DATA_WIDTH   = 16;
    localparam POOL_DEPTH   = 1024;
    localparam POOL_ADDR_BITS = 10;
    localparam COUNT_BITS   = 10;
    localparam THRESHOLD    = 16'sd1000;
    localparam LEAK_RATE    = 16'sd3;
    localparam ROUTE_FANOUT = 8;
    localparam ROUTE_SLOT_BITS = 3;
    localparam GLOBAL_ROUTE_SLOTS = 4;
    localparam GLOBAL_ROUTE_SLOT_BITS = 2;

    reg clk, rst_n;
    always #5000 clk = ~clk;

    reg start;
    reg prog_pool_we;
    reg [CORE_ID_BITS-1:0] prog_pool_core;
    reg [POOL_ADDR_BITS-1:0] prog_pool_addr;
    reg [NEURON_BITS-1:0] prog_pool_src, prog_pool_target;
    reg signed [DATA_WIDTH-1:0] prog_pool_weight;
    reg [1:0] prog_pool_comp;

    reg prog_index_we;
    reg [CORE_ID_BITS-1:0] prog_index_core;
    reg [NEURON_BITS-1:0] prog_index_neuron;
    reg [POOL_ADDR_BITS-1:0] prog_index_base;
    reg [COUNT_BITS-1:0] prog_index_count;
    reg [1:0] prog_index_format;

    reg prog_route_we;
    reg [CORE_ID_BITS-1:0] prog_route_src_core;
    reg [NEURON_BITS-1:0] prog_route_src_neuron;
    reg [ROUTE_SLOT_BITS-1:0] prog_route_slot;
    reg [CORE_ID_BITS-1:0] prog_route_dest_core;
    reg [NEURON_BITS-1:0] prog_route_dest_neuron;
    reg signed [DATA_WIDTH-1:0] prog_route_weight;

    reg prog_global_route_we;
    reg [CORE_ID_BITS-1:0] prog_global_route_src_core;
    reg [NEURON_BITS-1:0] prog_global_route_src_neuron;
    reg [GLOBAL_ROUTE_SLOT_BITS-1:0] prog_global_route_slot;
    reg [CORE_ID_BITS-1:0] prog_global_route_dest_core;
    reg [NEURON_BITS-1:0] prog_global_route_dest_neuron;
    reg signed [DATA_WIDTH-1:0] prog_global_route_weight;

    reg learn_enable, graded_enable, dendritic_enable, async_enable;
    reg threefactor_enable, noise_enable, skip_idle_enable;
    reg signed [DATA_WIDTH-1:0] reward_value;

    reg prog_delay_we;
    reg [CORE_ID_BITS-1:0] prog_delay_core;
    reg [POOL_ADDR_BITS-1:0] prog_delay_addr;
    reg [5:0] prog_delay_value;

    reg prog_ucode_we;
    reg [CORE_ID_BITS-1:0] prog_ucode_core;
    reg [5:0] prog_ucode_addr;
    reg [31:0] prog_ucode_data;

    reg prog_param_we;
    reg [CORE_ID_BITS-1:0] prog_param_core;
    reg [NEURON_BITS-1:0] prog_param_neuron;
    reg [3:0] prog_param_id;
    reg signed [DATA_WIDTH-1:0] prog_param_value;

    reg ext_valid;
    reg [CORE_ID_BITS-1:0] ext_core;
    reg [NEURON_BITS-1:0] ext_neuron_id;
    reg signed [DATA_WIDTH-1:0] ext_current;

    reg probe_read;
    reg [CORE_ID_BITS-1:0] probe_core;
    reg [NEURON_BITS-1:0] probe_neuron;
    reg [3:0] probe_state_id;
    reg [POOL_ADDR_BITS-1:0] probe_pool_addr;
    wire signed [DATA_WIDTH-1:0] probe_data;
    wire probe_valid;

    wire timestep_done;
    wire [NUM_CORES-1:0] spike_valid_bus;
    wire [NUM_CORES*NEURON_BITS-1:0] spike_id_bus;
    wire [5:0] mesh_state_out;
    wire [31:0] total_spikes, timestep_count;
    wire [NUM_CORES-1:0] core_idle_bus;

    // Internal (mesh ↔ chip_link)
    wire        cl_tx_push, cl_tx_full;
    wire [CORE_ID_BITS-1:0] cl_tx_core;
    wire [NEURON_BITS-1:0]  cl_tx_neuron;
    wire [7:0]              cl_tx_payload;
    wire [CORE_ID_BITS-1:0] cl_rx_core;
    wire [NEURON_BITS-1:0]  cl_rx_neuron;
    wire signed [DATA_WIDTH-1:0] cl_rx_current;
    wire        cl_rx_pop, cl_rx_empty;

    wire [7:0] link_tx_data;
    wire       link_tx_valid;
    wire       link_rx_ready;

    // Testbench-driven external signals
    reg        tb_tx_ready;
    reg  [7:0] tb_rx_data;
    reg        tb_rx_valid;

    reg loopback_en;

    // Muxed link signals
    wire       eff_tx_ready = loopback_en ? link_rx_ready : tb_tx_ready;
    wire [7:0] eff_rx_data  = loopback_en ? link_tx_data  : tb_rx_data;
    wire       eff_rx_valid = loopback_en ? link_tx_valid  : tb_rx_valid;

    neuromorphic_mesh #(
        .NUM_CORES(NUM_CORES), .CORE_ID_BITS(CORE_ID_BITS),
        .NUM_NEURONS(NUM_NEURONS), .NEURON_BITS(NEURON_BITS),
        .DATA_WIDTH(DATA_WIDTH), .POOL_DEPTH(POOL_DEPTH),
        .POOL_ADDR_BITS(POOL_ADDR_BITS), .COUNT_BITS(COUNT_BITS),
        .THRESHOLD(THRESHOLD), .LEAK_RATE(LEAK_RATE),
        .ROUTE_FANOUT(ROUTE_FANOUT), .ROUTE_SLOT_BITS(ROUTE_SLOT_BITS),
        .GLOBAL_ROUTE_SLOTS(GLOBAL_ROUTE_SLOTS),
        .GLOBAL_ROUTE_SLOT_BITS(GLOBAL_ROUTE_SLOT_BITS),
        .CHIP_LINK_EN(1)
    ) uut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .prog_pool_we(prog_pool_we), .prog_pool_core(prog_pool_core),
        .prog_pool_addr(prog_pool_addr), .prog_pool_src(prog_pool_src),
        .prog_pool_target(prog_pool_target), .prog_pool_weight(prog_pool_weight),
        .prog_pool_comp(prog_pool_comp),
        .prog_index_we(prog_index_we), .prog_index_core(prog_index_core),
        .prog_index_neuron(prog_index_neuron), .prog_index_base(prog_index_base),
        .prog_index_count(prog_index_count), .prog_index_format(prog_index_format),
        .prog_route_we(prog_route_we), .prog_route_src_core(prog_route_src_core),
        .prog_route_src_neuron(prog_route_src_neuron), .prog_route_slot(prog_route_slot),
        .prog_route_dest_core(prog_route_dest_core),
        .prog_route_dest_neuron(prog_route_dest_neuron),
        .prog_route_weight(prog_route_weight),
        .prog_global_route_we(prog_global_route_we),
        .prog_global_route_src_core(prog_global_route_src_core),
        .prog_global_route_src_neuron(prog_global_route_src_neuron),
        .prog_global_route_slot(prog_global_route_slot),
        .prog_global_route_dest_core(prog_global_route_dest_core),
        .prog_global_route_dest_neuron(prog_global_route_dest_neuron),
        .prog_global_route_weight(prog_global_route_weight),
        .learn_enable(learn_enable), .graded_enable(graded_enable),
        .dendritic_enable(dendritic_enable), .async_enable(async_enable),
        .threefactor_enable(threefactor_enable), .noise_enable(noise_enable),
        .skip_idle_enable(skip_idle_enable),
        .scale_u_enable(1'b0),
        .reward_value(reward_value),
        .prog_delay_we(prog_delay_we), .prog_delay_core(prog_delay_core),
        .prog_delay_addr(prog_delay_addr), .prog_delay_value(prog_delay_value),
        .prog_ucode_we(prog_ucode_we), .prog_ucode_core(prog_ucode_core),
        .prog_ucode_addr(prog_ucode_addr), .prog_ucode_data(prog_ucode_data),
        .prog_param_we(prog_param_we), .prog_param_core(prog_param_core),
        .prog_param_neuron(prog_param_neuron), .prog_param_id(prog_param_id),
        .prog_param_value(prog_param_value),
        .probe_read(probe_read), .probe_core(probe_core),
        .probe_neuron(probe_neuron), .probe_state_id(probe_state_id),
        .probe_pool_addr(probe_pool_addr),
        .probe_data(probe_data), .probe_valid(probe_valid),
        .ext_valid(ext_valid), .ext_core(ext_core),
        .ext_neuron_id(ext_neuron_id), .ext_current(ext_current),
        .timestep_done(timestep_done), .spike_valid_bus(spike_valid_bus),
        .spike_id_bus(spike_id_bus), .mesh_state_out(mesh_state_out),
        .total_spikes(total_spikes), .timestep_count(timestep_count),
        .core_idle_bus(core_idle_bus),
        // P21E: Chip link
        .link_tx_push(cl_tx_push), .link_tx_core(cl_tx_core),
        .link_tx_neuron(cl_tx_neuron), .link_tx_payload(cl_tx_payload),
        .link_tx_full(cl_tx_full),
        .link_rx_core(cl_rx_core), .link_rx_neuron(cl_rx_neuron),
        .link_rx_current(cl_rx_current),
        .link_rx_pop(cl_rx_pop), .link_rx_empty(cl_rx_empty)
    );

    chip_link #(
        .CORE_ID_BITS(CORE_ID_BITS),
        .NEURON_BITS(NEURON_BITS),
        .DATA_WIDTH(DATA_WIDTH),
        .TX_DEPTH(256),
        .RX_DEPTH(256)
    ) u_link (
        .clk(clk), .rst_n(rst_n),
        // Internal TX (from mesh)
        .tx_push(cl_tx_push), .tx_core(cl_tx_core),
        .tx_neuron(cl_tx_neuron), .tx_payload(cl_tx_payload),
        .tx_full(cl_tx_full),
        // Internal RX (to mesh)
        .rx_core(cl_rx_core), .rx_neuron(cl_rx_neuron),
        .rx_current(cl_rx_current),
        .rx_pop(cl_rx_pop), .rx_empty(cl_rx_empty),
        // External link (to testbench / loopback)
        .link_tx_data(link_tx_data), .link_tx_valid(link_tx_valid),
        .link_tx_ready(eff_tx_ready),
        .link_rx_data(eff_rx_data), .link_rx_valid(eff_rx_valid),
        .link_rx_ready(link_rx_ready)
    );

    task clear_prog;
        begin
            prog_pool_we <= 0; prog_index_we <= 0; prog_route_we <= 0;
            prog_global_route_we <= 0; prog_delay_we <= 0; prog_ucode_we <= 0;
            prog_param_we <= 0; ext_valid <= 0;
        end
    endtask

    task run_timestep;
        begin
            start <= 1; @(posedge clk); start <= 0;
            wait(timestep_done); @(posedge clk);
        end
    endtask

    task inject(input [CORE_ID_BITS-1:0] core, input [NEURON_BITS-1:0] nrn,
                input signed [DATA_WIDTH-1:0] current);
        begin
            ext_valid <= 1; ext_core <= core; ext_neuron_id <= nrn; ext_current <= current;
            @(posedge clk); ext_valid <= 0; @(posedge clk);
        end
    endtask

    task prog_global_route(input [CORE_ID_BITS-1:0] src_core,
                           input [NEURON_BITS-1:0] src_neuron,
                           input [GLOBAL_ROUTE_SLOT_BITS-1:0] slot,
                           input [CORE_ID_BITS-1:0] dest_core,
                           input [NEURON_BITS-1:0] dest_neuron,
                           input signed [DATA_WIDTH-1:0] wt);
        begin
            prog_global_route_we <= 1;
            prog_global_route_src_core <= src_core;
            prog_global_route_src_neuron <= src_neuron;
            prog_global_route_slot <= slot;
            prog_global_route_dest_core <= dest_core;
            prog_global_route_dest_neuron <= dest_neuron;
            prog_global_route_weight <= wt;
            @(posedge clk); prog_global_route_we <= 0; @(posedge clk);
        end
    endtask

    task do_probe(input [CORE_ID_BITS-1:0] core, input [NEURON_BITS-1:0] neuron,
                  input [3:0] sid, input [POOL_ADDR_BITS-1:0] paddr);
        begin
            probe_read <= 1; probe_core <= core; probe_neuron <= neuron;
            probe_state_id <= sid; probe_pool_addr <= paddr;
            @(posedge clk); probe_read <= 0;
            wait(probe_valid); @(posedge clk);
        end
    endtask

    // Send one byte on the external link RX (with valid handshake)
    task send_rx_byte(input [7:0] data);
        begin
            tb_rx_data <= data;
            tb_rx_valid <= 1;
            @(posedge clk);
            tb_rx_valid <= 0;
            @(posedge clk);
        end
    endtask

    integer pass_count, fail_count;
    integer i;
    reg signed [DATA_WIDTH-1:0] potential;

    // TX capture (concurrent)
    reg [7:0] captured_bytes [0:3];
    integer byte_idx;
    reg capture_en;

    // Concurrent TX byte capture — runs in parallel with initial block
    always @(posedge clk) begin
        if (capture_en && link_tx_valid && byte_idx < 4) begin
            captured_bytes[byte_idx] <= link_tx_data;
            byte_idx <= byte_idx + 1;
        end
    end

    initial begin
        clk = 0; rst_n = 0; start = 0;
        clear_prog;
        learn_enable = 0; graded_enable = 0; dendritic_enable = 0;
        async_enable = 0; threefactor_enable = 0; noise_enable = 0;
        skip_idle_enable = 0;
        reward_value = 0;
        probe_read = 0; probe_core = 0; probe_neuron = 0;
        probe_state_id = 0; probe_pool_addr = 0;
        tb_tx_ready = 1;
        tb_rx_data = 0;
        tb_rx_valid = 0;
        loopback_en = 0;
        capture_en = 0;
        byte_idx = 0;
        pass_count = 0; fail_count = 0;

        #20000 rst_n = 1;
        @(posedge clk); @(posedge clk);

        $display("\n========================================");
        $display("TEST 1: TX - local spike routes to off-chip output");
        $display("========================================");
        // Program global route: core 0, neuron 5, slot 0 → off-chip
        // dest_core=1, dest_neuron=20, weight=16'hFFFF (negative = off-chip flag)
        prog_global_route(2'd0, 10'd5, 2'd0, 2'd1, 10'd20, 16'shFFFF);

        // Inject above threshold to make core 0, neuron 5 spike
        inject(0, 5, 16'sd1500);

        // Enable concurrent TX byte capture BEFORE starting timestep
        byte_idx = 0;
        capture_en = 1;

        // Run timestep — TX bytes are sent during routing phase
        start <= 1; @(posedge clk); start <= 0;
        wait(timestep_done); @(posedge clk);

        // Wait extra cycles for TX serializer to finish
        repeat(50) @(posedge clk);
        capture_en = 0;

        $display("  Captured %0d TX bytes", byte_idx);
        for (i = 0; i < byte_idx; i = i + 1)
            $display("  TX byte %0d: 0x%02h", i, captured_bytes[i]);

        // Verify: 4 bytes, byte 0 has start marker (bit 7) and dest_core=1
        if (byte_idx == 4 && captured_bytes[0][7] == 1'b1 &&
            captured_bytes[0][1:0] == 2'd1) begin
            $display("TEST 1 PASSED (4 TX bytes, start marker + dest_core=1)");
            pass_count = pass_count + 1;
        end else begin
            $display("TEST 1 FAILED (byte_idx=%0d, byte0=0x%02h)", byte_idx, captured_bytes[0]);
            fail_count = fail_count + 1;
        end

        $display("\n========================================");
        $display("TEST 2: RX - external spike injection into local core");
        $display("========================================");
        // Send a spike packet to core 0, neuron 30, payload=200
        // neuron 30 = 10'b0000011110
        //   Byte 0: {1'b1, 5'b0, core=0} = 8'h80
        //   Byte 1: neuron[9:2] = 8'b00000111 = 7
        //   Byte 2: {neuron[1:0], payload[7:2]} = {2'b10, 6'b110010} = {2'b10, 50} = 8'hB2
        //   Byte 3: {payload[1:0], 6'd0} = {2'b00, 6'd0} = 8'h00
        // payload = 200 = 8'b11001000
        //   payload[7:2] = 6'b110010 = 50
        //   payload[1:0] = 2'b00
        send_rx_byte(8'h80);  // Byte 0: start + core 0
        send_rx_byte(8'd7);   // Byte 1: neuron[9:2] = 7
        send_rx_byte(8'hB2);  // Byte 2: {neuron[1:0]=10, payload[7:2]=110010}
        send_rx_byte(8'h00);  // Byte 3: {payload[1:0]=00, 6'd0}

        // Wait a few cycles for RX FIFO to be written
        repeat(5) @(posedge clk);

        // Run timestep — SM_LINK_RX_DRAIN will inject the RX spike
        run_timestep;

        // Read membrane potential of core 0, neuron 30
        // Should be ~200 - leak = 197
        do_probe(0, 30, 4'd0, 0);
        potential = $signed(probe_data);
        $display("  Core 0, neuron 30 potential = %0d", potential);

        if (potential > 100 && potential < 300) begin
            $display("TEST 2 PASSED (RX injection: potential = %0d)", potential);
            pass_count = pass_count + 1;
        end else begin
            $display("TEST 2 FAILED (potential = %0d, expected ~197)", potential);
            fail_count = fail_count + 1;
        end

        $display("\n========================================");
        $display("TEST 3: Loopback - TX→RX → spike arrives at destination");
        $display("========================================");
        // Enable loopback: chip_link TX output feeds directly to RX input
        loopback_en = 1;
        tb_tx_ready = 1;  // Not used in loopback mode

        // Program global route: core 1, neuron 10, slot 0 → off-chip
        // dest_core=2, dest_neuron=50, weight=16'hFF00 (negative = off-chip)
        prog_global_route(2'd1, 10'd10, 2'd0, 2'd2, 10'd50, 16'shFF00);

        // Inject above threshold into core 1, neuron 10
        inject(1, 10, 16'sd1500);

        // Run timestep: spike → TX → loopback → RX FIFO
        run_timestep;

        // Wait for TX serialization to complete and RX to deserialize
        repeat(20) @(posedge clk);

        // Run another timestep: SM_LINK_RX_DRAIN injects looped-back spike
        run_timestep;

        // Read membrane potential of core 2, neuron 50
        // The loopback injects the spike payload as unsigned current
        do_probe(2, 50, 4'd0, 0);
        potential = $signed(probe_data);
        $display("  Core 2, neuron 50 potential = %0d (loopback)", potential);

        if (potential > 0) begin
            $display("TEST 3 PASSED (loopback injection: potential = %0d)", potential);
            pass_count = pass_count + 1;
        end else begin
            $display("TEST 3 FAILED (potential = %0d, expected > 0)", potential);
            fail_count = fail_count + 1;
        end

        loopback_en = 0;

        $display("\n========================================");
        $display("TEST 4: FIFO back-pressure - TX stalls when link busy");
        $display("========================================");
        // Hold link_tx_ready = 0 (receiver not ready)
        tb_tx_ready = 0;

        // Program global route: core 0, neuron 40, slot 0 → off-chip
        prog_global_route(2'd0, 10'd40, 2'd0, 2'd3, 10'd60, 16'shFFFF);

        // Inject above threshold
        inject(0, 40, 16'sd1500);
        run_timestep;

        // TX should be stalled — link_tx_valid should eventually assert
        // but no data consumed (tx_ready=0)
        // Wait and check that chip_link TX FSM is holding
        repeat(10) @(posedge clk);
        $display("  link_tx_valid=%b, tb_tx_ready=%b (stalled)", link_tx_valid, tb_tx_ready);

        // Now release back-pressure and capture bytes
        byte_idx = 0;
        capture_en = 1;
        tb_tx_ready = 1;

        // Wait for all 4 bytes to be serialized
        repeat(50) @(posedge clk);
        capture_en = 0;

        $display("  After releasing: captured %0d bytes", byte_idx);
        if (byte_idx == 4 && captured_bytes[0][7] == 1'b1) begin
            $display("TEST 4 PASSED (back-pressure: %0d bytes after release)", byte_idx);
            pass_count = pass_count + 1;
        end else begin
            $display("TEST 4 FAILED (byte_idx=%0d)", byte_idx);
            fail_count = fail_count + 1;
        end

        $display("\n========================================");
        $display("P21E RESULTS: %0d/%0d passed", pass_count, pass_count + fail_count);
        $display("========================================");
        if (fail_count == 0)
            $display("All tests passed!");
        else
            $display("SOME TESTS FAILED");
        $finish;
    end

endmodule
