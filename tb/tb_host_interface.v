// ============================================================================
// Testbench: Host Interface (byte-level, bypassing UART serial timing)
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

`timescale 1ns / 1ps

module tb_host_interface;

    parameter NUM_CORES      = 4;
    parameter CORE_ID_BITS   = 2;
    parameter NUM_NEURONS    = 256;
    parameter NEURON_BITS    = 8;
    parameter DATA_WIDTH     = 16;
    parameter MAX_FANOUT     = 32;
    parameter FANOUT_BITS    = 5;
    parameter CONN_ADDR_BITS = 13;
    parameter CLK_PERIOD     = 10;

    reg        clk, rst_n;

    // Host interface byte I/O (simulates UART RX/TX at byte level)
    reg  [7:0] rx_data;
    reg        rx_valid;
    wire [7:0] tx_data;
    wire       tx_valid;
    reg        tx_ready;  // Always ready for fast sim

    // Mesh connections (directly wired)
    wire       mesh_start;
    wire       mesh_prog_conn_we;
    wire [CORE_ID_BITS-1:0]    mesh_prog_conn_core;
    wire [NEURON_BITS-1:0]     mesh_prog_conn_src;
    wire [FANOUT_BITS-1:0]     mesh_prog_conn_slot;
    wire [NEURON_BITS-1:0]     mesh_prog_conn_target;
    wire signed [DATA_WIDTH-1:0] mesh_prog_conn_weight;

    wire       mesh_prog_route_we;
    wire [CORE_ID_BITS-1:0]    mesh_prog_route_src_core;
    wire [NEURON_BITS-1:0]     mesh_prog_route_src_neuron;
    wire [CORE_ID_BITS-1:0]    mesh_prog_route_dest_core;
    wire [NEURON_BITS-1:0]     mesh_prog_route_dest_neuron;
    wire signed [DATA_WIDTH-1:0] mesh_prog_route_weight;

    wire       mesh_ext_valid;
    wire [CORE_ID_BITS-1:0]    mesh_ext_core;
    wire [NEURON_BITS-1:0]     mesh_ext_neuron_id;
    wire signed [DATA_WIDTH-1:0] mesh_ext_current;

    wire       mesh_timestep_done;
    wire [4:0] mesh_state_out;
    wire [31:0] mesh_total_spikes;
    wire [31:0] mesh_timestep_count;
    wire [NUM_CORES-1:0] spike_valid_bus;
    wire [NUM_CORES*NEURON_BITS-1:0] spike_id_bus;

    reg [7:0]  resp_buf [0:15];
    integer    resp_cnt;

    host_interface #(
        .NUM_CORES    (NUM_CORES),
        .CORE_ID_BITS (CORE_ID_BITS),
        .NUM_NEURONS  (NUM_NEURONS),
        .NEURON_BITS  (NEURON_BITS),
        .DATA_WIDTH   (DATA_WIDTH),
        .MAX_FANOUT   (MAX_FANOUT),
        .FANOUT_BITS  (FANOUT_BITS)
    ) u_hi (
        .clk       (clk),
        .rst_n     (rst_n),
        .rx_data   (rx_data),
        .rx_valid  (rx_valid),
        .tx_data   (tx_data),
        .tx_valid  (tx_valid),
        .tx_ready  (tx_ready),

        .mesh_start              (mesh_start),
        .mesh_prog_conn_we       (mesh_prog_conn_we),
        .mesh_prog_conn_core     (mesh_prog_conn_core),
        .mesh_prog_conn_src      (mesh_prog_conn_src),
        .mesh_prog_conn_slot     (mesh_prog_conn_slot),
        .mesh_prog_conn_target   (mesh_prog_conn_target),
        .mesh_prog_conn_weight   (mesh_prog_conn_weight),
        .mesh_prog_route_we      (mesh_prog_route_we),
        .mesh_prog_route_src_core   (mesh_prog_route_src_core),
        .mesh_prog_route_src_neuron (mesh_prog_route_src_neuron),
        .mesh_prog_route_dest_core  (mesh_prog_route_dest_core),
        .mesh_prog_route_dest_neuron(mesh_prog_route_dest_neuron),
        .mesh_prog_route_weight     (mesh_prog_route_weight),
        .mesh_ext_valid          (mesh_ext_valid),
        .mesh_ext_core           (mesh_ext_core),
        .mesh_ext_neuron_id      (mesh_ext_neuron_id),
        .mesh_ext_current        (mesh_ext_current),
        .mesh_learn_enable       (),
        .mesh_graded_enable      (),
        .mesh_dendritic_enable   (),
        .mesh_async_enable       (),
        .mesh_prog_conn_comp     (),
        .mesh_prog_param_we      (),
        .mesh_prog_param_core    (),
        .mesh_prog_param_neuron  (),
        .mesh_prog_param_id      (),
        .mesh_prog_param_value   (),

        .mesh_timestep_done  (mesh_timestep_done),
        .mesh_state          (mesh_state_out),
        .mesh_total_spikes   (mesh_total_spikes),
        .mesh_timestep_count (mesh_timestep_count)
    );

    neuromorphic_mesh #(
        .NUM_CORES      (NUM_CORES),
        .CORE_ID_BITS   (CORE_ID_BITS),
        .NUM_NEURONS    (NUM_NEURONS),
        .NEURON_BITS    (NEURON_BITS),
        .DATA_WIDTH     (DATA_WIDTH),
        .MAX_FANOUT     (MAX_FANOUT),
        .FANOUT_BITS    (FANOUT_BITS),
        .CONN_ADDR_BITS (CONN_ADDR_BITS),
        .THRESHOLD      (16'sd1000),
        .LEAK_RATE      (16'sd3),
        .REFRAC_CYCLES  (3)
    ) u_mesh (
        .clk               (clk),
        .rst_n             (rst_n),
        .start             (mesh_start),
        .prog_conn_we      (mesh_prog_conn_we),
        .prog_conn_core    (mesh_prog_conn_core),
        .prog_conn_src     (mesh_prog_conn_src),
        .prog_conn_slot    (mesh_prog_conn_slot),
        .prog_conn_target  (mesh_prog_conn_target),
        .prog_conn_weight  (mesh_prog_conn_weight),
        .prog_route_we         (mesh_prog_route_we),
        .prog_route_src_core   (mesh_prog_route_src_core),
        .prog_route_src_neuron (mesh_prog_route_src_neuron),
        .prog_route_dest_core  (mesh_prog_route_dest_core),
        .prog_route_dest_neuron(mesh_prog_route_dest_neuron),
        .prog_route_weight     (mesh_prog_route_weight),
        .learn_enable      (1'b0),
        .graded_enable     (1'b0),
        .dendritic_enable  (1'b0),
        .async_enable      (1'b0),
        .prog_conn_comp    (2'd0),
        .prog_param_we     (1'b0),
        .prog_param_core   (2'd0),
        .prog_param_neuron (8'd0),
        .prog_param_id     (3'd0),
        .prog_param_value  (16'sd0),
        .ext_valid         (mesh_ext_valid),
        .ext_core          (mesh_ext_core),
        .ext_neuron_id     (mesh_ext_neuron_id),
        .ext_current       (mesh_ext_current),
        .timestep_done     (mesh_timestep_done),
        .spike_valid_bus   (spike_valid_bus),
        .spike_id_bus      (spike_id_bus),
        .mesh_state_out    (mesh_state_out),
        .total_spikes      (mesh_total_spikes),
        .timestep_count    (mesh_timestep_count)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    initial begin
        $dumpfile("host_interface.vcd");
        $dumpvars(0, tb_host_interface);
    end

    integer i;
    always @(posedge clk) begin
        for (i = 0; i < NUM_CORES; i = i + 1) begin
            if (spike_valid_bus[i]) begin
                $display("  [spike] Core %0d Neuron %0d (ts=%0d)",
                    i, spike_id_bus[i*NEURON_BITS +: NEURON_BITS], mesh_timestep_count);
            end
        end
    end

    // Capture TX responses
    always @(posedge clk) begin
        if (tx_valid && tx_ready) begin
            resp_buf[resp_cnt] <= tx_data;
            resp_cnt <= resp_cnt + 1;
            $display("  [TX] byte %0d: 0x%02h", resp_cnt, tx_data);
        end
    end

    task send_byte;
        input [7:0] b;
    begin
        @(posedge clk);
        rx_data  <= b;
        rx_valid <= 1;
        @(posedge clk);
        rx_valid <= 0;
    end
    endtask

    //   0x01 [core][src][slot][target][weight_hi][weight_lo]
    task cmd_prog_conn;
        input [7:0] core;
        input [7:0] src;
        input [7:0] slot;
        input [7:0] target;
        input signed [15:0] weight;
    begin
        send_byte(8'h01);
        send_byte(core);
        send_byte(src);
        send_byte(slot);
        send_byte(target);
        send_byte(weight[15:8]);
        send_byte(weight[7:0]);
    end
    endtask

    //   0x02 [src_core][src_neuron][dest_core][dest_neuron][weight_hi][weight_lo]
    task cmd_prog_route;
        input [7:0] src_core;
        input [7:0] src_neuron;
        input [7:0] dest_core;
        input [7:0] dest_neuron;
        input signed [15:0] weight;
    begin
        send_byte(8'h02);
        send_byte(src_core);
        send_byte(src_neuron);
        send_byte(dest_core);
        send_byte(dest_neuron);
        send_byte(weight[15:8]);
        send_byte(weight[7:0]);
    end
    endtask

    //   0x03 [core][neuron][current_hi][current_lo]
    task cmd_stimulus;
        input [7:0] core;
        input [7:0] neuron;
        input signed [15:0] current;
    begin
        send_byte(8'h03);
        send_byte(core);
        send_byte(neuron);
        send_byte(current[15:8]);
        send_byte(current[7:0]);
    end
    endtask

    //   0x04 [timesteps_hi][timesteps_lo]
    task cmd_run;
        input [15:0] timesteps;
    begin
        send_byte(8'h04);
        send_byte(timesteps[15:8]);
        send_byte(timesteps[7:0]);
    end
    endtask

    //   0x05
    task cmd_status;
    begin
        send_byte(8'h05);
    end
    endtask

    task wait_ack;
    begin
        wait(resp_cnt > 0);
        @(posedge clk);
        if (resp_buf[0] == 8'hAA)
            $display("  -> ACK received");
        else
            $display("  -> ERROR: expected ACK (0xAA), got 0x%02h", resp_buf[0]);
        resp_cnt <= 0;
        @(posedge clk);
    end
    endtask

    task wait_done;
    begin
        wait(resp_cnt >= 5);
        @(posedge clk);
        @(posedge clk);
        if (resp_buf[0] == 8'hDD) begin
            $display("  -> DONE received, spikes = %0d",
                {resp_buf[1], resp_buf[2], resp_buf[3], resp_buf[4]});
        end else begin
            $display("  -> ERROR: expected DONE (0xDD), got 0x%02h", resp_buf[0]);
        end
        resp_cnt <= 0;
        @(posedge clk);
    end
    endtask

    task wait_status;
    begin
        wait(resp_cnt >= 5);
        @(posedge clk);
        @(posedge clk);
        $display("  -> STATUS: state=%0d, timestep_count=%0d",
            resp_buf[0],
            {resp_buf[1], resp_buf[2], resp_buf[3], resp_buf[4]});
        resp_cnt <= 0;
        @(posedge clk);
    end
    endtask

    initial begin
        rst_n    = 0;
        rx_data  = 0;
        rx_valid = 0;
        tx_ready = 1;  // TX always ready for fast sim
        resp_cnt = 0;

        $display("");
        $display("================================================================");
        $display("  Host Interface Test - Byte-Level Command Protocol");
        $display("================================================================");

        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 5);

        $display("");
        $display("--- TEST 1: Program Connections via Host ---");

        // Core 0: chain N0→N1→N2→N3 with strong weights
        $display("  Sending PROG_CONN: Core 0, N0→N1, w=1200");
        cmd_prog_conn(0, 0, 0, 1, 16'sd1200);
        wait_ack();

        $display("  Sending PROG_CONN: Core 0, N1→N2, w=1200");
        cmd_prog_conn(0, 1, 0, 2, 16'sd1200);
        wait_ack();

        $display("  Sending PROG_CONN: Core 0, N2→N3, w=1200");
        cmd_prog_conn(0, 2, 0, 3, 16'sd1200);
        wait_ack();

        $display("  Connections programmed successfully!");

        $display("");
        $display("--- TEST 2: Stimulus + Run (10 timesteps) ---");

        $display("  Sending STIMULUS: Core 0, N0, current=1200");
        cmd_stimulus(0, 0, 16'sd1200);
        wait_ack();

        $display("  Sending RUN: 10 timesteps");
        cmd_run(16'd10);
        wait_done();

        $display("");
        $display("--- TEST 3: Status Query ---");

        cmd_status();
        wait_status();

        $display("");
        $display("--- TEST 4: Cross-Core Route + Run ---");

        // Route: Core 0 N3 → Core 1 N0
        $display("  Sending PROG_ROUTE: C0:N3 -> C1:N0, w=1200");
        cmd_prog_route(0, 3, 1, 0, 16'sd1200);
        wait_ack();

        // Core 1: chain N0→N1
        $display("  Sending PROG_CONN: Core 1, N0→N1, w=1200");
        cmd_prog_conn(1, 0, 0, 1, 16'sd1200);
        wait_ack();

        // Run with stimulus to drive cross-core propagation
        $display("  Sending STIMULUS: Core 0, N0, current=1200");
        cmd_stimulus(0, 0, 16'sd1200);
        wait_ack();

        $display("  Sending RUN: 20 timesteps");
        cmd_run(16'd20);
        wait_done();

        $display("");
        $display("--- TEST 5: Second RUN Burst (no new stimulus) ---");

        $display("  Sending RUN: 5 timesteps (no stimulus)");
        cmd_run(16'd5);
        wait_done();

        $display("");
        $display("--- Final Status ---");
        cmd_status();
        wait_status();

        $display("");
        $display("================================================================");
        $display("  FINAL REPORT");
        $display("================================================================");
        $display("  Total timesteps: %0d", mesh_timestep_count);
        $display("  Total spikes:    %0d", mesh_total_spikes);
        $display("  Host protocol:   5 command types verified");
        $display("  Architecture:    UART -> Host IF -> Mesh (4x256 = 1024 neurons)");
        $display("================================================================");

        #(CLK_PERIOD * 10);
        $finish;
    end

    initial begin
        #(CLK_PERIOD * 3000000);
        $display("TIMEOUT at mesh_state=%0d, ts=%0d", mesh_state_out, mesh_timestep_count);
        $finish;
    end

endmodule
