`timescale 1ns/1ps

module tb_p22h_power;

    parameter NUM_CORES      = 4;
    parameter CORE_ID_BITS   = 2;
    parameter NUM_NEURONS    = 1024;
    parameter NEURON_BITS    = 10;
    parameter DATA_WIDTH     = 16;
    parameter POOL_DEPTH     = 32768;
    parameter POOL_ADDR_BITS = 15;
    parameter COUNT_BITS     = 10;

    reg clk, rst_n;
    reg start;

    reg                         prog_pool_we;
    reg [CORE_ID_BITS-1:0]     prog_pool_core;
    reg [POOL_ADDR_BITS-1:0]   prog_pool_addr;
    reg [NEURON_BITS-1:0]      prog_pool_src, prog_pool_target;
    reg signed [DATA_WIDTH-1:0] prog_pool_weight;
    reg [1:0]                   prog_pool_comp;

    reg                         prog_index_we;
    reg [CORE_ID_BITS-1:0]     prog_index_core;
    reg [NEURON_BITS-1:0]      prog_index_neuron;
    reg [POOL_ADDR_BITS-1:0]   prog_index_base;
    reg [COUNT_BITS-1:0]       prog_index_count;
    reg [1:0]                   prog_index_format;

    reg                         prog_route_we;
    reg [CORE_ID_BITS-1:0]     prog_route_src_core;
    reg [NEURON_BITS-1:0]      prog_route_src_neuron;
    reg [2:0]                   prog_route_slot;
    reg [CORE_ID_BITS-1:0]     prog_route_dest_core;
    reg [NEURON_BITS-1:0]      prog_route_dest_neuron;
    reg signed [DATA_WIDTH-1:0] prog_route_weight;

    reg        learn_enable, graded_enable, dendritic_enable, async_enable;
    reg        threefactor_enable, noise_enable, skip_idle_enable;
    reg signed [DATA_WIDTH-1:0] reward_value;

    reg                         prog_delay_we;
    reg [CORE_ID_BITS-1:0]     prog_delay_core;
    reg [POOL_ADDR_BITS-1:0]   prog_delay_addr;
    reg [5:0]                   prog_delay_value;

    reg                         prog_ucode_we;
    reg [CORE_ID_BITS-1:0]     prog_ucode_core;
    reg [6:0]                   prog_ucode_addr;
    reg [31:0]                  prog_ucode_data;

    reg                         prog_param_we;
    reg [CORE_ID_BITS-1:0]     prog_param_core;
    reg [NEURON_BITS-1:0]      prog_param_neuron;
    reg [4:0]                   prog_param_id;
    reg signed [DATA_WIDTH-1:0] prog_param_value;

    reg        ext_valid;
    reg [CORE_ID_BITS-1:0] ext_core;
    reg [NEURON_BITS-1:0]  ext_neuron_id;
    reg signed [DATA_WIDTH-1:0] ext_current;

    reg        probe_read;
    reg [CORE_ID_BITS-1:0] probe_core;
    reg [NEURON_BITS-1:0]  probe_neuron;
    reg [4:0]              probe_state_id;
    reg [POOL_ADDR_BITS-1:0] probe_pool_addr;
    wire signed [DATA_WIDTH-1:0] probe_data;
    wire       probe_valid;

    reg [7:0] dvfs_stall;

    reg                        prog_global_route_we;
    reg [CORE_ID_BITS-1:0]    prog_global_route_src_core;
    reg [NEURON_BITS-1:0]     prog_global_route_src_neuron;
    reg [1:0]                  prog_global_route_slot;
    reg [CORE_ID_BITS-1:0]    prog_global_route_dest_core;
    reg [NEURON_BITS-1:0]     prog_global_route_dest_neuron;
    reg signed [DATA_WIDTH-1:0] prog_global_route_weight;

    wire timestep_done;
    wire [NUM_CORES-1:0] spike_valid_bus;
    wire [NUM_CORES*NEURON_BITS-1:0] spike_id_bus;
    wire [5:0] mesh_state_out;
    wire [31:0] total_spikes, timestep_count;
    wire [NUM_CORES-1:0] core_idle_bus;

    neuromorphic_mesh #(
        .NUM_CORES      (NUM_CORES),
        .CORE_ID_BITS   (CORE_ID_BITS),
        .NUM_NEURONS    (NUM_NEURONS),
        .NEURON_BITS    (NEURON_BITS),
        .DATA_WIDTH     (DATA_WIDTH),
        .POOL_DEPTH     (POOL_DEPTH),
        .POOL_ADDR_BITS (POOL_ADDR_BITS),
        .COUNT_BITS     (COUNT_BITS),
        .THRESHOLD      (16'sd1000),
        .LEAK_RATE      (16'sd3),
        .REFRAC_CYCLES  (3)
    ) DUT (
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
        .prog_index_format (prog_index_format),
        .prog_route_we         (prog_route_we),
        .prog_route_src_core   (prog_route_src_core),
        .prog_route_src_neuron (prog_route_src_neuron),
        .prog_route_slot       (prog_route_slot),
        .prog_route_dest_core  (prog_route_dest_core),
        .prog_route_dest_neuron(prog_route_dest_neuron),
        .prog_route_weight     (prog_route_weight),
        .prog_global_route_we          (prog_global_route_we),
        .prog_global_route_src_core    (prog_global_route_src_core),
        .prog_global_route_src_neuron  (prog_global_route_src_neuron),
        .prog_global_route_slot        (prog_global_route_slot),
        .prog_global_route_dest_core   (prog_global_route_dest_core),
        .prog_global_route_dest_neuron (prog_global_route_dest_neuron),
        .prog_global_route_weight      (prog_global_route_weight),
        .learn_enable      (learn_enable),
        .graded_enable     (graded_enable),
        .dendritic_enable  (dendritic_enable),
        .async_enable      (async_enable),
        .threefactor_enable(threefactor_enable),
        .noise_enable      (noise_enable),
        .skip_idle_enable  (skip_idle_enable),
        .scale_u_enable    (1'b0),
        .reward_value      (reward_value),
        .prog_delay_we     (prog_delay_we),
        .prog_delay_core   (prog_delay_core),
        .prog_delay_addr   (prog_delay_addr),
        .prog_delay_value  (prog_delay_value),
        .prog_ucode_we     (prog_ucode_we),
        .prog_ucode_core   (prog_ucode_core),
        .prog_ucode_addr   (prog_ucode_addr),
        .prog_ucode_data   (prog_ucode_data),
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
        .dvfs_stall        (dvfs_stall),
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

    always #5 clk = ~clk;

    integer passed, failed;

    task set_param(input [CORE_ID_BITS-1:0] core, input [NEURON_BITS-1:0] nrn,
                   input [4:0] pid, input signed [DATA_WIDTH-1:0] val);
    begin
        @(posedge clk);
        prog_param_we     <= 1;
        prog_param_core   <= core;
        prog_param_neuron <= nrn;
        prog_param_id     <= pid;
        prog_param_value  <= val;
        @(posedge clk);
        prog_param_we <= 0;
        @(posedge clk);
    end
    endtask

    task add_pool(input [CORE_ID_BITS-1:0] core, input [POOL_ADDR_BITS-1:0] addr,
                  input [NEURON_BITS-1:0] src, input [NEURON_BITS-1:0] tgt,
                  input signed [DATA_WIDTH-1:0] wt);
    begin
        @(posedge clk);
        prog_pool_we     <= 1;
        prog_pool_core   <= core;
        prog_pool_addr   <= addr;
        prog_pool_src    <= src;
        prog_pool_target <= tgt;
        prog_pool_weight <= wt;
        prog_pool_comp   <= 2'd0;
        @(posedge clk);
        prog_pool_we <= 0;
        @(posedge clk);
    end
    endtask

    task add_index(input [CORE_ID_BITS-1:0] core, input [NEURON_BITS-1:0] nrn,
                   input [POOL_ADDR_BITS-1:0] base, input [COUNT_BITS-1:0] cnt);
    begin
        @(posedge clk);
        prog_index_we     <= 1;
        prog_index_core   <= core;
        prog_index_neuron <= nrn;
        prog_index_base   <= base;
        prog_index_count  <= cnt;
        prog_index_format <= 2'd0;
        @(posedge clk);
        prog_index_we <= 0;
        @(posedge clk);
    end
    endtask

    task inject_stim(input [CORE_ID_BITS-1:0] core, input [NEURON_BITS-1:0] nrn,
                     input signed [DATA_WIDTH-1:0] cur);
    begin
        @(posedge clk);
        ext_valid     <= 1;
        ext_core      <= core;
        ext_neuron_id <= nrn;
        ext_current   <= cur;
        @(posedge clk);
        ext_valid <= 0;
    end
    endtask

    task run_one_ts;
    begin
        @(posedge clk);
        start <= 1;
        @(posedge clk);
        start <= 0;
        wait(timestep_done);
        @(posedge clk);
    end
    endtask

    task probe_read_val(input [CORE_ID_BITS-1:0] core, input [NEURON_BITS-1:0] nrn,
                        input [4:0] sid, output reg signed [DATA_WIDTH-1:0] val);
    begin
        @(posedge clk);
        probe_read     <= 1;
        probe_core     <= core;
        probe_neuron   <= nrn;
        probe_state_id <= sid;
        @(posedge clk);
        probe_read <= 0;
        wait(probe_valid);
        val = probe_data;
        @(posedge clk);
    end
    endtask

    reg signed [DATA_WIDTH-1:0] pval;
    integer t1_start, t1_end, t2_start, t2_end;
    integer cycles_fast, cycles_slow;

    initial begin
        clk = 0; rst_n = 0;
        start = 0;
        prog_pool_we = 0; prog_index_we = 0; prog_route_we = 0;
        prog_delay_we = 0; prog_ucode_we = 0; prog_param_we = 0;
        prog_global_route_we = 0;
        ext_valid = 0; probe_read = 0;
        learn_enable = 0; graded_enable = 0; dendritic_enable = 0;
        async_enable = 0; threefactor_enable = 0; noise_enable = 0;
        skip_idle_enable = 0; reward_value = 0;
        dvfs_stall = 0;
        passed = 0; failed = 0;

        repeat (5) @(posedge clk);
        rst_n = 1;
        repeat (3) @(posedge clk);

        $display("\nTest 1: Performance counters");

        set_param(0, 0, 5'd0, 16'sd500);

        add_pool(0, 0, 1, 0, 16'sd600);
        add_index(0, 1, 0, 1);

        inject_stim(0, 1, 16'sd1100);

        run_one_ts;

        run_one_ts;

        probe_read_val(0, 0, 5'd14, pval);
        $display("  perf_spike_count[15:0] = %0d", pval);

        begin
            reg signed [DATA_WIDTH-1:0] syn_ops;
            probe_read_val(0, 0, 5'd18, syn_ops);
            $display("  perf_synaptic_ops[15:0] = %0d", syn_ops);
            if (pval >= 2 && syn_ops >= 1) begin
                $display("  PASSED: spike_count=%0d, synaptic_ops=%0d", pval, syn_ops);
                passed = passed + 1;
            end else begin
                $display("  FAILED: spike_count=%0d (exp>=2), synaptic_ops=%0d (exp>=1)", pval, syn_ops);
                failed = failed + 1;
            end
        end

        $display("\nTest 2: Trace FIFO");

        rst_n = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        repeat (3) @(posedge clk);

        set_param(0, 0, 5'd27, 16'sd1);

        set_param(0, 5, 5'd0, 16'sd200);

        inject_stim(0, 5, 16'sd300);
        run_one_ts;

        repeat (5) begin
            run_one_ts;
        end
        inject_stim(0, 5, 16'sd300);
        run_one_ts;

        probe_read_val(0, 0, 5'd24, pval);
        $display("  trace FIFO count = %0d", pval);

        if (pval >= 1) begin
            begin
                reg signed [DATA_WIDTH-1:0] trace_lo, trace_hi;
                probe_read_val(0, 0, 5'd22, trace_lo);
                $display("  trace entry lo (neuron) = %0d", trace_lo);
                probe_read_val(0, 0, 5'd23, trace_hi);
                $display("  trace entry hi (timestamp) = %0d", trace_hi);
                if (trace_lo[9:0] == 10'd5 && trace_hi >= 0) begin
                    $display("  PASSED: trace recorded neuron 5, timestamp=%0d", trace_hi);
                    passed = passed + 1;
                end else begin
                    $display("  FAILED: trace neuron=%0d (exp 5), ts=%0d", trace_lo[9:0], trace_hi);
                    failed = failed + 1;
                end
            end
        end else begin
            $display("  FAILED: trace FIFO empty (count=%0d)", pval);
            failed = failed + 1;
        end

        $display("\nTest 3: DVFS stall");

        rst_n = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        repeat (3) @(posedge clk);
        dvfs_stall = 0;

        t1_start = $time;
        run_one_ts;
        t1_end = $time;
        cycles_fast = (t1_end - t1_start) / 10;

        dvfs_stall = 8'd100;

        t2_start = $time;
        run_one_ts;
        t2_end = $time;
        cycles_slow = (t2_end - t2_start) / 10;

        $display("  fast cycles = %0d, slow cycles = %0d", cycles_fast, cycles_slow);
        if (cycles_slow > cycles_fast + 80) begin
            $display("  PASSED: DVFS stall added %0d extra cycles", cycles_slow - cycles_fast);
            passed = passed + 1;
        end else begin
            $display("  FAILED: insufficient DVFS stall effect (delta=%0d)", cycles_slow - cycles_fast);
            failed = failed + 1;
        end

        dvfs_stall = 0;

        $display("\nTest 4: Power estimate");

        rst_n = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        repeat (3) @(posedge clk);

        probe_read_val(0, 0, 5'd20, pval);
        $display("  idle power estimate (lo) = %0d", pval);

        set_param(0, 10, 5'd0, 16'sd100);
        inject_stim(0, 10, 16'sd200);
        run_one_ts;
        run_one_ts;

        begin
            reg signed [DATA_WIDTH-1:0] pwr, act;
            probe_read_val(0, 0, 5'd20, pwr);
            $display("  active power estimate (lo) = %0d", pwr);
            probe_read_val(0, 0, 5'd16, act);
            $display("  active_cycles (lo) = %0d", act);
            if (pwr > 0 && act > 0) begin
                $display("  PASSED: power=%0d, active_cycles=%0d (both > 0)", pwr, act);
                passed = passed + 1;
            end else begin
                $display("  FAILED: power=%0d, active_cycles=%0d", pwr, act);
                failed = failed + 1;
            end
        end

        $display("P22H RESULTS: %0d/%0d passed", passed, passed+failed);
        if (failed == 0)
            $display("All tests passed!");

        $finish;
    end

    initial begin
        #5_000_000;
        $display("TIMEOUT!");
        $finish;
    end

endmodule
