`timescale 1ns/1ps

module tb_p13c;

    parameter NUM_CORES      = 4;
    parameter CORE_ID_BITS   = 2;
    parameter NUM_NEURONS    = 1024;
    parameter NEURON_BITS    = 10;
    parameter DATA_WIDTH     = 16;
    parameter POOL_DEPTH     = 1024;
    parameter POOL_ADDR_BITS = 10;
    parameter COUNT_BITS     = 10;
    parameter REV_FANIN      = 32;
    parameter REV_SLOT_BITS  = 5;
    parameter ROUTE_FANOUT   = 8;
    parameter ROUTE_SLOT_BITS = 3;
    parameter CLK_PERIOD     = 10;

    reg clk, rst_n;
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

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
    reg  signed [DATA_WIDTH-1:0] reward_value;

    reg                         prog_param_we;
    reg  [CORE_ID_BITS-1:0]    prog_param_core;
    reg  [NEURON_BITS-1:0]     prog_param_neuron;
    reg  [2:0]                  prog_param_id;
    reg  signed [DATA_WIDTH-1:0] prog_param_value;

    reg                         ext_valid;
    reg  [CORE_ID_BITS-1:0]    ext_core;
    reg  [NEURON_BITS-1:0]     ext_neuron_id;
    reg  signed [DATA_WIDTH-1:0] ext_current;

    wire                        timestep_done;
    wire [NUM_CORES-1:0]        spike_valid_bus;
    wire [NUM_CORES*NEURON_BITS-1:0] spike_id_bus;
    wire [4:0]                  mesh_state_out;
    wire [31:0]                 total_spikes;
    wire [31:0]                 timestep_count;

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
    ) dut (
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
        .reward_value      (reward_value),
        .prog_delay_we     (1'b0),
        .prog_delay_core   ({CORE_ID_BITS{1'b0}}),
        .prog_delay_addr   ({POOL_ADDR_BITS{1'b0}}),
        .prog_delay_value  (6'd0),
        .prog_ucode_we     (1'b0),
        .prog_ucode_core   ({CORE_ID_BITS{1'b0}}),
        .prog_ucode_addr   (6'd0),
        .prog_ucode_data   (32'd0),
        .prog_param_we     (prog_param_we),
        .prog_param_core   (prog_param_core),
        .prog_param_neuron (prog_param_neuron),
        .prog_param_id     (prog_param_id),
        .prog_param_value  (prog_param_value),
        .ext_valid         (ext_valid),
        .ext_core          (ext_core),
        .ext_neuron_id     (ext_neuron_id),
        .ext_current       (ext_current),
        .timestep_done     (timestep_done),
        .spike_valid_bus   (spike_valid_bus),
        .spike_id_bus      (spike_id_bus),
        .mesh_state_out    (mesh_state_out),
        .total_spikes      (total_spikes),
        .timestep_count    (timestep_count)
    );

    always @(posedge clk) begin : spike_monitor
        integer c;
        for (c = 0; c < NUM_CORES; c = c + 1) begin
            if (spike_valid_bus[c]) begin
                $display("  [t=%0d] Core %0d Neuron %0d spiked",
                    timestep_count, c, spike_id_bus[c*NEURON_BITS +: NEURON_BITS]);
            end
        end
    end

    initial begin
        $dumpfile("tb_p13c.vcd");
        $dumpvars(0, tb_p13c);
    end

    task add_pool;
        input [CORE_ID_BITS-1:0]     core;
        input [POOL_ADDR_BITS-1:0]   addr;
        input [NEURON_BITS-1:0]      src;
        input [NEURON_BITS-1:0]      target;
        input signed [DATA_WIDTH-1:0] weight;
        input [1:0]                   comp;
    begin
        @(posedge clk);
        prog_pool_we     <= 1;
        prog_pool_core   <= core;
        prog_pool_addr   <= addr;
        prog_pool_src    <= src;
        prog_pool_target <= target;
        prog_pool_weight <= weight;
        prog_pool_comp   <= comp;
        @(posedge clk);
        prog_pool_we <= 0;
    end
    endtask

    task set_index;
        input [CORE_ID_BITS-1:0]     core;
        input [NEURON_BITS-1:0]      neuron;
        input [POOL_ADDR_BITS-1:0]   base;
        input [COUNT_BITS-1:0]       count;
    begin
        @(posedge clk);
        prog_index_we     <= 1;
        prog_index_core   <= core;
        prog_index_neuron <= neuron;
        prog_index_base   <= base;
        prog_index_count  <= count;
        @(posedge clk);
        prog_index_we <= 0;
    end
    endtask

    task run_timestep;
        input [CORE_ID_BITS-1:0]     core;
        input [NEURON_BITS-1:0]      neuron;
        input signed [DATA_WIDTH-1:0] current;
    begin
        @(posedge clk);
        ext_valid     <= 1;
        ext_core      <= core;
        ext_neuron_id <= neuron;
        ext_current   <= current;
        @(posedge clk);
        ext_valid <= 0;
        start     <= 1;
        @(posedge clk);
        start <= 0;
        wait (timestep_done);
        @(posedge clk);
    end
    endtask

    task run_empty;
    begin
        @(posedge clk);
        start <= 1;
        @(posedge clk);
        start <= 0;
        wait (timestep_done);
        @(posedge clk);
    end
    endtask

    integer pass_count;
    integer fail_count;
    reg [31:0] spikes_before;
    reg signed [DATA_WIDTH-1:0] wt_before, wt_after;
    reg signed [DATA_WIDTH-1:0] elig_val;
    integer ts;

    initial begin
        start = 0;
        prog_pool_we = 0; prog_pool_core = 0; prog_pool_addr = 0;
        prog_pool_src = 0; prog_pool_target = 0; prog_pool_weight = 0; prog_pool_comp = 0;
        prog_index_we = 0; prog_index_core = 0; prog_index_neuron = 0;
        prog_index_base = 0; prog_index_count = 0;
        prog_route_we = 0; prog_route_src_core = 0; prog_route_src_neuron = 0;
        prog_route_slot = 0;
        prog_route_dest_core = 0; prog_route_dest_neuron = 0; prog_route_weight = 0;
        learn_enable = 0; graded_enable = 0; dendritic_enable = 0; async_enable = 0;
        threefactor_enable = 0; reward_value = 0;
        prog_param_we = 0; prog_param_core = 0; prog_param_neuron = 0;
        prog_param_id = 0; prog_param_value = 0;
        ext_valid = 0; ext_core = 0; ext_neuron_id = 0; ext_current = 0;

        pass_count = 0;
        fail_count = 0;

        rst_n = 0;
        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 5);

        $display("test 1: Elig accumulation (no reward)");

        add_pool(0, 0, 500, 501, 16'sd1200, 0);
        set_index(0, 500, 0, 1);

        learn_enable       = 1;
        threefactor_enable = 1;
        reward_value       = 16'sd0;

        wt_before = dut.gen_core[0].core.pool_weight_mem.mem[0];
        $display("  Initial weight[0] = %0d", wt_before);

        for (ts = 0; ts < 10; ts = ts + 1)
            run_timestep(0, 500, 16'sd1200);

        wt_after = dut.gen_core[0].core.pool_weight_mem.mem[0];
        elig_val = dut.gen_core[0].core.elig_mem.mem[0];
        $display("  After 10 timesteps: weight[0] = %0d, elig[0] = %0d", wt_after, elig_val);

        if (wt_after == wt_before) begin
            $display("TEST 1 PASSED (weight unchanged without reward)");
            pass_count = pass_count + 1;
        end else begin
            $display("TEST 1 FAILED - weight changed from %0d to %0d", wt_before, wt_after);
            fail_count = fail_count + 1;
        end

        $display("test 2: Reward application");

        wt_before = dut.gen_core[0].core.pool_weight_mem.mem[0];
        elig_val  = dut.gen_core[0].core.elig_mem.mem[0];
        $display("  Before reward: weight[0] = %0d, elig[0] = %0d", wt_before, elig_val);

        reward_value = 16'sd500;

        for (ts = 0; ts < 5; ts = ts + 1)
            run_timestep(0, 500, 16'sd1200);

        wt_after = dut.gen_core[0].core.pool_weight_mem.mem[0];
        elig_val = dut.gen_core[0].core.elig_mem.mem[0];
        $display("  After reward: weight[0] = %0d, elig[0] = %0d", wt_after, elig_val);

        if (wt_after > wt_before) begin
            $display("TEST 2 PASSED (weight increased with positive reward)");
            pass_count = pass_count + 1;
        end else begin
            $display("TEST 2 FAILED - weight didn't increase: before=%0d after=%0d", wt_before, wt_after);
            fail_count = fail_count + 1;
        end

        $display("test 3: Eligibility decay");

        learn_enable = 0;
        reward_value = 16'sd0;

        elig_val = dut.gen_core[0].core.elig_mem.mem[0];
        $display("  Initial elig[0] = %0d", elig_val);

        for (ts = 0; ts < 20; ts = ts + 1)
            run_empty();

        wt_before = dut.gen_core[0].core.pool_weight_mem.mem[0];
        elig_val  = dut.gen_core[0].core.elig_mem.mem[0];
        $display("  After 20 decay steps: elig[0] = %0d, weight[0] = %0d", elig_val, wt_before);

        if (elig_val == 0 || elig_val < 16'sd5) begin
            $display("TEST 3 PASSED (elig decayed to near-zero)");
            pass_count = pass_count + 1;
        end else begin
            $display("TEST 3 FAILED - elig still %0d after decay", elig_val);
            fail_count = fail_count + 1;
        end

        $display("test 4: Delayed reward");

        learn_enable       = 1;
        threefactor_enable = 1;
        reward_value       = 16'sd0;

        add_pool(0, 50, 600, 601, 16'sd1200, 0);
        set_index(0, 600, 50, 1);

        for (ts = 0; ts < 10; ts = ts + 1)
            run_timestep(0, 600, 16'sd1200);

        elig_val = dut.gen_core[0].core.elig_mem.mem[50];
        $display("  After stimulation: elig[50] = %0d", elig_val);

        learn_enable = 0;
        for (ts = 0; ts < 5; ts = ts + 1)
            run_empty();

        elig_val = dut.gen_core[0].core.elig_mem.mem[50];
        $display("  After 5 decay steps: elig[50] = %0d", elig_val);

        wt_before = dut.gen_core[0].core.pool_weight_mem.mem[50];
        reward_value = 16'sd500;

        for (ts = 0; ts < 3; ts = ts + 1)
            run_empty();

        wt_after = dut.gen_core[0].core.pool_weight_mem.mem[50];
        $display("  Delayed reward: weight before=%0d, after=%0d", wt_before, wt_after);

        if (wt_after > wt_before) begin
            $display("TEST 4 PASSED (delayed reward changed weight)");
            pass_count = pass_count + 1;
        end else begin
            $display("TEST 4 FAILED - weight unchanged: before=%0d after=%0d", wt_before, wt_after);
            fail_count = fail_count + 1;
        end

        $display("P13c RESULTS: %0d/%0d passed", pass_count, pass_count + fail_count);

        if (fail_count > 0)
            $display("*** SOME TESTS FAILED ***");
        else
            $display("All tests passed!");

        $finish;
    end

    initial begin
        #(CLK_PERIOD * 10_000_000);
        $display("TIMEOUT");
        $finish;
    end

endmodule
