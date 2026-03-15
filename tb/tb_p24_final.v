`timescale 1ns/1ps

module tb_p24_final;
    reg clk, rst_n;
    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;
    integer total_tests = 8;

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

    function [31:0] enc_fcvt_s_w;
        input [4:0] fd, rs1;
        enc_fcvt_s_w = {7'b1101000, 5'b00000, rs1, 3'b000, fd, 7'b1010011};
    endfunction

    function [31:0] enc_fcvt_w_s;
        input [4:0] rd, fs1;
        enc_fcvt_w_s = {7'b1100000, 5'b00000, fs1, 3'b000, rd, 7'b1010011};
    endfunction

    function [31:0] enc_fadd;
        input [4:0] fd, fs1, fs2;
        enc_fadd = {7'b0000000, fs2, fs1, 3'b000, fd, 7'b1010011};
    endfunction

    function [31:0] enc_fmul;
        input [4:0] fd, fs1, fs2;
        enc_fmul = {7'b0001000, fs2, fs1, 3'b000, fd, 7'b1010011};
    endfunction

    function [31:0] enc_fdiv;
        input [4:0] fd, fs1, fs2;
        enc_fdiv = {7'b0001100, fs2, fs1, 3'b000, fd, 7'b1010011};
    endfunction

    function [31:0] enc_flt;
        input [4:0] rd, fs1, fs2;
        enc_flt = {7'b1010000, fs2, fs1, 3'b001, rd, 7'b1010011};
    endfunction

    localparam [31:0] ECALL = 32'h00000073;

    localparam IMEM_D = 65536;
    localparam IMEM_A = 16;
    localparam DMEM_D = 65536;
    localparam DMEM_A = 16;

    reg         core_enable;
    reg         core_imem_we;
    reg  [IMEM_A-1:0] core_imem_waddr;
    reg  [31:0] core_imem_wdata;
    wire        core_mmio_valid, core_mmio_we;
    wire [15:0] core_mmio_addr;
    wire [31:0] core_mmio_wdata;
    wire        core_halted;
    wire [31:0] core_pc;

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
        .halted(core_halted), .pc_out(core_pc)
    );

    reg [31:0] mmio_capture [0:7];
    reg [2:0]  mmio_cap_idx;

    always @(posedge clk) begin
        if (core_mmio_valid && core_mmio_we && core_mmio_ready) begin
            mmio_capture[mmio_cap_idx] <= core_mmio_wdata;
            mmio_cap_idx <= mmio_cap_idx + 1;
        end
    end

    localparam CL_IMEM_D = 256;
    localparam CL_IMEM_A = 8;
    localparam CL_DMEM_D = 256;
    localparam CL_DMEM_A = 8;

    reg  [2:0]  cl_enable;
    reg         cl_imem_we_0, cl_imem_we_1, cl_imem_we_2;
    reg  [CL_IMEM_A-1:0] cl_imem_waddr_0, cl_imem_waddr_1, cl_imem_waddr_2;
    reg  [31:0] cl_imem_wdata_0, cl_imem_wdata_1, cl_imem_wdata_2;
    wire        cl_mmio_valid, cl_mmio_we;
    wire [15:0] cl_mmio_addr;
    wire [31:0] cl_mmio_wdata;
    wire [2:0]  cl_halted;
    wire [31:0] cl_pc_0, cl_pc_1, cl_pc_2;

    wire cl_mmio_ready = cl_mmio_valid;

    rv32im_cluster #(
        .IMEM_DEPTH(CL_IMEM_D), .IMEM_ADDR_BITS(CL_IMEM_A),
        .DMEM_DEPTH(CL_DMEM_D), .DMEM_ADDR_BITS(CL_DMEM_A)
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

    task core_reset_and_run;
        begin
            core_enable  <= 0;
            mmio_cap_idx <= 0;
            @(posedge clk); @(posedge clk);
            core_enable  <= 1;
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

    task cluster_program_core;
        input integer core_id;
        input [CL_IMEM_A-1:0] addr;
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

    initial begin
        $dumpfile("tb_p24_final.vcd");
        $dumpvars(0, tb_p24_final);

        rst_n = 0;
        core_enable = 0;
        core_imem_we = 0; core_imem_waddr = 0; core_imem_wdata = 0;
        mmio_cap_idx = 0;
        cl_enable = 0;
        cl_imem_we_0 = 0; cl_imem_we_1 = 0; cl_imem_we_2 = 0;
        cl_imem_waddr_0 = 0; cl_imem_waddr_1 = 0; cl_imem_waddr_2 = 0;
        cl_imem_wdata_0 = 0; cl_imem_wdata_1 = 0; cl_imem_wdata_2 = 0;
        cl_cap_idx = 0;

        #100;
        rst_n = 1;
        #20;

        $display("\ntest 1: RISC-V high memory (P24A)");
        core_program(0,  enc_addi(5'd1, 5'd0, 12'd42));
        core_program(1,  enc_lui(5'd2, 20'h00027));
        core_program(2,  enc_addi(5'd2, 5'd2, 12'h100));
        core_program(3,  enc_sw(5'd1, 5'd2, 12'd0));
        core_program(4,  enc_lw(5'd3, 5'd2, 12'd0));
        core_program(5,  enc_lui(5'd31, 20'hFFFF0));
        core_program(6,  enc_sw(5'd3, 5'd31, 12'd0));
        core_program(7,  ECALL);
        core_reset_and_run;
        wait_core_halt(200);

        if (mmio_capture[0] === 32'd42) begin
            $display("  PASSED: High memory store/load returned %0d", mmio_capture[0]);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: Expected 42, got %0d", mmio_capture[0]);
            fail_count = fail_count + 1;
        end

        $display("\ntest 2: RISC-V large IMEM (P24A)");
        core_enable <= 0;
        @(posedge clk); @(posedge clk);
        core_program(0, enc_lui(5'd1, 20'h0002A));
        core_program(1, {12'd0, 5'd1, 3'b000, 5'd0, 7'b1100111});
        core_program(16'hA800, enc_addi(5'd10, 5'd0, 12'd99));
        core_program(16'hA801, enc_lui(5'd31, 20'hFFFF0));
        core_program(16'hA802, enc_sw(5'd10, 5'd31, 12'd0));
        core_program(16'hA803, ECALL);
        core_reset_and_run;
        wait_core_halt(200);

        if (mmio_capture[0] === 32'd99) begin
            $display("  PASSED: Executed at high IMEM address, got %0d", mmio_capture[0]);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: Expected 99, got %0d", mmio_capture[0]);
            fail_count = fail_count + 1;
        end

        $display("\ntest 3: FPU FADD+FMUL (P24D)");
        core_enable <= 0;
        @(posedge clk); @(posedge clk);
        core_program(0,  enc_addi(5'd1, 5'd0, 12'd3));
        core_program(1,  enc_fcvt_s_w(5'd1, 5'd1));
        core_program(2,  enc_addi(5'd2, 5'd0, 12'd4));
        core_program(3,  enc_fcvt_s_w(5'd2, 5'd2));
        core_program(4,  enc_fadd(5'd3, 5'd1, 5'd2));
        core_program(5,  enc_addi(5'd3, 5'd0, 12'd10));
        core_program(6,  enc_fcvt_s_w(5'd4, 5'd3));
        core_program(7,  enc_fmul(5'd5, 5'd3, 5'd4));
        core_program(8,  enc_fcvt_w_s(5'd10, 5'd5));
        core_program(9,  enc_lui(5'd31, 20'hFFFF0));
        core_program(10, enc_sw(5'd10, 5'd31, 12'd0));
        core_program(11, ECALL);
        core_reset_and_run;
        wait_core_halt(200);

        if (mmio_capture[0] === 32'd70) begin
            $display("  PASSED: FADD+FMUL round-trip = %0d", mmio_capture[0]);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: Expected 70, got %0d (0x%08h)", mmio_capture[0], mmio_capture[0]);
            fail_count = fail_count + 1;
        end

        $display("\ntest 4: FPU FDIV+compare (P24D)");
        core_enable <= 0;
        @(posedge clk); @(posedge clk);
        core_program(0,  enc_addi(5'd1, 5'd0, 12'd100));
        core_program(1,  enc_fcvt_s_w(5'd1, 5'd1));
        core_program(2,  enc_addi(5'd2, 5'd0, 12'd3));
        core_program(3,  enc_fcvt_s_w(5'd2, 5'd2));
        core_program(4,  enc_fdiv(5'd3, 5'd1, 5'd2));
        core_program(5,  enc_fcvt_w_s(5'd10, 5'd3));
        core_program(6,  enc_addi(5'd3, 5'd0, 12'd34));
        core_program(7,  enc_fcvt_s_w(5'd4, 5'd3));
        core_program(8,  enc_flt(5'd11, 5'd3, 5'd4));
        core_program(9,  enc_lui(5'd31, 20'hFFFF0));
        core_program(10, enc_sw(5'd10, 5'd31, 12'd0));
        core_program(11, enc_sw(5'd11, 5'd31, 12'd4));
        core_program(12, ECALL);
        core_reset_and_run;
        wait_core_halt(200);

        if (mmio_capture[0] === 32'd33 && mmio_capture[1] === 32'd1) begin
            $display("  PASSED: FDIV=33, FLT=1");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: Expected 33 & 1, got %0d & %0d", mmio_capture[0], mmio_capture[1]);
            fail_count = fail_count + 1;
        end

        $display("\ntest 5: Triple RISC-V cluster (P24C)");
        cluster_program_core(0, 0, enc_addi(5'd1, 5'd0, 12'h0AA));
        cluster_program_core(0, 1, enc_lui(5'd31, 20'hFFFF0));
        cluster_program_core(0, 2, enc_sw(5'd1, 5'd31, 12'd0));
        cluster_program_core(0, 3, ECALL);
        cluster_program_core(1, 0, enc_addi(5'd1, 5'd0, 12'h0BB));
        cluster_program_core(1, 1, enc_lui(5'd31, 20'hFFFF0));
        cluster_program_core(1, 2, enc_sw(5'd1, 5'd31, 12'd0));
        cluster_program_core(1, 3, ECALL);
        cluster_program_core(2, 0, enc_addi(5'd1, 5'd0, 12'h0CC));
        cluster_program_core(2, 1, enc_lui(5'd31, 20'hFFFF0));
        cluster_program_core(2, 2, enc_sw(5'd1, 5'd31, 12'd0));
        cluster_program_core(2, 3, ECALL);

        cl_cap_idx <= 0;
        cl_enable  <= 3'b111;
        #2000;

        if (cl_halted === 3'b111) begin
            $display("  PASSED: All 3 cores halted");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: halted=%b, expected 111", cl_halted);
            fail_count = fail_count + 1;
        end

        $display("\ntest 6: Cluster MMIO values (P24C)");
        begin
            reg found_aa, found_bb, found_cc;
            integer ci;
            found_aa = 0; found_bb = 0; found_cc = 0;
            for (ci = 0; ci < 3; ci = ci + 1) begin
                if (cl_mmio_cap[ci] == 32'h0AA) found_aa = 1;
                if (cl_mmio_cap[ci] == 32'h0BB) found_bb = 1;
                if (cl_mmio_cap[ci] == 32'h0CC) found_cc = 1;
            end
            if (found_aa && found_bb && found_cc) begin
                $display("  PASSED: All 3 MMIO values received (0xAA, 0xBB, 0xCC)");
                pass_count = pass_count + 1;
            end else begin
                $display("  FAILED: Missing MMIO values. Got: [0]=%0h [1]=%0h [2]=%0h",
                         cl_mmio_cap[0], cl_mmio_cap[1], cl_mmio_cap[2]);
                fail_count = fail_count + 1;
            end
        end

        $display("\ntest 7: FPU sign injection (P24D)");
        core_enable <= 0;
        @(posedge clk); @(posedge clk);
        core_program(0,  enc_addi(5'd1, 5'd0, 12'd5));
        core_program(1,  enc_fcvt_s_w(5'd1, 5'd1));
        core_program(2,  enc_lui(5'd2, 20'hBF800));
        core_program(3,  {7'b1111000, 5'b00000, 5'd2, 3'b000, 5'd2, 7'b1010011});
        core_program(4,  {7'b0010000, 5'd2, 5'd1, 3'b000, 5'd3, 7'b1010011});
        core_program(5,  {7'b1110000, 5'b00000, 5'd3, 3'b000, 5'd10, 7'b1010011});
        core_program(6,  enc_lui(5'd31, 20'hFFFF0));
        core_program(7,  enc_sw(5'd10, 5'd31, 12'd0));
        core_program(8,  ECALL);
        core_reset_and_run;
        wait_core_halt(200);

        if (mmio_capture[0] === 32'hC0A00000) begin
            $display("  PASSED: FSGNJ(-5.0) = 0x%08h", mmio_capture[0]);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: Expected 0xC0A00000, got 0x%08h", mmio_capture[0]);
            fail_count = fail_count + 1;
        end

        $display("\ntest 8: FPU FMIN/FMAX (P24D)");
        core_enable <= 0;
        @(posedge clk); @(posedge clk);
        core_program(0,  enc_addi(5'd1, 5'd0, 12'd7));
        core_program(1,  enc_fcvt_s_w(5'd1, 5'd1));
        core_program(2,  enc_addi(5'd2, 5'd0, 12'd3));
        core_program(3,  enc_fcvt_s_w(5'd2, 5'd2));
        core_program(4,  {7'b0010100, 5'd2, 5'd1, 3'b000, 5'd3, 7'b1010011});
        core_program(5,  {7'b0010100, 5'd2, 5'd1, 3'b001, 5'd4, 7'b1010011});
        core_program(6,  enc_fcvt_w_s(5'd10, 5'd3));
        core_program(7,  enc_fcvt_w_s(5'd11, 5'd4));
        core_program(8,  enc_lui(5'd31, 20'hFFFF0));
        core_program(9,  enc_sw(5'd10, 5'd31, 12'd0));
        core_program(10, enc_sw(5'd11, 5'd31, 12'd4));
        core_program(11, ECALL);
        core_reset_and_run;
        wait_core_halt(200);

        if (mmio_capture[0] === 32'd3 && mmio_capture[1] === 32'd7) begin
            $display("  PASSED: FMIN=3, FMAX=7");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: Expected 3 & 7, got %0d & %0d", mmio_capture[0], mmio_capture[1]);
            fail_count = fail_count + 1;
        end

        $display("\n=== P24 RESULTS: %0d passed, %0d failed out of %0d ===",
                 pass_count, fail_count, total_tests);
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");

        #100;
        $finish;
    end

    initial begin
        #500000;
        $display("TIMEOUT!");
        $finish;
    end

endmodule
