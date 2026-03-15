// ============================================================================
// tb_p24_final.v - P24 Validation Testbench
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

`timescale 1ns/1ps

module tb_p24_final;
    reg clk, rst_n;
    initial clk = 0;
    always #5 clk = ~clk;  // 100 MHz

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

    function [31:0] enc_fcvt_s_w;  // FCVT.S.W fd, rs1 (int→float)
        input [4:0] fd, rs1;
        enc_fcvt_s_w = {7'b1101000, 5'b00000, rs1, 3'b000, fd, 7'b1010011};
    endfunction

    function [31:0] enc_fcvt_w_s;  // FCVT.W.S rd, fs1 (float→int, truncate)
        input [4:0] rd, fs1;
        enc_fcvt_w_s = {7'b1100000, 5'b00000, fs1, 3'b000, rd, 7'b1010011};
    endfunction

    function [31:0] enc_fadd;  // FADD.S fd, fs1, fs2
        input [4:0] fd, fs1, fs2;
        enc_fadd = {7'b0000000, fs2, fs1, 3'b000, fd, 7'b1010011};
    endfunction

    function [31:0] enc_fmul;  // FMUL.S fd, fs1, fs2
        input [4:0] fd, fs1, fs2;
        enc_fmul = {7'b0001000, fs2, fs1, 3'b000, fd, 7'b1010011};
    endfunction

    function [31:0] enc_fdiv;  // FDIV.S fd, fs1, fs2
        input [4:0] fd, fs1, fs2;
        enc_fdiv = {7'b0001100, fs2, fs1, 3'b000, fd, 7'b1010011};
    endfunction

    function [31:0] enc_flt;  // FLT.S rd, fs1, fs2 (float less-than → int)
        input [4:0] rd, fs1, fs2;
        enc_flt = {7'b1010000, fs2, fs1, 3'b001, rd, 7'b1010011};
    endfunction

    localparam [31:0] ECALL = 32'h00000073;

    localparam IMEM_D = 65536;  // P24A: 256KB
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

    // Instant MMIO ack
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

    // Capture MMIO writes
    reg [31:0] mmio_capture [0:7];
    reg [2:0]  mmio_cap_idx;

    always @(posedge clk) begin
        if (core_mmio_valid && core_mmio_we && core_mmio_ready) begin
            mmio_capture[mmio_cap_idx] <= core_mmio_wdata;
            mmio_cap_idx <= mmio_cap_idx + 1;
        end
    end

    localparam CL_IMEM_D = 256;   // Small for test
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

    // Capture cluster MMIO writes
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

        // Store 42 at DMEM word address 40000, load back, output via MMIO
        $display("\n--- TEST 1: RISC-V high memory (P24A) ---");
        core_program(0,  enc_addi(5'd1, 5'd0, 12'd42));        // x1 = 42
        core_program(1,  enc_lui(5'd2, 20'h00027));             // x2 = 0x27000
        core_program(2,  enc_addi(5'd2, 5'd2, 12'h100));       // x2 = 0x27100 (word addr 0x9C40)
        core_program(3,  enc_sw(5'd1, 5'd2, 12'd0));           // SW x1, 0(x2)
        core_program(4,  enc_lw(5'd3, 5'd2, 12'd0));           // LW x3, 0(x2)
        core_program(5,  enc_lui(5'd31, 20'hFFFF0));            // x31 = 0xFFFF0000
        core_program(6,  enc_sw(5'd3, 5'd31, 12'd0));          // MMIO write x3
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

        // Execute instruction at word address 40000
        $display("\n--- TEST 2: RISC-V large IMEM (P24A) ---");
        core_enable <= 0;
        @(posedge clk); @(posedge clk);
        // Program a jump to high address, and the instruction there
        core_program(0, enc_lui(5'd1, 20'h0002A));              // x1 = 0x2A000
        // JAL x0, offset → need to encode JAL to address 40000*4 = 160000 = 0x27100
        // Simpler: use JALR to jump to x1
        // JALR x0, x1, 0 = {12'd0, rs1=1, 3'b000, rd=0, 7'b1100111}
        core_program(1, {12'd0, 5'd1, 3'b000, 5'd0, 7'b1100111}); // JALR x0, x1, 0
        // At word address 0x2A000/4 = 0xA800:
        core_program(16'hA800, enc_addi(5'd10, 5'd0, 12'd99));  // x10 = 99
        core_program(16'hA801, enc_lui(5'd31, 20'hFFFF0));       // x31 = MMIO base
        core_program(16'hA802, enc_sw(5'd10, 5'd31, 12'd0));    // MMIO write 99
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

        // 3.0 + 4.0 = 7.0, 7.0 * 10.0 = 70.0, convert to int → 70
        $display("\n--- TEST 3: FPU FADD+FMUL (P24D) ---");
        core_enable <= 0;
        @(posedge clk); @(posedge clk);
        core_program(0,  enc_addi(5'd1, 5'd0, 12'd3));          // x1 = 3
        core_program(1,  enc_fcvt_s_w(5'd1, 5'd1));             // f1 = 3.0
        core_program(2,  enc_addi(5'd2, 5'd0, 12'd4));          // x2 = 4
        core_program(3,  enc_fcvt_s_w(5'd2, 5'd2));             // f2 = 4.0
        core_program(4,  enc_fadd(5'd3, 5'd1, 5'd2));           // f3 = 7.0
        core_program(5,  enc_addi(5'd3, 5'd0, 12'd10));         // x3 = 10
        core_program(6,  enc_fcvt_s_w(5'd4, 5'd3));             // f4 = 10.0
        core_program(7,  enc_fmul(5'd5, 5'd3, 5'd4));           // f5 = 70.0
        core_program(8,  enc_fcvt_w_s(5'd10, 5'd5));            // x10 = 70
        core_program(9,  enc_lui(5'd31, 20'hFFFF0));             // x31 = MMIO base
        core_program(10, enc_sw(5'd10, 5'd31, 12'd0));          // MMIO write 70
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

        // 100.0 / 3.0 = 33.333..., truncate to 33
        // 33.333 < 34.0 → 1
        $display("\n--- TEST 4: FPU FDIV+compare (P24D) ---");
        core_enable <= 0;
        @(posedge clk); @(posedge clk);
        core_program(0,  enc_addi(5'd1, 5'd0, 12'd100));        // x1 = 100
        core_program(1,  enc_fcvt_s_w(5'd1, 5'd1));             // f1 = 100.0
        core_program(2,  enc_addi(5'd2, 5'd0, 12'd3));          // x2 = 3
        core_program(3,  enc_fcvt_s_w(5'd2, 5'd2));             // f2 = 3.0
        core_program(4,  enc_fdiv(5'd3, 5'd1, 5'd2));           // f3 = 33.333...
        core_program(5,  enc_fcvt_w_s(5'd10, 5'd3));            // x10 = 33
        core_program(6,  enc_addi(5'd3, 5'd0, 12'd34));         // x3 = 34
        core_program(7,  enc_fcvt_s_w(5'd4, 5'd3));             // f4 = 34.0
        core_program(8,  enc_flt(5'd11, 5'd3, 5'd4));           // x11 = FLT(f3, f4)
        core_program(9,  enc_lui(5'd31, 20'hFFFF0));
        core_program(10, enc_sw(5'd10, 5'd31, 12'd0));          // MMIO write 33
        core_program(11, enc_sw(5'd11, 5'd31, 12'd4));          // MMIO write FLT result
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

        $display("\n--- TEST 5: Triple RISC-V cluster (P24C) ---");
        // Core 0: write 0xAA to MMIO
        cluster_program_core(0, 0, enc_addi(5'd1, 5'd0, 12'h0AA));
        cluster_program_core(0, 1, enc_lui(5'd31, 20'hFFFF0));
        cluster_program_core(0, 2, enc_sw(5'd1, 5'd31, 12'd0));
        cluster_program_core(0, 3, ECALL);
        // Core 1: write 0xBB to MMIO
        cluster_program_core(1, 0, enc_addi(5'd1, 5'd0, 12'h0BB));
        cluster_program_core(1, 1, enc_lui(5'd31, 20'hFFFF0));
        cluster_program_core(1, 2, enc_sw(5'd1, 5'd31, 12'd0));
        cluster_program_core(1, 3, ECALL);
        // Core 2: write 0xCC to MMIO
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

        $display("\n--- TEST 6: Cluster MMIO values (P24C) ---");
        // Verify all 3 MMIO writes arrived (order: 0xAA, 0xBB, 0xCC due to priority)
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

        // FSGNJ.S: copy sign from f2 to f1
        $display("\n--- TEST 7: FPU sign injection (P24D) ---");
        core_enable <= 0;
        @(posedge clk); @(posedge clk);
        // f1 = 5.0 (positive)
        core_program(0,  enc_addi(5'd1, 5'd0, 12'd5));
        core_program(1,  enc_fcvt_s_w(5'd1, 5'd1));             // f1 = 5.0
        // f2 = -1.0 (negative) via FMV.W.X with 0xBF800000
        // Load 0xBF800000 into x2 (IEEE 754 for -1.0)
        // LUI x2, 0xBF800 then no ADDI needed (bottom 12 bits are 0)
        core_program(2,  enc_lui(5'd2, 20'hBF800));
        // FMV.W.X f2, x2: {7'b1111000, 5'b00000, rs1=x2, 3'b000, fd=2, 7'b1010011}
        core_program(3,  {7'b1111000, 5'b00000, 5'd2, 3'b000, 5'd2, 7'b1010011});
        // FSGNJ.S f3, f1, f2: copy sign of f2 (negative) to f1's magnitude
        // {7'b0010000, fs2=2, fs1=1, 3'b000, fd=3, 7'b1010011}
        core_program(4,  {7'b0010000, 5'd2, 5'd1, 3'b000, 5'd3, 7'b1010011});
        // FMV.X.W x10, f3: bitcast float to int
        // {7'b1110000, 5'b00000, fs1=3, 3'b000, rd=10, 7'b1010011}
        core_program(5,  {7'b1110000, 5'b00000, 5'd3, 3'b000, 5'd10, 7'b1010011});
        core_program(6,  enc_lui(5'd31, 20'hFFFF0));
        core_program(7,  enc_sw(5'd10, 5'd31, 12'd0));
        core_program(8,  ECALL);
        core_reset_and_run;
        wait_core_halt(200);

        // -5.0 in IEEE 754 = 0xC0A00000
        if (mmio_capture[0] === 32'hC0A00000) begin
            $display("  PASSED: FSGNJ(-5.0) = 0x%08h", mmio_capture[0]);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: Expected 0xC0A00000, got 0x%08h", mmio_capture[0]);
            fail_count = fail_count + 1;
        end

        $display("\n--- TEST 8: FPU FMIN/FMAX (P24D) ---");
        core_enable <= 0;
        @(posedge clk); @(posedge clk);
        core_program(0,  enc_addi(5'd1, 5'd0, 12'd7));          // x1 = 7
        core_program(1,  enc_fcvt_s_w(5'd1, 5'd1));             // f1 = 7.0
        core_program(2,  enc_addi(5'd2, 5'd0, 12'd3));          // x2 = 3
        core_program(3,  enc_fcvt_s_w(5'd2, 5'd2));             // f2 = 3.0
        // FMIN.S f3, f1, f2: {7'b0010100, fs2=2, fs1=1, 3'b000, fd=3, 7'b1010011}
        core_program(4,  {7'b0010100, 5'd2, 5'd1, 3'b000, 5'd3, 7'b1010011});
        // FMAX.S f4, f1, f2: {7'b0010100, fs2=2, fs1=1, 3'b001, fd=4, 7'b1010011}
        core_program(5,  {7'b0010100, 5'd2, 5'd1, 3'b001, 5'd4, 7'b1010011});
        core_program(6,  enc_fcvt_w_s(5'd10, 5'd3));            // x10 = int(min) = 3
        core_program(7,  enc_fcvt_w_s(5'd11, 5'd4));            // x11 = int(max) = 7
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
            $display("SOME TESTS FAILED!");

        #100;
        $finish;
    end

    initial begin
        #500000;
        $display("TIMEOUT!");
        $finish;
    end

endmodule
