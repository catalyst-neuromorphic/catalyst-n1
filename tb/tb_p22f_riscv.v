// ============================================================================
// P22F Testbench: Embedded RISC-V Core + MMIO Bridge
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

module tb_p22f_riscv;

    parameter CLK_PERIOD = 10;

    reg clk, rst_n;
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    reg         rv_enable;
    reg         imem_we;
    reg  [11:0] imem_waddr;
    reg  [31:0] imem_wdata;

    // MMIO bridge outputs (directly observed)
    wire        mmio_valid, mmio_we;
    wire [15:0] mmio_addr;
    wire [31:0] mmio_wdata_w;
    reg  [31:0] mmio_rdata;
    reg         mmio_ready;

    wire        rv_halted;
    wire [31:0] pc_out;

    rv32i_core #(
        .IMEM_DEPTH(4096),
        .IMEM_ADDR_BITS(12),
        .DMEM_DEPTH(4096),
        .DMEM_ADDR_BITS(12)
    ) dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .enable     (rv_enable),
        .imem_we    (imem_we),
        .imem_waddr (imem_waddr),
        .imem_wdata (imem_wdata),
        .mmio_valid (mmio_valid),
        .mmio_we    (mmio_we),
        .mmio_addr  (mmio_addr),
        .mmio_wdata (mmio_wdata_w),
        .mmio_rdata (mmio_rdata),
        .mmio_ready (mmio_ready),
        .halted     (rv_halted),
        .pc_out     (pc_out)
    );

    // MMIO auto-acknowledge (1-cycle ready)
    always @(posedge clk) begin
        mmio_ready <= mmio_valid;
    end

    // Capture MMIO writes for verification
    reg [31:0] last_mmio_addr;
    reg [31:0] last_mmio_wdata;
    reg        last_mmio_we;
    reg        mmio_write_seen;

    always @(posedge clk) begin
        if (mmio_valid && mmio_we) begin
            last_mmio_addr  <= {16'hFFFF, mmio_addr};
            last_mmio_wdata <= mmio_wdata_w;
            last_mmio_we    <= 1'b1;
            mmio_write_seen <= 1'b1;
        end
    end


    // R-type: funct7[6:0] rs2[4:0] rs1[4:0] funct3[2:0] rd[4:0] opcode[6:0]
    function [31:0] r_type;
        input [6:0] funct7;
        input [4:0] rs2, rs1;
        input [2:0] funct3;
        input [4:0] rd;
        input [6:0] opcode;
        r_type = {funct7, rs2, rs1, funct3, rd, opcode};
    endfunction

    // I-type: imm[11:0] rs1[4:0] funct3[2:0] rd[4:0] opcode[6:0]
    function [31:0] i_type;
        input [11:0] imm;
        input [4:0]  rs1;
        input [2:0]  funct3;
        input [4:0]  rd;
        input [6:0]  opcode;
        i_type = {imm, rs1, funct3, rd, opcode};
    endfunction

    // S-type: imm[11:5] rs2[4:0] rs1[4:0] funct3[2:0] imm[4:0] opcode[6:0]
    function [31:0] s_type;
        input [11:0] imm;
        input [4:0]  rs2, rs1;
        input [2:0]  funct3;
        input [6:0]  opcode;
        s_type = {imm[11:5], rs2, rs1, funct3, imm[4:0], opcode};
    endfunction

    // U-type: imm[31:12] rd[4:0] opcode[6:0]
    function [31:0] u_type;
        input [19:0] imm;
        input [4:0]  rd;
        input [6:0]  opcode;
        u_type = {imm, rd, opcode};
    endfunction

    localparam OP_IMM   = 7'b0010011;
    localparam OP_REG   = 7'b0110011;
    localparam OP_LUI   = 7'b0110111;
    localparam OP_LOAD  = 7'b0000011;
    localparam OP_STORE = 7'b0100011;
    localparam OP_ECALL = 7'b1110011;

    // Funct3 for ALU
    localparam F3_ADD  = 3'b000;
    localparam F3_SLL  = 3'b001;
    localparam F3_SLT  = 3'b010;
    localparam F3_SLTU = 3'b011;
    localparam F3_XOR  = 3'b100;
    localparam F3_SRL  = 3'b101;
    localparam F3_OR   = 3'b110;
    localparam F3_AND  = 3'b111;

    // Funct3 for load/store
    localparam F3_W    = 3'b010;

    function [31:0] ADDI;
        input [4:0] rd, rs1;
        input [11:0] imm;
        ADDI = i_type(imm, rs1, F3_ADD, rd, OP_IMM);
    endfunction

    function [31:0] ADD;
        input [4:0] rd, rs1, rs2;
        ADD = r_type(7'b0000000, rs2, rs1, F3_ADD, rd, OP_REG);
    endfunction

    function [31:0] SUB;
        input [4:0] rd, rs1, rs2;
        SUB = r_type(7'b0100000, rs2, rs1, F3_ADD, rd, OP_REG);
    endfunction

    function [31:0] AND_R;
        input [4:0] rd, rs1, rs2;
        AND_R = r_type(7'b0000000, rs2, rs1, F3_AND, rd, OP_REG);
    endfunction

    function [31:0] OR_R;
        input [4:0] rd, rs1, rs2;
        OR_R = r_type(7'b0000000, rs2, rs1, F3_OR, rd, OP_REG);
    endfunction

    function [31:0] SLLI;
        input [4:0] rd, rs1, shamt;
        SLLI = i_type({7'b0000000, shamt}, rs1, F3_SLL, rd, OP_IMM);
    endfunction

    function [31:0] SRLI;
        input [4:0] rd, rs1, shamt;
        SRLI = i_type({7'b0000000, shamt}, rs1, F3_SRL, rd, OP_IMM);
    endfunction

    function [31:0] SRAI;
        input [4:0] rd, rs1, shamt;
        SRAI = i_type({7'b0100000, shamt}, rs1, F3_SRL, rd, OP_IMM);
    endfunction

    function [31:0] LUI;
        input [4:0]  rd;
        input [19:0] imm;
        LUI = u_type(imm, rd, OP_LUI);
    endfunction

    function [31:0] SW;
        input [4:0]  rs2, rs1;
        input [11:0] offset;
        SW = s_type(offset, rs2, rs1, F3_W, OP_STORE);
    endfunction

    function [31:0] LW;
        input [4:0] rd, rs1;
        input [11:0] offset;
        LW = i_type(offset, rs1, F3_W, rd, OP_LOAD);
    endfunction

    function [31:0] ECALL;
        input dummy;
        ECALL = 32'h00000073;
    endfunction

    task prog_instr;
        input [11:0] addr;
        input [31:0] data;
    begin
        @(posedge clk);
        imem_we    <= 1;
        imem_waddr <= addr;
        imem_wdata <= data;
        @(posedge clk);
        imem_we <= 0;
    end
    endtask

    task wait_halt;
        integer timeout;
    begin
        timeout = 0;
        while (!rv_halted && timeout < 2000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        if (timeout >= 2000)
            $display("  WARNING: halt timeout");
    end
    endtask

    integer pass_count, fail_count;

    initial begin
        #5000000;
        $display("TIMEOUT");
        $finish;
    end

    initial begin
        clk = 0; rst_n = 0;
        rv_enable = 0;
        imem_we = 0; imem_waddr = 0; imem_wdata = 0;
        mmio_rdata = 0; mmio_ready = 0;
        mmio_write_seen = 0;
        last_mmio_addr = 0; last_mmio_wdata = 0; last_mmio_we = 0;
        pass_count = 0; fail_count = 0;

        #100;
        rst_n = 1;
        #100;

        // Test 1: ALU operations
        //   x1 = 100       (ADDI x1, x0, 100)
        //   x2 = 200       (ADDI x2, x0, 200)
        //   x3 = x1 + x2   (ADD x3, x1, x2)       → 300
        //   x4 = x2 - x1   (SUB x4, x2, x1)       → 100
        //   x5 = x1 & x2   (AND x5, x1, x2)       → 100 & 200 = 64
        //   x6 = x1 | x2   (OR  x6, x1, x2)       → 100 | 200 = 236
        //   x7 = x1 << 2   (SLLI x7, x1, 2)       → 400
        //   x8 = x2 >> 3   (SRLI x8, x2, 3)       → 25
        //   ECALL (halt)
        $display("\n=== Test 1: ALU operations ===");

        prog_instr(12'd0, ADDI(5'd1, 5'd0, 12'd100));     // x1 = 100
        prog_instr(12'd1, ADDI(5'd2, 5'd0, 12'd200));     // x2 = 200
        prog_instr(12'd2, ADD(5'd3, 5'd1, 5'd2));          // x3 = x1+x2
        prog_instr(12'd3, SUB(5'd4, 5'd2, 5'd1));          // x4 = x2-x1
        prog_instr(12'd4, AND_R(5'd5, 5'd1, 5'd2));        // x5 = x1&x2
        prog_instr(12'd5, OR_R(5'd6, 5'd1, 5'd2));         // x6 = x1|x2
        prog_instr(12'd6, SLLI(5'd7, 5'd1, 5'd2));         // x7 = x1<<2
        prog_instr(12'd7, SRLI(5'd8, 5'd2, 5'd3));         // x8 = x2>>3
        prog_instr(12'd8, ECALL(0));                         // halt

        rv_enable = 1;
        wait_halt;

        // Verify registers by accessing DUT internals
        if (dut.regfile[1] == 100 && dut.regfile[2] == 200 &&
            dut.regfile[3] == 300 && dut.regfile[4] == 100 &&
            dut.regfile[5] == (100 & 200) && dut.regfile[6] == (100 | 200) &&
            dut.regfile[7] == 400 && dut.regfile[8] == 25) begin
            $display("  PASSED: ALU x1=%0d x2=%0d x3=%0d x4=%0d x5=%0d x6=%0d x7=%0d x8=%0d",
                dut.regfile[1], dut.regfile[2], dut.regfile[3], dut.regfile[4],
                dut.regfile[5], dut.regfile[6], dut.regfile[7], dut.regfile[8]);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: x1=%0d x2=%0d x3=%0d x4=%0d x5=%0d x6=%0d x7=%0d x8=%0d",
                dut.regfile[1], dut.regfile[2], dut.regfile[3], dut.regfile[4],
                dut.regfile[5], dut.regfile[6], dut.regfile[7], dut.regfile[8]);
            fail_count = fail_count + 1;
        end

        // Disable and reset for next test
        rv_enable = 0;
        #50;

        // Test 2: Memory load/store
        //   x1 = 0x1234     (LUI + ADDI)
        //   SW x1, 0(x0)    (store to dmem[0])
        //   LW x2, 0(x0)    (load from dmem[0])
        //   x3 = 0xABCD
        //   SW x3, 4(x0)    (store to dmem[1])
        //   LW x4, 4(x0)    (load from dmem[1])
        //   ECALL
        $display("\n=== Test 2: Memory load/store ===");

        prog_instr(12'd0, ADDI(5'd1, 5'd0, 12'h234));  // x1 = 0x234 (low 12 bits)
        prog_instr(12'd1, SW(5'd1, 5'd0, 12'd0));       // dmem[0] = x1
        prog_instr(12'd2, LW(5'd2, 5'd0, 12'd0));       // x2 = dmem[0]
        prog_instr(12'd3, ADDI(5'd3, 5'd0, 12'hBCD));   // x3 = sign-ext 0xBCD = -1075
        prog_instr(12'd4, SW(5'd3, 5'd0, 12'd4));       // dmem[1] = x3
        prog_instr(12'd5, LW(5'd4, 5'd0, 12'd4));       // x4 = dmem[1]
        prog_instr(12'd6, ECALL(0));

        rv_enable = 1;
        wait_halt;

        // 0x234 = 564
        // 0xBCD sign-extended = 0xFFFFFBCD = -1075
        if (dut.regfile[2] == 32'h234 && dut.regfile[4] == 32'hFFFFFBCD) begin
            $display("  PASSED: x2=0x%08h x4=0x%08h", dut.regfile[2], dut.regfile[4]);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: x2=0x%08h (exp 0x234) x4=0x%08h (exp 0xFFFFFBCD)",
                dut.regfile[2], dut.regfile[4]);
            fail_count = fail_count + 1;
        end

        rv_enable = 0;
        #50;

        // Test 3: MMIO spike inject
        //   Write to 0xFFFF_0018 (spike inject)
        //   The MMIO bridge receives this and asserts ext_valid
        //
        //   Program: load 0xFFFF into x10 upper, then add offset
        //   x10 = 0xFFFF0000   (LUI x10, 0xFFFFF)
        //   x11 = 42           (neuron 42, current in upper bits)
        //   SW x11, 0x18(x10)  (write to spike inject register)
        //   ECALL
        $display("\n=== Test 3: MMIO spike inject ===");

        // LUI x10, 0xFFFFF → x10 = 0xFFFFF000
        // ADDI x10, x10, 0 → already have 0xFFFFF000
        // LUI x10, 0xFFFF0 → x10 = 0xFFFF0000
        prog_instr(12'd0, u_type(20'hFFFF0, 5'd10, OP_LUI));  // x10 = 0xFFFF0000
        prog_instr(12'd1, ADDI(5'd11, 5'd0, 12'd42));          // x11 = 42
        // SW x11, 0x18(x10) → store x11 to addr 0xFFFF0018
        prog_instr(12'd2, SW(5'd11, 5'd10, 12'h018));
        prog_instr(12'd3, ECALL(0));

        mmio_write_seen = 0;
        rv_enable = 1;
        wait_halt;

        if (mmio_write_seen && last_mmio_addr == 32'hFFFF0018) begin
            $display("  PASSED: MMIO write to 0x%08h data=0x%08h",
                last_mmio_addr, last_mmio_wdata);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: mmio_write_seen=%0b addr=0x%08h",
                mmio_write_seen, last_mmio_addr);
            fail_count = fail_count + 1;
        end

        rv_enable = 0;
        #50;

        // Test 4: MMIO UART TX
        //   Write byte 0x55 to UART TX register (0xFFFF0020)
        $display("\n=== Test 4: MMIO UART TX write ===");

        prog_instr(12'd0, u_type(20'hFFFF0, 5'd10, OP_LUI));  // x10 = 0xFFFF0000
        prog_instr(12'd1, ADDI(5'd11, 5'd0, 12'h055));         // x11 = 0x55
        prog_instr(12'd2, SW(5'd11, 5'd10, 12'h020));          // SW to 0xFFFF0020
        prog_instr(12'd3, ECALL(0));

        mmio_write_seen = 0;
        rv_enable = 1;
        wait_halt;

        if (mmio_write_seen && last_mmio_addr == 32'hFFFF0020 &&
            last_mmio_wdata[7:0] == 8'h55) begin
            $display("  PASSED: UART TX byte=0x%02h", last_mmio_wdata[7:0]);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: mmio_write_seen=%0b addr=0x%08h data=0x%08h",
                mmio_write_seen, last_mmio_addr, last_mmio_wdata);
            fail_count = fail_count + 1;
        end

        rv_enable = 0;

        $display("\n====================================");
        $display("P22F RESULTS: %0d/%0d passed", pass_count, pass_count + fail_count);
        $display("====================================\n");

        if (fail_count > 0)
            $display("SOME TESTS FAILED");

        $finish;
    end

endmodule
