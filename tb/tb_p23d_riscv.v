// ============================================================================
// P23D Testbench: RV32IM + CSR + Timer Interrupts + 64KB SRAM
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

module tb_p23d_riscv;

    parameter CLK_PERIOD = 10;

    reg clk, rst_n;
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    reg         rv_enable;
    reg         imem_we;
    reg  [13:0] imem_waddr;
    reg  [31:0] imem_wdata;

    wire        mmio_valid, mmio_we;
    wire [15:0] mmio_addr;
    wire [31:0] mmio_wdata_w;
    reg  [31:0] mmio_rdata;
    reg         mmio_ready;

    wire        rv_halted;
    wire [31:0] pc_out;

    rv32i_core #(
        .IMEM_DEPTH(16384),
        .IMEM_ADDR_BITS(14),
        .DMEM_DEPTH(16384),
        .DMEM_ADDR_BITS(14)
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

    // MMIO auto-acknowledge
    always @(posedge clk) begin
        mmio_ready <= mmio_valid;
    end

    // Capture MMIO writes
    reg [31:0] last_mmio_addr;
    reg [31:0] last_mmio_wdata;
    reg        mmio_write_seen;

    always @(posedge clk) begin
        if (mmio_valid && mmio_we) begin
            last_mmio_addr  <= {16'hFFFF, mmio_addr};
            last_mmio_wdata <= mmio_wdata_w;
            mmio_write_seen <= 1'b1;
        end
    end

    localparam OP_IMM    = 7'b0010011;
    localparam OP_REG    = 7'b0110011;
    localparam OP_LUI    = 7'b0110111;
    localparam OP_LOAD   = 7'b0000011;
    localparam OP_STORE  = 7'b0100011;
    localparam OP_SYSTEM = 7'b1110011;
    localparam OP_JAL    = 7'b1101111;

    localparam F3_ADD  = 3'b000;
    localparam F3_SLL  = 3'b001;
    localparam F3_SLT  = 3'b010;
    localparam F3_SLTU = 3'b011;
    localparam F3_XOR  = 3'b100;
    localparam F3_SRL  = 3'b101;
    localparam F3_OR   = 3'b110;
    localparam F3_AND  = 3'b111;
    localparam F3_W    = 3'b010;

    // R-type
    function [31:0] r_type;
        input [6:0] funct7;
        input [4:0] rs2, rs1;
        input [2:0] funct3;
        input [4:0] rd;
        input [6:0] opcode;
        r_type = {funct7, rs2, rs1, funct3, rd, opcode};
    endfunction

    // I-type
    function [31:0] i_type;
        input [11:0] imm;
        input [4:0]  rs1;
        input [2:0]  funct3;
        input [4:0]  rd;
        input [6:0]  opcode;
        i_type = {imm, rs1, funct3, rd, opcode};
    endfunction

    // S-type
    function [31:0] s_type;
        input [11:0] imm;
        input [4:0]  rs2, rs1;
        input [2:0]  funct3;
        input [6:0]  opcode;
        s_type = {imm[11:5], rs2, rs1, funct3, imm[4:0], opcode};
    endfunction

    // U-type
    function [31:0] u_type;
        input [19:0] imm;
        input [4:0]  rd;
        input [6:0]  opcode;
        u_type = {imm, rd, opcode};
    endfunction

    function [31:0] ADDI;
        input [4:0] rd, rs1;
        input [11:0] imm;
        ADDI = i_type(imm, rs1, F3_ADD, rd, OP_IMM);
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

    // M-extension: MUL rd, rs1, rs2 (funct7=0000001, funct3=000)
    function [31:0] MUL;
        input [4:0] rd, rs1, rs2;
        MUL = r_type(7'b0000001, rs2, rs1, 3'b000, rd, OP_REG);
    endfunction

    // MULH rd, rs1, rs2 (funct7=0000001, funct3=001)
    function [31:0] MULH;
        input [4:0] rd, rs1, rs2;
        MULH = r_type(7'b0000001, rs2, rs1, 3'b001, rd, OP_REG);
    endfunction

    // MULHU rd, rs1, rs2 (funct7=0000001, funct3=011)
    function [31:0] MULHU;
        input [4:0] rd, rs1, rs2;
        MULHU = r_type(7'b0000001, rs2, rs1, 3'b011, rd, OP_REG);
    endfunction

    // DIV rd, rs1, rs2 (funct7=0000001, funct3=100)
    function [31:0] DIV;
        input [4:0] rd, rs1, rs2;
        DIV = r_type(7'b0000001, rs2, rs1, 3'b100, rd, OP_REG);
    endfunction

    // DIVU rd, rs1, rs2 (funct7=0000001, funct3=101)
    function [31:0] DIVU;
        input [4:0] rd, rs1, rs2;
        DIVU = r_type(7'b0000001, rs2, rs1, 3'b101, rd, OP_REG);
    endfunction

    // REM rd, rs1, rs2 (funct7=0000001, funct3=110)
    function [31:0] REM;
        input [4:0] rd, rs1, rs2;
        REM = r_type(7'b0000001, rs2, rs1, 3'b110, rd, OP_REG);
    endfunction

    function [31:0] ECALL;
        input dummy;
        ECALL = 32'h00000073;
    endfunction

    // CSRRW rd, csr, rs1: {csr[11:0], rs1[4:0], 001, rd[4:0], 1110011}
    function [31:0] CSRRW;
        input [4:0] rd;
        input [11:0] csr;
        input [4:0] rs1;
        CSRRW = {csr, rs1, 3'b001, rd, OP_SYSTEM};
    endfunction

    // CSRRS rd, csr, rs1: {csr[11:0], rs1[4:0], 010, rd[4:0], 1110011}
    function [31:0] CSRRS;
        input [4:0] rd;
        input [11:0] csr;
        input [4:0] rs1;
        CSRRS = {csr, rs1, 3'b010, rd, OP_SYSTEM};
    endfunction

    // MRET: 0x30200073
    function [31:0] MRET;
        input dummy;
        MRET = 32'h30200073;
    endfunction

    task prog_instr;
        input [13:0] addr;
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
        while (!rv_halted && timeout < 10000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        if (timeout >= 10000)
            $display("  WARNING: halt timeout");
    end
    endtask

    task reset_cpu;
    begin
        rv_enable <= 0;
        @(posedge clk); @(posedge clk);
    end
    endtask

    integer pass_count, fail_count;

    initial begin
        #50000000;
        $display("TIMEOUT");
        $finish;
    end

    initial begin
        clk = 0; rst_n = 0;
        rv_enable = 0;
        imem_we = 0; imem_waddr = 0; imem_wdata = 0;
        mmio_rdata = 0; mmio_ready = 0;
        mmio_write_seen = 0;
        last_mmio_addr = 0; last_mmio_wdata = 0;
        pass_count = 0; fail_count = 0;

        #100;
        rst_n = 1;
        #100;

        // TEST 1: MUL / MULH
        //
        // x1 = 100, x2 = 200
        // x3 = MUL(x1, x2) = 20000 (low 32 bits)
        // x4 = MULH(x1, x2) = 0 (high bits of 100*200)
        //
        // For MULHU: x5 = 0xFFFFFFFF * 0x02 → high word = 0x00000001
        $display("\n=== TEST 1: MUL/MULH ===");
        reset_cpu;

        prog_instr(14'd0, ADDI(5'd1, 5'd0, 12'd100));    // x1 = 100
        prog_instr(14'd1, ADDI(5'd2, 5'd0, 12'd200));    // x2 = 200
        prog_instr(14'd2, MUL(5'd3, 5'd1, 5'd2));        // x3 = MUL(x1, x2) = 20000
        prog_instr(14'd3, MULH(5'd4, 5'd1, 5'd2));       // x4 = MULH(x1, x2) = 0
        // x5 = 0xFFFFFFFF: LUI + ADDI
        prog_instr(14'd4, LUI(5'd5, 20'hFFFFF));          // x5 = 0xFFFFF000
        prog_instr(14'd5, ADDI(5'd5, 5'd5, 12'hFFF));    // x5 = 0xFFFFFFFF
        prog_instr(14'd6, ADDI(5'd6, 5'd0, 12'd2));      // x6 = 2
        prog_instr(14'd7, MULHU(5'd7, 5'd5, 5'd6));      // x7 = MULHU(0xFFFFFFFF, 2) high word = 1
        // Write x3 to MMIO for verification
        prog_instr(14'd8, LUI(5'd8, 20'hFFFF0));          // x8 = 0xFFFF0000
        prog_instr(14'd9, SW(5'd3, 5'd8, 12'd0));         // MMIO[0] = x3
        prog_instr(14'd10, ECALL(0));                      // halt

        mmio_write_seen <= 0;
        rv_enable <= 1;
        wait_halt;

        $display("  x3 (MUL 100*200) = %0d, x4 (MULH) = %0d, x7 (MULHU 0xFFFF_FFFF*2) = %0d",
            dut.regfile[3], dut.regfile[4], dut.regfile[7]);

        if (dut.regfile[3] == 32'd20000 && dut.regfile[4] == 32'd0 && dut.regfile[7] == 32'd1) begin
            $display("  PASSED: MUL/MULH/MULHU correct");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: Expected x3=20000, x4=0, x7=1");
            fail_count = fail_count + 1;
        end

        // TEST 2: DIV/REM + Edge Cases
        //
        // x1 = 100, x2 = 7
        // x3 = DIV(100, 7) = 14
        // x4 = REM(100, 7) = 2
        // x5 = DIV(100, 0) = -1 (0xFFFFFFFF)
        // x6 = REM(100, 0) = 100
        $display("\n=== TEST 2: DIV/REM + Edge Cases ===");
        reset_cpu;

        prog_instr(14'd0, ADDI(5'd1, 5'd0, 12'd100));    // x1 = 100
        prog_instr(14'd1, ADDI(5'd2, 5'd0, 12'd7));      // x2 = 7
        prog_instr(14'd2, DIV(5'd3, 5'd1, 5'd2));         // x3 = 100/7 = 14
        prog_instr(14'd3, REM(5'd4, 5'd1, 5'd2));         // x4 = 100%7 = 2
        // Divide by zero
        prog_instr(14'd4, DIV(5'd5, 5'd1, 5'd0));         // x5 = 100/0 = -1
        prog_instr(14'd5, REM(5'd6, 5'd1, 5'd0));         // x6 = 100%0 = 100
        prog_instr(14'd6, ECALL(0));

        rv_enable <= 1;
        wait_halt;

        $display("  x3 (100/7) = %0d, x4 (100%%7) = %0d", dut.regfile[3], dut.regfile[4]);
        $display("  x5 (100/0) = 0x%08h, x6 (100%%0) = %0d", dut.regfile[5], dut.regfile[6]);

        if (dut.regfile[3] == 32'd14 && dut.regfile[4] == 32'd2 &&
            dut.regfile[5] == 32'hFFFFFFFF && dut.regfile[6] == 32'd100) begin
            $display("  PASSED: DIV/REM + divide-by-zero correct");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED");
            fail_count = fail_count + 1;
        end

        // TEST 3: Timer Interrupt
        //
        // Program: set mtvec=0x100, mtimecmp=10 (low), enable MIE+MTIE.
        // Main program loops. Timer fires, handler writes sentinel to x10.
        //
        // Handler at 0x100: ADDI x10, x0, 42; MRET
        // Main: loop checking x10 until it's 42, then ECALL.
        //
        // CSR addresses:
        //   mstatus = 0x300, mie = 0x304, mtvec = 0x305
        //   mtimecmp = 0x7C0 (custom)
        $display("\n=== TEST 3: Timer Interrupt ===");
        reset_cpu;

        // Handler at word address 64 (byte address 0x100)
        // ADDI x10, x0, 42
        prog_instr(14'd64, ADDI(5'd10, 5'd0, 12'd42));
        // Disable further timer interrupts by clearing MIE in mstatus
        // CSRRW x0, mstatus, x0 (clear mstatus → MIE=0)
        prog_instr(14'd65, CSRRW(5'd0, 12'h300, 5'd0));
        // MRET
        prog_instr(14'd66, MRET(0));

        // Main program at word address 0 (byte address 0x000)
        // Step 1: x1 = handler address = 0x100 = 256
        prog_instr(14'd0, ADDI(5'd1, 5'd0, 12'd256));     // x1 = 256

        // Step 2: CSRRW x0, mtvec, x1 (set mtvec = 256)
        prog_instr(14'd1, CSRRW(5'd0, 12'h305, 5'd1));

        // Step 3: x2 = 10 (low mtimecmp)
        prog_instr(14'd2, ADDI(5'd2, 5'd0, 12'd10));

        // Step 4: CSRRW x0, mtimecmp, x2 (set mtimecmp low = 10)
        prog_instr(14'd3, CSRRW(5'd0, 12'h7C0, 5'd2));

        // Step 5: x3 = 0 (high mtimecmp)
        // CSRRW x0, mtimecmph, x0 (set mtimecmp high = 0)
        prog_instr(14'd4, CSRRW(5'd0, 12'h7C1, 5'd0));

        // Step 6: x4 = 0x88 = MIE(bit3) + MTIE(bit7) → enable in mie
        prog_instr(14'd5, ADDI(5'd4, 5'd0, 12'h80));      // x4 = 0x80 (MTIE)
        prog_instr(14'd6, CSRRW(5'd0, 12'h304, 5'd4));    // mie = 0x80

        // Step 7: x5 = 0x08 (MIE bit in mstatus)
        prog_instr(14'd7, ADDI(5'd5, 5'd0, 12'h08));      // x5 = 8
        prog_instr(14'd8, CSRRW(5'd0, 12'h300, 5'd5));    // mstatus = 8 (MIE=1)

        // Step 8: Loop until x10 != 0 (handler sets x10 = 42)
        // Loop: check x10, branch back if zero
        // BEQ x10, x0, -4 → branch self-loop (offset = 0)
        // We use a simple spin: just NOP a lot of times then check
        // NOP = ADDI x0, x0, 0
        prog_instr(14'd9,  ADDI(5'd0, 5'd0, 12'd0));  // NOP
        prog_instr(14'd10, ADDI(5'd0, 5'd0, 12'd0));   // NOP
        prog_instr(14'd11, ADDI(5'd0, 5'd0, 12'd0));   // NOP
        prog_instr(14'd12, ADDI(5'd0, 5'd0, 12'd0));   // NOP
        prog_instr(14'd13, ADDI(5'd0, 5'd0, 12'd0));   // NOP
        prog_instr(14'd14, ADDI(5'd0, 5'd0, 12'd0));   // NOP
        prog_instr(14'd15, ADDI(5'd0, 5'd0, 12'd0));   // NOP
        prog_instr(14'd16, ADDI(5'd0, 5'd0, 12'd0));   // NOP
        prog_instr(14'd17, ADDI(5'd0, 5'd0, 12'd0));   // NOP
        prog_instr(14'd18, ADDI(5'd0, 5'd0, 12'd0));   // NOP
        // After NOPs, x10 should be 42 from interrupt handler
        prog_instr(14'd19, ECALL(0));                    // halt

        rv_enable <= 1;
        wait_halt;

        $display("  x10 = %0d (expected 42 from interrupt handler)", dut.regfile[10]);
        $display("  mcycle = %0d, mtimecmp = %0d", dut.csr_mcycle, dut.csr_mtimecmp);

        if (dut.regfile[10] == 32'd42) begin
            $display("  PASSED: Timer interrupt fired, handler executed");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: x10 = %0d, expected 42", dut.regfile[10]);
            fail_count = fail_count + 1;
        end

        // TEST 4: 64KB SRAM
        //
        // Program instruction at high address (word 15000 = byte 0xEA60)
        // Execute: LUI to create address, jump there, execute instruction, halt
        $display("\n=== TEST 4: 64KB SRAM ===");
        reset_cpu;

        // Place ADDI x20, x0, 99 at word 15000, then ECALL at 15001
        prog_instr(14'd15000, ADDI(5'd20, 5'd0, 12'd99));
        prog_instr(14'd15001, ECALL(0));

        // At address 0: jump to byte address 15000*4 = 60000 = 0xEA60
        // JAL x0, offset (offset is PC-relative)
        // Byte address 60000 = 0xEA60. From PC=0, offset=0xEA60.
        // JAL format: imm[20|10:1|11|19:12] rd opcode
        // x1 = 0xEA60 → LUI x1, 0x0000F (0xF000) + ADDI x1, x1, 0xA60(-0x5A0 won't work)
        // 0xF000 - 1440 = 0xF000 - 0x5A0 = 0xEA60. But -1440 in 12-bit signed is -1440.
        // 12-bit signed range: -2048..+2047. -1440 = -0x5A0. OK, fits.
        // Hmm wait, LUI sets upper 20 bits: LUI x1, 0x0000F → x1 = 0x0000F000
        // ADDI x1, x1, -0x5A0 → x1 = 0x0000F000 - 0x5A0 = 0x0000EA60
        // JALR x0, x1, 0 → jump to x1
        prog_instr(14'd0, LUI(5'd1, 20'h0000F));          // x1 = 0xF000
        prog_instr(14'd1, ADDI(5'd1, 5'd1, -12'sd1440));  // x1 = 0xEA60
        // JALR x0, x1, 0: {imm[11:0], rs1, 000, rd, 1100111}
        prog_instr(14'd2, i_type(12'd0, 5'd1, 3'b000, 5'd0, 7'b1100111)); // JALR x0, x1, 0

        rv_enable <= 1;
        wait_halt;

        $display("  x20 = %0d (expected 99, from word address 15000)", dut.regfile[20]);

        if (dut.regfile[20] == 32'd99) begin
            $display("  PASSED: 64KB SRAM accessible at high address");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAILED: x20 = %0d, expected 99", dut.regfile[20]);
            fail_count = fail_count + 1;
        end

        $display("\n=== P23D RESULTS: %0d passed, %0d failed out of %0d ===",
            pass_count, fail_count, pass_count + fail_count);
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");
        $finish;
    end

endmodule
