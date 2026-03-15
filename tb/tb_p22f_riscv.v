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

    always @(posedge clk) begin
        mmio_ready <= mmio_valid;
    end

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

    function [31:0] r_type;
        input [6:0] funct7;
        input [4:0] rs2, rs1;
        input [2:0] funct3;
        input [4:0] rd;
        input [6:0] opcode;
        r_type = {funct7, rs2, rs1, funct3, rd, opcode};
    endfunction

    function [31:0] i_type;
        input [11:0] imm;
        input [4:0]  rs1;
        input [2:0]  funct3;
        input [4:0]  rd;
        input [6:0]  opcode;
        i_type = {imm, rs1, funct3, rd, opcode};
    endfunction

    function [31:0] s_type;
        input [11:0] imm;
        input [4:0]  rs2, rs1;
        input [2:0]  funct3;
        input [6:0]  opcode;
        s_type = {imm[11:5], rs2, rs1, funct3, imm[4:0], opcode};
    endfunction

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

    localparam F3_ADD  = 3'b000;
    localparam F3_SLL  = 3'b001;
    localparam F3_SLT  = 3'b010;
    localparam F3_SLTU = 3'b011;
    localparam F3_XOR  = 3'b100;
    localparam F3_SRL  = 3'b101;
    localparam F3_OR   = 3'b110;
    localparam F3_AND  = 3'b111;

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

        $display("\nTest 1: ALU operations");

        prog_instr(12'd0, ADDI(5'd1, 5'd0, 12'd100));
        prog_instr(12'd1, ADDI(5'd2, 5'd0, 12'd200));
        prog_instr(12'd2, ADD(5'd3, 5'd1, 5'd2));
        prog_instr(12'd3, SUB(5'd4, 5'd2, 5'd1));
        prog_instr(12'd4, AND_R(5'd5, 5'd1, 5'd2));
        prog_instr(12'd5, OR_R(5'd6, 5'd1, 5'd2));
        prog_instr(12'd6, SLLI(5'd7, 5'd1, 5'd2));
        prog_instr(12'd7, SRLI(5'd8, 5'd2, 5'd3));
        prog_instr(12'd8, ECALL(0));

        rv_enable = 1;
        wait_halt;

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

        rv_enable = 0;
        #50;

        $display("\nTest 2: Memory load/store");

        prog_instr(12'd0, ADDI(5'd1, 5'd0, 12'h234));
        prog_instr(12'd1, SW(5'd1, 5'd0, 12'd0));
        prog_instr(12'd2, LW(5'd2, 5'd0, 12'd0));
        prog_instr(12'd3, ADDI(5'd3, 5'd0, 12'hBCD));
        prog_instr(12'd4, SW(5'd3, 5'd0, 12'd4));
        prog_instr(12'd5, LW(5'd4, 5'd0, 12'd4));
        prog_instr(12'd6, ECALL(0));

        rv_enable = 1;
        wait_halt;

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

        $display("\nTest 3: MMIO spike inject");

        prog_instr(12'd0, u_type(20'hFFFF0, 5'd10, OP_LUI));
        prog_instr(12'd1, ADDI(5'd11, 5'd0, 12'd42));
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

        $display("\nTest 4: MMIO UART TX write");

        prog_instr(12'd0, u_type(20'hFFFF0, 5'd10, OP_LUI));
        prog_instr(12'd1, ADDI(5'd11, 5'd0, 12'h055));
        prog_instr(12'd2, SW(5'd11, 5'd10, 12'h020));
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

        $display("P22F RESULTS: %0d/%0d passed", pass_count, pass_count + fail_count);

        if (fail_count > 0)
            $display("SOME TESTS FAILED");

        $finish;
    end

endmodule
