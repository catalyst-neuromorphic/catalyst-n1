`timescale 1ns/1ps

module rv32i_core #(
    parameter IMEM_DEPTH     = 65536,
    parameter IMEM_ADDR_BITS = 16,
    parameter DMEM_DEPTH     = 65536,
    parameter DMEM_ADDR_BITS = 16
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        enable,
    input  wire        imem_we,
    input  wire [IMEM_ADDR_BITS-1:0] imem_waddr,
    input  wire [31:0] imem_wdata,
    output reg         mmio_valid,
    output reg         mmio_we,
    output reg  [15:0] mmio_addr,
    output reg  [31:0] mmio_wdata,
    input  wire [31:0] mmio_rdata,
    input  wire        mmio_ready,
    output wire        halted,
    output wire [31:0] pc_out,
    input  wire [31:0] debug_bp_addr_0,
    input  wire [31:0] debug_bp_addr_1,
    input  wire [31:0] debug_bp_addr_2,
    input  wire [31:0] debug_bp_addr_3,
    input  wire [3:0]  debug_bp_enable,
    input  wire        debug_resume,
    input  wire        debug_halt_req,
    input  wire        debug_single_step
);

    reg [31:0] regfile [0:31];

    reg [31:0] fregfile [0:31];

    reg [31:0] imem [0:IMEM_DEPTH-1];

    always @(posedge clk) begin
        if (imem_we)
            imem[imem_waddr] <= imem_wdata;
    end

    reg [31:0] dmem [0:DMEM_DEPTH-1];

    reg [31:0] pc;
    reg [31:0] instr;
    reg        fetch_valid;
    reg        halt_r;

    assign pc_out = pc;
    assign halted = halt_r;

    wire [IMEM_ADDR_BITS-1:0] pc_word = pc[IMEM_ADDR_BITS+1:2];
    wire [31:0] fetched_instr = imem[pc_word];

    wire [6:0]  opcode = instr[6:0];
    wire [4:0]  rd     = instr[11:7];
    wire [2:0]  funct3 = instr[14:12];
    wire [4:0]  rs1    = instr[19:15];
    wire [4:0]  rs2    = instr[24:20];
    wire [6:0]  funct7 = instr[31:25];

    wire [31:0] imm_i = {{20{instr[31]}}, instr[31:20]};
    wire [31:0] imm_s = {{20{instr[31]}}, instr[31:25], instr[11:7]};
    wire [31:0] imm_b = {{19{instr[31]}}, instr[31], instr[7], instr[30:25], instr[11:8], 1'b0};
    wire [31:0] imm_u = {instr[31:12], 12'b0};
    wire [31:0] imm_j = {{11{instr[31]}}, instr[31], instr[19:12], instr[20], instr[30:21], 1'b0};

    wire [31:0] rs1_val = (rs1 == 5'd0) ? 32'd0 : regfile[rs1];
    wire [31:0] rs2_val = (rs2 == 5'd0) ? 32'd0 : regfile[rs2];

    localparam OP_LUI    = 7'b0110111;
    localparam OP_AUIPC  = 7'b0010111;
    localparam OP_JAL    = 7'b1101111;
    localparam OP_JALR   = 7'b1100111;
    localparam OP_BRANCH = 7'b1100011;
    localparam OP_LOAD   = 7'b0000011;
    localparam OP_STORE  = 7'b0100011;
    localparam OP_IMM    = 7'b0010011;
    localparam OP_REG    = 7'b0110011;
    localparam OP_FENCE  = 7'b0001111;
    localparam OP_SYSTEM = 7'b1110011;

    localparam OP_FLW    = 7'b0000111;
    localparam OP_FSW    = 7'b0100111;
    localparam OP_FP     = 7'b1010011;

    function real f32_to_real;
        input [31:0] f;
        reg [63:0] d;
        begin
            if (f[30:0] == 31'd0) begin
                d = {f[31], 63'd0};
            end else if (f[30:23] == 8'hFF) begin
                d = {f[31], 11'h7FF, f[22:0], 29'd0};
            end else begin
                d[63]    = f[31];
                d[62:52] = {3'd0, f[30:23]} + 11'd896;
                d[51:0]  = {f[22:0], 29'd0};
            end
            f32_to_real = $bitstoreal(d);
        end
    endfunction

    function [31:0] real_to_f32;
        input real r;
        reg [63:0] d;
        reg [10:0] dexp;
        reg [7:0]  fexp;
        begin
            d = $realtobits(r);
            if (d[62:0] == 63'd0) begin
                real_to_f32 = {d[63], 31'd0};
            end else begin
                dexp = d[62:52];
                if (dexp >= 11'd1151) begin
                    real_to_f32 = {d[63], 8'hFF, 23'd0};
                end else if (dexp <= 11'd896) begin
                    real_to_f32 = {d[63], 31'd0};
                end else begin
                    fexp = dexp - 11'd896;
                    real_to_f32 = {d[63], fexp, d[51:29]};
                end
            end
        end
    endfunction

    function real fp_sqrt;
        input real x;
        real guess;
        integer i;
        begin
            if (x <= 0.0) begin
                fp_sqrt = 0.0;
            end else begin
                guess = x;
                for (i = 0; i < 25; i = i + 1)
                    guess = (guess + x / guess) / 2.0;
                fp_sqrt = guess;
            end
        end
    endfunction

    wire is_muldiv = (opcode == OP_REG) && (funct7 == 7'b0000001);

    wire signed [63:0] mul_ss = $signed(rs1_val) * $signed(rs2_val);
    wire        [63:0] mul_uu = rs1_val * rs2_val;
    wire signed [63:0] mul_su = $signed(rs1_val) * $signed({1'b0, rs2_val});

    wire signed [31:0] div_s = (rs2_val == 0) ? -32'sd1 :
                               (rs1_val == 32'h80000000 && rs2_val == 32'hFFFFFFFF) ? 32'h80000000 :
                               $signed(rs1_val) / $signed(rs2_val);
    wire        [31:0] div_u = (rs2_val == 0) ? 32'hFFFFFFFF : rs1_val / rs2_val;
    wire signed [31:0] rem_s = (rs2_val == 0) ? $signed(rs1_val) :
                               (rs1_val == 32'h80000000 && rs2_val == 32'hFFFFFFFF) ? 32'sd0 :
                               $signed(rs1_val) % $signed(rs2_val);
    wire        [31:0] rem_u = (rs2_val == 0) ? rs1_val : rs1_val % rs2_val;

    reg [31:0] muldiv_result;
    always @(*) begin
        case (funct3)
            3'b000: muldiv_result = mul_ss[31:0];
            3'b001: muldiv_result = mul_ss[63:32];
            3'b010: muldiv_result = mul_su[63:32];
            3'b011: muldiv_result = mul_uu[63:32];
            3'b100: muldiv_result = div_s;
            3'b101: muldiv_result = div_u;
            3'b110: muldiv_result = rem_s;
            3'b111: muldiv_result = rem_u;
        endcase
    end

    reg [31:0] csr_mtvec;
    reg [31:0] csr_mepc;
    reg [31:0] csr_mcause;
    reg [31:0] csr_mstatus;
    reg [31:0] csr_mie;
    reg [31:0] csr_mip;
    reg [63:0] csr_mcycle;
    reg [63:0] csr_mtimecmp;

    localparam CSR_MSTATUS  = 12'h300;
    localparam CSR_MIE      = 12'h304;
    localparam CSR_MTVEC    = 12'h305;
    localparam CSR_MEPC     = 12'h341;
    localparam CSR_MCAUSE   = 12'h342;
    localparam CSR_MIP      = 12'h344;
    localparam CSR_MCYCLE   = 12'hB00;
    localparam CSR_MCYCLEH  = 12'hB80;
    localparam CSR_MTIMECMP  = 12'h7C0;
    localparam CSR_MTIMECMPH = 12'h7C1;

    wire [11:0] csr_addr = instr[31:20];
    wire [4:0]  csr_zimm = rs1;

    reg [31:0] csr_rdata;
    always @(*) begin
        case (csr_addr)
            CSR_MSTATUS:  csr_rdata = csr_mstatus;
            CSR_MIE:      csr_rdata = csr_mie;
            CSR_MTVEC:    csr_rdata = csr_mtvec;
            CSR_MEPC:     csr_rdata = csr_mepc;
            CSR_MCAUSE:   csr_rdata = csr_mcause;
            CSR_MIP:      csr_rdata = csr_mip;
            CSR_MCYCLE:   csr_rdata = csr_mcycle[31:0];
            CSR_MCYCLEH:  csr_rdata = csr_mcycle[63:32];
            CSR_MTIMECMP: csr_rdata = csr_mtimecmp[31:0];
            CSR_MTIMECMPH:csr_rdata = csr_mtimecmp[63:32];
            default:      csr_rdata = 32'd0;
        endcase
    end

    wire timer_pending = (csr_mcycle >= csr_mtimecmp);

    wire timer_irq = timer_pending && csr_mstatus[3] && csr_mie[7];

    wire [31:0] alu_b = (opcode == OP_REG) ? rs2_val : imm_i;
    wire [4:0]  shamt = alu_b[4:0];

    reg [31:0] alu_result;
    always @(*) begin
        case (funct3)
            3'b000: alu_result = (opcode == OP_REG && funct7[5]) ?
                                 (rs1_val - rs2_val) : (rs1_val + alu_b);
            3'b001: alu_result = rs1_val << shamt;
            3'b010: alu_result = ($signed(rs1_val) < $signed(alu_b)) ? 32'd1 : 32'd0;
            3'b011: alu_result = (rs1_val < alu_b) ? 32'd1 : 32'd0;
            3'b100: alu_result = rs1_val ^ alu_b;
            3'b101: alu_result = funct7[5] ? ($signed(rs1_val) >>> shamt) :
                                             (rs1_val >> shamt);
            3'b110: alu_result = rs1_val | alu_b;
            3'b111: alu_result = rs1_val & alu_b;
            default: alu_result = 32'd0;
        endcase
    end

    reg branch_taken;
    always @(*) begin
        case (funct3)
            3'b000: branch_taken = (rs1_val == rs2_val);
            3'b001: branch_taken = (rs1_val != rs2_val);
            3'b100: branch_taken = ($signed(rs1_val) < $signed(rs2_val));
            3'b101: branch_taken = ($signed(rs1_val) >= $signed(rs2_val));
            3'b110: branch_taken = (rs1_val < rs2_val);
            3'b111: branch_taken = (rs1_val >= rs2_val);
            default: branch_taken = 1'b0;
        endcase
    end

    wire [31:0] mem_addr = rs1_val + ((opcode == OP_STORE) ? imm_s : imm_i);
    wire        is_mmio  = (mem_addr[31:16] == 16'hFFFF);
    wire [DMEM_ADDR_BITS-1:0] dmem_word_addr = mem_addr[DMEM_ADDR_BITS+1:2];

    localparam S_FETCH      = 4'd0;
    localparam S_EXEC       = 4'd1;
    localparam S_MEM_RD     = 4'd2;
    localparam S_MEM_WR     = 4'd3;
    localparam S_HALT       = 4'd4;
    localparam S_TRAP       = 4'd5;
    localparam S_DEBUG_HALT = 4'd6;

    reg [3:0] state;

    reg debug_single_step_pending;

    wire bp_match = (debug_bp_enable[0] && (pc == debug_bp_addr_0)) ||
                    (debug_bp_enable[1] && (pc == debug_bp_addr_1)) ||
                    (debug_bp_enable[2] && (pc == debug_bp_addr_2)) ||
                    (debug_bp_enable[3] && (pc == debug_bp_addr_3));

    real fp_op_a, fp_op_b, fp_op_r;
    reg  mem_rd_is_float;

    integer ri;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pc          <= 32'd0;
            instr       <= 32'd0;
            fetch_valid <= 1'b0;
            halt_r      <= 1'b0;
            state       <= S_FETCH;
            mmio_valid  <= 1'b0;
            mmio_we     <= 1'b0;
            mmio_addr   <= 16'd0;
            mmio_wdata  <= 32'd0;

            csr_mtvec    <= 32'd0;
            csr_mepc     <= 32'd0;
            csr_mcause   <= 32'd0;
            csr_mstatus  <= 32'd0;
            csr_mie      <= 32'd0;
            csr_mip      <= 32'd0;
            csr_mcycle   <= 64'd0;
            csr_mtimecmp <= 64'hFFFFFFFF_FFFFFFFF;
            mem_rd_is_float <= 1'b0;
            debug_single_step_pending <= 1'b0;
            for (ri = 0; ri < 32; ri = ri + 1) begin
                regfile[ri]  <= 32'd0;
                fregfile[ri] <= 32'd0;
            end
        end else if (!enable) begin
            state <= S_FETCH;
            pc    <= 32'd0;
            halt_r <= 1'b0;
            mmio_valid <= 1'b0;
            mem_rd_is_float <= 1'b0;
            csr_mcycle <= 64'd0;
            debug_single_step_pending <= 1'b0;
        end else begin

            csr_mcycle <= csr_mcycle + 64'd1;

            csr_mip[7] <= timer_pending;

            case (state)
                S_FETCH: begin

                    if (debug_halt_req) begin
                        halt_r <= 1'b1;
                        state  <= S_DEBUG_HALT;
                    end

                    else if (bp_match) begin
                        halt_r <= 1'b1;
                        state  <= S_DEBUG_HALT;
                    end

                    else if (debug_single_step_pending) begin
                        debug_single_step_pending <= 1'b0;
                        halt_r <= 1'b1;
                        state  <= S_DEBUG_HALT;
                    end

                    else if (timer_irq) begin
                        csr_mepc    <= pc;
                        csr_mcause  <= 32'h80000007;
                        csr_mstatus[3] <= 1'b0;
                        csr_mstatus[7] <= csr_mstatus[3];
                        pc          <= csr_mtvec & ~32'd3;
                        state       <= S_FETCH;
                    end else begin
                        instr       <= fetched_instr;
                        fetch_valid <= 1'b1;
                        state       <= S_EXEC;
                    end
                end

                S_EXEC: begin
                    mmio_valid <= 1'b0;

                    case (opcode)
                        OP_LUI: begin
                            if (rd != 0) regfile[rd] <= imm_u;
                            pc <= pc + 4;
                            state <= S_FETCH;
                        end

                        OP_AUIPC: begin
                            if (rd != 0) regfile[rd] <= pc + imm_u;
                            pc <= pc + 4;
                            state <= S_FETCH;
                        end

                        OP_JAL: begin
                            if (rd != 0) regfile[rd] <= pc + 4;
                            pc <= pc + imm_j;
                            state <= S_FETCH;
                        end

                        OP_JALR: begin
                            if (rd != 0) regfile[rd] <= pc + 4;
                            pc <= (rs1_val + imm_i) & ~32'd1;
                            state <= S_FETCH;
                        end

                        OP_BRANCH: begin
                            pc <= branch_taken ? (pc + imm_b) : (pc + 4);
                            state <= S_FETCH;
                        end

                        OP_LOAD: begin
                            if (is_mmio) begin
                                mmio_valid <= 1'b1;
                                mmio_we    <= 1'b0;
                                mmio_addr  <= mem_addr[15:0];
                                mem_rd_is_float <= 1'b0;
                                state      <= S_MEM_RD;
                            end else begin

                                if (rd != 0) begin
                                    case (funct3)
                                        3'b000: begin
                                            case (mem_addr[1:0])
                                                2'd0: regfile[rd] <= {{24{dmem[dmem_word_addr][7]}},  dmem[dmem_word_addr][7:0]};
                                                2'd1: regfile[rd] <= {{24{dmem[dmem_word_addr][15]}}, dmem[dmem_word_addr][15:8]};
                                                2'd2: regfile[rd] <= {{24{dmem[dmem_word_addr][23]}}, dmem[dmem_word_addr][23:16]};
                                                2'd3: regfile[rd] <= {{24{dmem[dmem_word_addr][31]}}, dmem[dmem_word_addr][31:24]};
                                            endcase
                                        end
                                        3'b001: begin
                                            if (mem_addr[1])
                                                regfile[rd] <= {{16{dmem[dmem_word_addr][31]}}, dmem[dmem_word_addr][31:16]};
                                            else
                                                regfile[rd] <= {{16{dmem[dmem_word_addr][15]}}, dmem[dmem_word_addr][15:0]};
                                        end
                                        3'b010: regfile[rd] <= dmem[dmem_word_addr];
                                        3'b100: begin
                                            case (mem_addr[1:0])
                                                2'd0: regfile[rd] <= {24'd0, dmem[dmem_word_addr][7:0]};
                                                2'd1: regfile[rd] <= {24'd0, dmem[dmem_word_addr][15:8]};
                                                2'd2: regfile[rd] <= {24'd0, dmem[dmem_word_addr][23:16]};
                                                2'd3: regfile[rd] <= {24'd0, dmem[dmem_word_addr][31:24]};
                                            endcase
                                        end
                                        3'b101: begin
                                            if (mem_addr[1])
                                                regfile[rd] <= {16'd0, dmem[dmem_word_addr][31:16]};
                                            else
                                                regfile[rd] <= {16'd0, dmem[dmem_word_addr][15:0]};
                                        end
                                        default: ;
                                    endcase
                                end
                                pc    <= pc + 4;
                                state <= S_FETCH;
                            end
                        end

                        OP_STORE: begin
                            if (is_mmio) begin
                                mmio_valid <= 1'b1;
                                mmio_we    <= 1'b1;
                                mmio_addr  <= mem_addr[15:0];
                                mmio_wdata <= rs2_val;
                                state      <= S_MEM_WR;
                            end else begin
                                case (funct3)
                                    3'b000: begin
                                        case (mem_addr[1:0])
                                            2'd0: dmem[dmem_word_addr][7:0]   <= rs2_val[7:0];
                                            2'd1: dmem[dmem_word_addr][15:8]  <= rs2_val[7:0];
                                            2'd2: dmem[dmem_word_addr][23:16] <= rs2_val[7:0];
                                            2'd3: dmem[dmem_word_addr][31:24] <= rs2_val[7:0];
                                        endcase
                                    end
                                    3'b001: begin
                                        if (mem_addr[1])
                                            dmem[dmem_word_addr][31:16] <= rs2_val[15:0];
                                        else
                                            dmem[dmem_word_addr][15:0]  <= rs2_val[15:0];
                                    end
                                    3'b010: dmem[dmem_word_addr] <= rs2_val;
                                    default: ;
                                endcase
                                pc    <= pc + 4;
                                state <= S_FETCH;
                            end
                        end

                        OP_IMM: begin
                            if (rd != 0) regfile[rd] <= alu_result;
                            pc    <= pc + 4;
                            state <= S_FETCH;
                        end

                        OP_REG: begin

                            if (is_muldiv) begin
                                if (rd != 0) regfile[rd] <= muldiv_result;
                            end else begin
                                if (rd != 0) regfile[rd] <= alu_result;
                            end
                            pc    <= pc + 4;
                            state <= S_FETCH;
                        end

                        OP_FENCE: begin

                            pc    <= pc + 4;
                            state <= S_FETCH;
                        end

                        OP_SYSTEM: begin
                            if (funct3 == 3'b000) begin

                                if (instr[31:20] == 12'h302) begin

                                    pc <= csr_mepc;
                                    csr_mstatus[3] <= csr_mstatus[7];
                                    csr_mstatus[7] <= 1'b1;
                                    state <= S_FETCH;
                                end else begin

                                    halt_r <= 1'b1;
                                    state  <= S_HALT;
                                end
                            end else begin

                                if (rd != 0) regfile[rd] <= csr_rdata;

                                case (funct3)
                                    3'b001: begin
                                        case (csr_addr)
                                            CSR_MSTATUS:  csr_mstatus  <= rs1_val;
                                            CSR_MIE:      csr_mie      <= rs1_val;
                                            CSR_MTVEC:    csr_mtvec    <= rs1_val;
                                            CSR_MEPC:     csr_mepc     <= rs1_val;
                                            CSR_MCAUSE:   csr_mcause   <= rs1_val;
                                            CSR_MTIMECMP: csr_mtimecmp[31:0]  <= rs1_val;
                                            CSR_MTIMECMPH:csr_mtimecmp[63:32] <= rs1_val;
                                            default: ;
                                        endcase
                                    end
                                    3'b010: begin
                                        if (rs1 != 0) begin
                                            case (csr_addr)
                                                CSR_MSTATUS:  csr_mstatus  <= csr_mstatus  | rs1_val;
                                                CSR_MIE:      csr_mie      <= csr_mie      | rs1_val;
                                                CSR_MTVEC:    csr_mtvec    <= csr_mtvec    | rs1_val;
                                                default: ;
                                            endcase
                                        end
                                    end
                                    3'b011: begin
                                        if (rs1 != 0) begin
                                            case (csr_addr)
                                                CSR_MSTATUS:  csr_mstatus  <= csr_mstatus  & ~rs1_val;
                                                CSR_MIE:      csr_mie      <= csr_mie      & ~rs1_val;
                                                default: ;
                                            endcase
                                        end
                                    end
                                    3'b101: begin
                                        case (csr_addr)
                                            CSR_MSTATUS:  csr_mstatus  <= {27'd0, csr_zimm};
                                            CSR_MIE:      csr_mie      <= {27'd0, csr_zimm};
                                            CSR_MTVEC:    csr_mtvec    <= {27'd0, csr_zimm};
                                            default: ;
                                        endcase
                                    end
                                    3'b110: begin
                                        if (csr_zimm != 0) begin
                                            case (csr_addr)
                                                CSR_MSTATUS:  csr_mstatus <= csr_mstatus | {27'd0, csr_zimm};
                                                CSR_MIE:      csr_mie     <= csr_mie     | {27'd0, csr_zimm};
                                                default: ;
                                            endcase
                                        end
                                    end
                                    3'b111: begin
                                        if (csr_zimm != 0) begin
                                            case (csr_addr)
                                                CSR_MSTATUS:  csr_mstatus <= csr_mstatus & ~{27'd0, csr_zimm};
                                                CSR_MIE:      csr_mie     <= csr_mie     & ~{27'd0, csr_zimm};
                                                default: ;
                                            endcase
                                        end
                                    end
                                    default: ;
                                endcase

                                pc    <= pc + 4;
                                state <= S_FETCH;
                            end
                        end

                        OP_FLW: begin
                            if (is_mmio) begin
                                mmio_valid <= 1'b1;
                                mmio_we    <= 1'b0;
                                mmio_addr  <= mem_addr[15:0];
                                mem_rd_is_float <= 1'b1;
                                state      <= S_MEM_RD;
                            end else begin
                                fregfile[rd] <= dmem[dmem_word_addr];
                                pc    <= pc + 4;
                                state <= S_FETCH;
                            end
                        end

                        OP_FSW: begin
                            if (is_mmio) begin
                                mmio_valid <= 1'b1;
                                mmio_we    <= 1'b1;
                                mmio_addr  <= mem_addr[15:0];
                                mmio_wdata <= fregfile[rs2];
                                state      <= S_MEM_WR;
                            end else begin
                                dmem[dmem_word_addr] <= fregfile[rs2];
                                pc    <= pc + 4;
                                state <= S_FETCH;
                            end
                        end

                        OP_FP: begin
                            case (funct7)
                                7'b0000000: begin
                                    fp_op_a = f32_to_real(fregfile[rs1]);
                                    fp_op_b = f32_to_real(fregfile[rs2]);
                                    fregfile[rd] <= real_to_f32(fp_op_a + fp_op_b);
                                end
                                7'b0000100: begin
                                    fp_op_a = f32_to_real(fregfile[rs1]);
                                    fp_op_b = f32_to_real(fregfile[rs2]);
                                    fregfile[rd] <= real_to_f32(fp_op_a - fp_op_b);
                                end
                                7'b0001000: begin
                                    fp_op_a = f32_to_real(fregfile[rs1]);
                                    fp_op_b = f32_to_real(fregfile[rs2]);
                                    fregfile[rd] <= real_to_f32(fp_op_a * fp_op_b);
                                end
                                7'b0001100: begin
                                    fp_op_a = f32_to_real(fregfile[rs1]);
                                    fp_op_b = f32_to_real(fregfile[rs2]);
                                    if (fp_op_b != 0.0)
                                        fregfile[rd] <= real_to_f32(fp_op_a / fp_op_b);
                                    else
                                        fregfile[rd] <= 32'h7FC00000;
                                end
                                7'b0101100: begin
                                    fp_op_a = f32_to_real(fregfile[rs1]);
                                    fp_op_r = fp_sqrt(fp_op_a);
                                    fregfile[rd] <= real_to_f32(fp_op_r);
                                end
                                7'b0010100: begin
                                    fp_op_a = f32_to_real(fregfile[rs1]);
                                    fp_op_b = f32_to_real(fregfile[rs2]);
                                    case (funct3)
                                        3'b000: fregfile[rd] <= (fp_op_a <= fp_op_b) ?
                                                    fregfile[rs1] : fregfile[rs2];
                                        3'b001: fregfile[rd] <= (fp_op_a >= fp_op_b) ?
                                                    fregfile[rs1] : fregfile[rs2];
                                        default: ;
                                    endcase
                                end
                                7'b0010000: begin
                                    case (funct3)
                                        3'b000: fregfile[rd] <= {fregfile[rs2][31],
                                                    fregfile[rs1][30:0]};
                                        3'b001: fregfile[rd] <= {~fregfile[rs2][31],
                                                    fregfile[rs1][30:0]};
                                        3'b010: fregfile[rd] <= {fregfile[rs1][31] ^
                                                    fregfile[rs2][31],
                                                    fregfile[rs1][30:0]};
                                        default: ;
                                    endcase
                                end
                                7'b1100000: begin
                                    fp_op_a = f32_to_real(fregfile[rs1]);
                                    if (rd != 0) regfile[rd] <= $rtoi(fp_op_a);
                                end
                                7'b1101000: begin
                                    fregfile[rd] <= real_to_f32($itor($signed(rs1_val)));
                                end
                                7'b1010000: begin
                                    fp_op_a = f32_to_real(fregfile[rs1]);
                                    fp_op_b = f32_to_real(fregfile[rs2]);
                                    if (rd != 0) begin
                                        case (funct3)
                                            3'b010: regfile[rd] <= (fp_op_a == fp_op_b) ?
                                                        32'd1 : 32'd0;
                                            3'b001: regfile[rd] <= (fp_op_a < fp_op_b) ?
                                                        32'd1 : 32'd0;
                                            3'b000: regfile[rd] <= (fp_op_a <= fp_op_b) ?
                                                        32'd1 : 32'd0;
                                            default: ;
                                        endcase
                                    end
                                end
                                7'b1110000: begin
                                    if (rd != 0) regfile[rd] <= fregfile[rs1];
                                end
                                7'b1111000: begin
                                    fregfile[rd] <= rs1_val;
                                end
                                default: ;
                            endcase
                            pc    <= pc + 4;
                            state <= S_FETCH;
                        end

                        default: begin
                            halt_r <= 1'b1;
                            state  <= S_HALT;
                        end
                    endcase
                end

                S_MEM_RD: begin
                    if (mmio_ready) begin
                        mmio_valid <= 1'b0;
                        if (mem_rd_is_float) begin
                            fregfile[rd] <= mmio_rdata;
                            mem_rd_is_float <= 1'b0;
                        end else begin
                            if (rd != 0) regfile[rd] <= mmio_rdata;
                        end
                        pc    <= pc + 4;
                        state <= S_FETCH;
                    end
                end

                S_MEM_WR: begin
                    if (mmio_ready) begin
                        mmio_valid <= 1'b0;
                        pc    <= pc + 4;
                        state <= S_FETCH;
                    end
                end

                S_HALT: begin
                end

                S_DEBUG_HALT: begin
                    if (debug_resume) begin
                        halt_r <= 1'b0;
                        state  <= S_FETCH;
                    end else if (debug_single_step) begin
                        halt_r <= 1'b0;
                        debug_single_step_pending <= 1'b1;
                        state  <= S_FETCH;
                    end
                end

                default: state <= S_HALT;
            endcase
        end
    end

endmodule
