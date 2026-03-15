`timescale 1ns/1ps

module rv32im_cluster #(
    parameter IMEM_DEPTH     = 65536,
    parameter IMEM_ADDR_BITS = 16,
    parameter DMEM_DEPTH     = 65536,
    parameter DMEM_ADDR_BITS = 16
)(
    input  wire        clk,
    input  wire        rst_n,

    input  wire [2:0]  enable,

    input  wire                        imem_we_0,
    input  wire [IMEM_ADDR_BITS-1:0]  imem_waddr_0,
    input  wire [31:0]                imem_wdata_0,

    input  wire                        imem_we_1,
    input  wire [IMEM_ADDR_BITS-1:0]  imem_waddr_1,
    input  wire [31:0]                imem_wdata_1,

    input  wire                        imem_we_2,
    input  wire [IMEM_ADDR_BITS-1:0]  imem_waddr_2,
    input  wire [31:0]                imem_wdata_2,

    output wire                        mmio_valid,
    output wire                        mmio_we,
    output wire [15:0]                mmio_addr,
    output wire [31:0]                mmio_wdata,
    input  wire [31:0]                mmio_rdata,
    input  wire                        mmio_ready,

    output wire [2:0]  halted,
    output wire [31:0] pc_out_0,
    output wire [31:0] pc_out_1,
    output wire [31:0] pc_out_2
);

    wire        c0_mmio_valid, c0_mmio_we;
    wire [15:0] c0_mmio_addr;
    wire [31:0] c0_mmio_wdata;

    rv32i_core #(
        .IMEM_DEPTH(IMEM_DEPTH), .IMEM_ADDR_BITS(IMEM_ADDR_BITS),
        .DMEM_DEPTH(DMEM_DEPTH), .DMEM_ADDR_BITS(DMEM_ADDR_BITS)
    ) core0 (
        .clk(clk), .rst_n(rst_n), .enable(enable[0]),
        .imem_we(imem_we_0), .imem_waddr(imem_waddr_0), .imem_wdata(imem_wdata_0),
        .mmio_valid(c0_mmio_valid), .mmio_we(c0_mmio_we),
        .mmio_addr(c0_mmio_addr), .mmio_wdata(c0_mmio_wdata),
        .mmio_rdata(combined_rdata),
        .mmio_ready(c0_mmio_valid ? combined_ready : 1'b0),
        .halted(halted[0]), .pc_out(pc_out_0),
        .debug_bp_addr_0(32'd0), .debug_bp_addr_1(32'd0),
        .debug_bp_addr_2(32'd0), .debug_bp_addr_3(32'd0),
        .debug_bp_enable(4'd0),
        .debug_resume(1'b0), .debug_halt_req(1'b0), .debug_single_step(1'b0)
    );

    wire        c1_mmio_valid, c1_mmio_we;
    wire [15:0] c1_mmio_addr;
    wire [31:0] c1_mmio_wdata;

    wire c1_grant = c1_mmio_valid && !c0_mmio_valid;

    rv32i_core #(
        .IMEM_DEPTH(IMEM_DEPTH), .IMEM_ADDR_BITS(IMEM_ADDR_BITS),
        .DMEM_DEPTH(DMEM_DEPTH), .DMEM_ADDR_BITS(DMEM_ADDR_BITS)
    ) core1 (
        .clk(clk), .rst_n(rst_n), .enable(enable[1]),
        .imem_we(imem_we_1), .imem_waddr(imem_waddr_1), .imem_wdata(imem_wdata_1),
        .mmio_valid(c1_mmio_valid), .mmio_we(c1_mmio_we),
        .mmio_addr(c1_mmio_addr), .mmio_wdata(c1_mmio_wdata),
        .mmio_rdata(combined_rdata),
        .mmio_ready(c1_grant ? combined_ready : 1'b0),
        .halted(halted[1]), .pc_out(pc_out_1),
        .debug_bp_addr_0(32'd0), .debug_bp_addr_1(32'd0),
        .debug_bp_addr_2(32'd0), .debug_bp_addr_3(32'd0),
        .debug_bp_enable(4'd0),
        .debug_resume(1'b0), .debug_halt_req(1'b0), .debug_single_step(1'b0)
    );

    wire        c2_mmio_valid, c2_mmio_we;
    wire [15:0] c2_mmio_addr;
    wire [31:0] c2_mmio_wdata;

    wire c2_grant = c2_mmio_valid && !c0_mmio_valid && !c1_mmio_valid;

    rv32i_core #(
        .IMEM_DEPTH(IMEM_DEPTH), .IMEM_ADDR_BITS(IMEM_ADDR_BITS),
        .DMEM_DEPTH(DMEM_DEPTH), .DMEM_ADDR_BITS(DMEM_ADDR_BITS)
    ) core2 (
        .clk(clk), .rst_n(rst_n), .enable(enable[2]),
        .imem_we(imem_we_2), .imem_waddr(imem_waddr_2), .imem_wdata(imem_wdata_2),
        .mmio_valid(c2_mmio_valid), .mmio_we(c2_mmio_we),
        .mmio_addr(c2_mmio_addr), .mmio_wdata(c2_mmio_wdata),
        .mmio_rdata(combined_rdata),
        .mmio_ready(c2_grant ? combined_ready : 1'b0),
        .halted(halted[2]), .pc_out(pc_out_2),
        .debug_bp_addr_0(32'd0), .debug_bp_addr_1(32'd0),
        .debug_bp_addr_2(32'd0), .debug_bp_addr_3(32'd0),
        .debug_bp_enable(4'd0),
        .debug_resume(1'b0), .debug_halt_req(1'b0), .debug_single_step(1'b0)
    );

    reg [31:0] mailbox [0:3];

    integer mbi;

    wire        arb_valid = c0_mmio_valid | c1_mmio_valid | c2_mmio_valid;
    wire [15:0] arb_addr  = c0_mmio_valid ? c0_mmio_addr :
                            c1_mmio_valid ? c1_mmio_addr :
                                            c2_mmio_addr;
    wire        arb_we    = c0_mmio_valid ? c0_mmio_we :
                            c1_mmio_valid ? c1_mmio_we :
                                            c2_mmio_we;
    wire [31:0] arb_wdata = c0_mmio_valid ? c0_mmio_wdata :
                            c1_mmio_valid ? c1_mmio_wdata :
                                            c2_mmio_wdata;

    wire is_mailbox = arb_valid && (arb_addr >= 16'h0080) && (arb_addr <= 16'h008C);
    wire [1:0] mailbox_idx = arb_addr[3:2];

    reg [31:0] mailbox_rdata;
    always @(*) begin
        mailbox_rdata = mailbox[mailbox_idx];
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (mbi = 0; mbi < 4; mbi = mbi + 1)
                mailbox[mbi] <= 32'd0;
        end else if (is_mailbox && arb_we) begin
            mailbox[mailbox_idx] <= arb_wdata;
        end
    end

    wire mailbox_ready = is_mailbox;

    assign mmio_valid = arb_valid && !is_mailbox;

    assign mmio_we = arb_we;

    assign mmio_addr = arb_addr;

    assign mmio_wdata = arb_wdata;

    wire [31:0] combined_rdata = is_mailbox ? mailbox_rdata : mmio_rdata;
    wire        combined_ready = is_mailbox ? mailbox_ready : mmio_ready;

endmodule
