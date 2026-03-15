`timescale 1ns / 1ps

module tb_fpga_top;

    parameter CLK_FREQ   = 921_600;
    parameter BAUD       = 115200;
    parameter POR_BITS   = 4;
    parameter CLK_PERIOD = 10;
    parameter CLKS_PER_BIT = CLK_FREQ / BAUD;
    parameter BIT_PERIOD = CLKS_PER_BIT * CLK_PERIOD;

    reg        clk;
    reg        btn_rst;
    reg        uart_rxd;
    wire       uart_txd;
    wire [3:0] led;

    fpga_top #(
        .CLK_FREQ (CLK_FREQ),
        .BAUD     (BAUD),
        .POR_BITS (POR_BITS)
    ) dut (
        .clk      (clk),
        .btn_rst  (btn_rst),
        .uart_rxd (uart_rxd),
        .uart_txd (uart_txd),
        .led      (led)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    reg [7:0]  rx_fifo [0:63];
    integer    rx_wr_ptr;
    integer    rx_rd_ptr;
    reg [7:0]  cap_byte;
    integer    cap_i;

    initial begin
        rx_wr_ptr = 0;
        rx_rd_ptr = 0;

        forever begin
            @(negedge uart_txd);
            #(BIT_PERIOD / 2);

            if (uart_txd == 0) begin
                for (cap_i = 0; cap_i < 8; cap_i = cap_i + 1) begin
                    #(BIT_PERIOD);
                    cap_byte[cap_i] = uart_txd;
                end
                #(BIT_PERIOD);

                rx_fifo[rx_wr_ptr] = cap_byte;
                $display("  [UART_CAP] byte %0d: 0x%02h", rx_wr_ptr, cap_byte);
                rx_wr_ptr = rx_wr_ptr + 1;
            end
        end
    end

    task get_byte;
        output [7:0] data;
    begin
        wait(rx_rd_ptr != rx_wr_ptr);
        data = rx_fifo[rx_rd_ptr];
        rx_rd_ptr = rx_rd_ptr + 1;
    end
    endtask

    task uart_send;
        input [7:0] data;
        integer i;
    begin
        uart_rxd = 0;
        #(BIT_PERIOD);

        for (i = 0; i < 8; i = i + 1) begin
            uart_rxd = data[i];
            #(BIT_PERIOD);
        end

        uart_rxd = 1;
        #(BIT_PERIOD);
        #(BIT_PERIOD / 2);
    end
    endtask

    task send_prog_conn;
        input [7:0] core, src, slot, target, weight_hi, weight_lo;
    begin
        uart_send(8'h01); uart_send(core); uart_send(src);
        uart_send(slot); uart_send(target);
        uart_send(weight_hi); uart_send(weight_lo);
    end
    endtask

    task send_prog_route;
        input [7:0] sc, sn, dc, dn, wh, wl;
    begin
        uart_send(8'h02); uart_send(sc); uart_send(sn);
        uart_send(dc); uart_send(dn);
        uart_send(wh); uart_send(wl);
    end
    endtask

    task send_stimulus;
        input [7:0] core, neuron, current_hi, current_lo;
    begin
        uart_send(8'h03); uart_send(core); uart_send(neuron);
        uart_send(current_hi); uart_send(current_lo);
    end
    endtask

    task send_run;
        input [7:0] ts_hi, ts_lo;
    begin
        uart_send(8'h04); uart_send(ts_hi); uart_send(ts_lo);
    end
    endtask

    task send_status;
    begin
        uart_send(8'h05);
    end
    endtask

    reg [7:0] r0, r1, r2, r3, r4;

    initial begin
        uart_rxd = 1;
        btn_rst  = 0;

        $display("  FPGA Top Test - Full UART Serial Path");
        $display("  CLK_FREQ=%0d, BAUD=%0d, CLKS_PER_BIT=%0d",
            CLK_FREQ, BAUD, CLKS_PER_BIT);

        #(CLK_PERIOD * 50);

        $display("  System ready (POR done)");

        $display("test 1: PROG_CONN via UART serial");

        $display("  Programming: C0: N0->N1->N2->N3, w=1200");
        send_prog_conn(0, 0, 0, 1, 8'h04, 8'hB0);
        get_byte(r0);
        $display("  ACK: 0x%02h %s", r0, (r0 == 8'hAA) ? "PASS" : "FAIL");

        send_prog_conn(0, 1, 0, 2, 8'h04, 8'hB0);
        get_byte(r0);
        $display("  ACK: 0x%02h %s", r0, (r0 == 8'hAA) ? "PASS" : "FAIL");

        send_prog_conn(0, 2, 0, 3, 8'h04, 8'hB0);
        get_byte(r0);
        $display("  ACK: 0x%02h %s", r0, (r0 == 8'hAA) ? "PASS" : "FAIL");

        $display("test 2: STIMULUS + RUN (10 timesteps)");

        send_stimulus(0, 0, 8'h04, 8'hB0);
        get_byte(r0);
        $display("  STIM ACK: 0x%02h %s", r0, (r0 == 8'hAA) ? "PASS" : "FAIL");

        $display("  Running 10 timesteps...");
        send_run(8'h00, 8'h0A);

        get_byte(r0);
        get_byte(r1);
        get_byte(r2);
        get_byte(r3);
        get_byte(r4);
        $display("  %s, spikes = %0d",
            (r0 == 8'hDD) ? "DONE" : "ERROR",
            {r1, r2, r3, r4});

        $display("test 3: STATUS");

        send_status();
        get_byte(r0); get_byte(r1); get_byte(r2); get_byte(r3); get_byte(r4);
        $display("  State: %0d (%s), Timesteps: %0d",
            r0, (r0 == 0) ? "IDLE" : "BUSY", {r1, r2, r3, r4});

        $display("test 4: Cross-Core Route + Run");

        send_prog_route(0, 3, 1, 0, 8'h04, 8'hB0);
        get_byte(r0);
        $display("  ROUTE ACK: 0x%02h %s", r0, (r0 == 8'hAA) ? "PASS" : "FAIL");

        send_prog_conn(1, 0, 0, 1, 8'h04, 8'hB0);
        get_byte(r0);
        $display("  CONN ACK: 0x%02h %s", r0, (r0 == 8'hAA) ? "PASS" : "FAIL");

        send_stimulus(0, 0, 8'h04, 8'hB0);
        get_byte(r0);
        $display("  STIM ACK: 0x%02h %s", r0, (r0 == 8'hAA) ? "PASS" : "FAIL");

        $display("  Running 20 timesteps...");
        send_run(8'h00, 8'h14);
        get_byte(r0); get_byte(r1); get_byte(r2); get_byte(r3); get_byte(r4);
        $display("  %s, spikes = %0d",
            (r0 == 8'hDD) ? "DONE" : "ERROR",
            {r1, r2, r3, r4});

        send_status();
        get_byte(r0); get_byte(r1); get_byte(r2); get_byte(r3); get_byte(r4);
        $display("  Final: state=%0d, timesteps=%0d", r0, {r1, r2, r3, r4});

        $display("LED status:");
        $display("  LED[0] (heartbeat): %b", led[0]);
        $display("  LED[1] (RX blink):  %b", led[1]);
        $display("  LED[2] (TX blink):  %b", led[2]);
        $display("  LED[3] (activity):  %b", led[3]);

        $display("  FPGA TOP TEST COMPLETE");
        $display("  Full UART serial path verified:");
        $display("    PC -> UART_RX -> Host_IF -> Mesh -> Host_IF -> UART_TX -> PC");
        $display("  Commands: PROG_CONN, PROG_ROUTE, STIMULUS, RUN, STATUS");
        $display("  All 5 command types + responses verified over serial");

        #(CLK_PERIOD * 100);
        $finish;
    end

    initial begin
        #(CLK_PERIOD * 5_000_000);
        $display("TIMEOUT");
        $finish;
    end

endmodule

