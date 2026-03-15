## ============================================================================
## Arty A7-100T Pin Constraints
## ============================================================================

## System Clock (100 MHz)
set_property -dict { PACKAGE_PIN E3    IOSTANDARD LVCMOS33 } [get_ports {clk}]
create_clock -add -name sys_clk_pin -period 10.00 -waveform {0 5} [get_ports {clk}]

## Reset (BTN0, active-high)
set_property -dict { PACKAGE_PIN D9    IOSTANDARD LVCMOS33 } [get_ports {btn_rst}]

## UART
set_property -dict { PACKAGE_PIN A9    IOSTANDARD LVCMOS33 } [get_ports {uart_rxd}]
set_property -dict { PACKAGE_PIN D10   IOSTANDARD LVCMOS33 } [get_ports {uart_txd}]

## Status LEDs
set_property -dict { PACKAGE_PIN H5    IOSTANDARD LVCMOS33 } [get_ports {led[0]}]
set_property -dict { PACKAGE_PIN J5    IOSTANDARD LVCMOS33 } [get_ports {led[1]}]
set_property -dict { PACKAGE_PIN T9    IOSTANDARD LVCMOS33 } [get_ports {led[2]}]
set_property -dict { PACKAGE_PIN T10   IOSTANDARD LVCMOS33 } [get_ports {led[3]}]

## RGB LEDs (unused)
#set_property -dict { PACKAGE_PIN F6    IOSTANDARD LVCMOS33 } [get_ports {led_r[0]}]
#set_property -dict { PACKAGE_PIN J4    IOSTANDARD LVCMOS33 } [get_ports {led_g[0]}]
#set_property -dict { PACKAGE_PIN J2    IOSTANDARD LVCMOS33 } [get_ports {led_b[0]}]

## Configuration
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property CFGBVS VCCO [current_design]

## Bitstream
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
