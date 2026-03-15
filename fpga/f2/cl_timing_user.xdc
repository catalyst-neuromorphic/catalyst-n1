# ===========================================================================
# CL Neuromorphic — User Timing Constraints
# ===========================================================================

# Generated clock from MMCME4 (62.5 MHz)
# The MMCM auto-generates clock constraints from its parameters,
# but we add explicit false paths between clock domains for CDC.

# Async FIFO CDC: false paths between AXI clock and neuro clock
# The Gray-code synchronizers in async_fifo handle the CDC safely.
set_false_path -from [get_clocks -of_objects [get_pins WRAPPER/CL/u_mmcm/CLKIN1]] \
               -to   [get_clocks -of_objects [get_pins WRAPPER/CL/u_mmcm/CLKOUT0]]
set_false_path -from [get_clocks -of_objects [get_pins WRAPPER/CL/u_mmcm/CLKOUT0]] \
               -to   [get_clocks -of_objects [get_pins WRAPPER/CL/u_mmcm/CLKIN1]]
