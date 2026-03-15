# ============================================================================
# CL Synthesis Constraints — Neuromorphic Chip on AWS F2
# ============================================================================
# These are applied during synthesis only (not implementation).

# No false paths or multicycle needed — single clock domain design.
# The Shell provides clk_main_a0 at 250 MHz (4.0 ns period).
# All neuromorphic logic is synchronous to this single clock.
