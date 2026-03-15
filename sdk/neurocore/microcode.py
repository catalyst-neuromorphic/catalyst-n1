"""P19 Microcode Learning Engine — ISA, assembler, and learning rule builder.

32-bit instruction format:
  {op[31:28], dst[27:25], src_a[24:22], src_b[21:19], shift[18:16], imm[15:0]}

8 registers:
  R0=trace1, R1=trace2, R2=weight, R3=eligibility, R4=constant,
  R5=temp0, R6=temp1, R7=reward

14 opcodes:
  NOP, ADD, SUB, MUL, SHR, SHL, MAX, MIN, LOADI,
  STORE_W, STORE_E, SKIP_Z, SKIP_NZ, HALT

Default programs reproduce P13 STDP + 3-factor behavior.
"""

# Opcodes (4-bit, bits[31:28])
OP_NOP      = 0
OP_ADD      = 1
OP_SUB      = 2
OP_MUL      = 3
OP_SHR      = 4
OP_SHL      = 5
OP_MAX      = 6
OP_MIN      = 7
OP_LOADI    = 8
OP_STORE_W  = 9
OP_STORE_E  = 10
OP_SKIP_Z   = 11
OP_SKIP_NZ  = 12
OP_HALT     = 13

OPCODE_NAMES = {
    OP_NOP: "NOP", OP_ADD: "ADD", OP_SUB: "SUB", OP_MUL: "MUL",
    OP_SHR: "SHR", OP_SHL: "SHL", OP_MAX: "MAX", OP_MIN: "MIN",
    OP_LOADI: "LOADI", OP_STORE_W: "STORE_W", OP_STORE_E: "STORE_E",
    OP_SKIP_Z: "SKIP_Z", OP_SKIP_NZ: "SKIP_NZ", OP_HALT: "HALT",
}
OPCODE_BY_NAME = {v: k for k, v in OPCODE_NAMES.items()}

# Registers (3-bit, 0-7)
R_TRACE1  = 0
R_TRACE2  = 1
R_WEIGHT  = 2
R_ELIG    = 3
R_CONST   = 4
R_TEMP0   = 5
R_TEMP1   = 6
R_REWARD  = 7

REGISTER_NAMES = {
    R_TRACE1: "R0", R_TRACE2: "R1", R_WEIGHT: "R2", R_ELIG: "R3",
    R_CONST: "R4", R_TEMP0: "R5", R_TEMP1: "R6", R_REWARD: "R7",
}
REGISTER_BY_NAME = {v: k for k, v in REGISTER_NAMES.items()}
# Also accept named aliases
REGISTER_BY_NAME.update({
    "TRACE1": R_TRACE1, "TRACE2": R_TRACE2, "WEIGHT": R_WEIGHT,
    "ELIG": R_ELIG, "CONST": R_CONST, "TEMP0": R_TEMP0,
    "TEMP1": R_TEMP1, "REWARD": R_REWARD,
})

# Microcode memory depth per core
MICROCODE_DEPTH = 64
# Program regions
LTD_START = 0
LTD_END   = 15
LTP_START = 16
LTP_END   = 31


def encode_instruction(op, dst=0, src_a=0, src_b=0, shift=0, imm=0):
    """Encode a 32-bit microcode instruction.

    Args:
        op: Opcode (0-13)
        dst: Destination register (0-7)
        src_a: Source register A (0-7)
        src_b: Source register B (0-7)
        shift: Shift amount (0-7)
        imm: 16-bit immediate (signed, -32768 to 32767)

    Returns:
        32-bit unsigned instruction word
    """
    if op < 0 or op > 13:
        raise ValueError(f"Invalid opcode: {op}")
    if any(r < 0 or r > 7 for r in (dst, src_a, src_b)):
        raise ValueError("Register index must be 0-7")
    if shift < 0 or shift > 7:
        raise ValueError(f"Shift must be 0-7, got {shift}")

    imm_u16 = imm & 0xFFFF
    word = ((op & 0xF) << 28) | ((dst & 0x7) << 25) | ((src_a & 0x7) << 22) \
        | ((src_b & 0x7) << 19) | ((shift & 0x7) << 16) | imm_u16
    return word & 0xFFFFFFFF


def decode_instruction(word):
    """Decode a 32-bit instruction word to its fields.

    Returns:
        dict with keys: op, dst, src_a, src_b, shift, imm, op_name
    """
    word = word & 0xFFFFFFFF
    op = (word >> 28) & 0xF
    dst = (word >> 25) & 0x7
    src_a = (word >> 22) & 0x7
    src_b = (word >> 19) & 0x7
    shift = (word >> 16) & 0x7
    imm = word & 0xFFFF
    # Sign-extend immediate
    if imm >= 0x8000:
        imm -= 0x10000
    return {
        "op": op, "dst": dst, "src_a": src_a, "src_b": src_b,
        "shift": shift, "imm": imm,
        "op_name": OPCODE_NAMES.get(op, f"UNKNOWN({op})"),
    }


def _default_stdp_program():
    """Build the default STDP program that reproduces P13 behavior.

    LTD (addresses 0-4): pre spiked, depress weight by post_trace >> 3
    LTP (addresses 16-20): post spiked, potentiate weight by pre_trace >> 3
    """
    program = [encode_instruction(OP_NOP)] * MICROCODE_DEPTH

    # LTD: R0=post_trace, R2=weight
    # 0: R5 = R0 >> 3   (delta = trace >> LEARN_SHIFT)
    program[0] = encode_instruction(OP_SHR, dst=R_TEMP0, src_a=R_TRACE1, shift=3)
    # 1: skip if R5 == 0
    program[1] = encode_instruction(OP_SKIP_Z, src_a=R_TEMP0)
    # 2: R2 = R2 - R5
    program[2] = encode_instruction(OP_SUB, dst=R_WEIGHT, src_a=R_WEIGHT, src_b=R_TEMP0)
    # 3: store weight
    program[3] = encode_instruction(OP_STORE_W, src_a=R_WEIGHT)
    # 4: halt
    program[4] = encode_instruction(OP_HALT)

    # LTP: R0=pre_trace, R2=weight
    # 16: R5 = R0 >> 3
    program[16] = encode_instruction(OP_SHR, dst=R_TEMP0, src_a=R_TRACE1, shift=3)
    # 17: skip if R5 == 0
    program[17] = encode_instruction(OP_SKIP_Z, src_a=R_TEMP0)
    # 18: R2 = R2 + R5
    program[18] = encode_instruction(OP_ADD, dst=R_WEIGHT, src_a=R_WEIGHT, src_b=R_TEMP0)
    # 19: store weight
    program[19] = encode_instruction(OP_STORE_W, src_a=R_WEIGHT)
    # 20: halt
    program[20] = encode_instruction(OP_HALT)

    return program


def _default_three_factor_program():
    """Build the default 3-factor program (STDP -> eligibility, not weight).

    LTD (addresses 0-4): elig -= post_trace >> 3
    LTP (addresses 16-20): elig += pre_trace >> 3
    """
    program = [encode_instruction(OP_NOP)] * MICROCODE_DEPTH

    # LTD: R0=post_trace, R3=eligibility
    program[0] = encode_instruction(OP_SHR, dst=R_TEMP0, src_a=R_TRACE1, shift=3)
    program[1] = encode_instruction(OP_SKIP_Z, src_a=R_TEMP0)
    program[2] = encode_instruction(OP_SUB, dst=R_ELIG, src_a=R_ELIG, src_b=R_TEMP0)
    program[3] = encode_instruction(OP_STORE_E, src_a=R_ELIG)
    program[4] = encode_instruction(OP_HALT)

    # LTP: R0=pre_trace, R3=eligibility
    program[16] = encode_instruction(OP_SHR, dst=R_TEMP0, src_a=R_TRACE1, shift=3)
    program[17] = encode_instruction(OP_SKIP_Z, src_a=R_TEMP0)
    program[18] = encode_instruction(OP_ADD, dst=R_ELIG, src_a=R_ELIG, src_b=R_TEMP0)
    program[19] = encode_instruction(OP_STORE_E, src_a=R_ELIG)
    program[20] = encode_instruction(OP_HALT)

    return program


DEFAULT_STDP_PROGRAM = _default_stdp_program()
DEFAULT_THREE_FACTOR_PROGRAM = _default_three_factor_program()


class LearningRule:
    """Configurable microcode learning rule.

    Usage:
        # Default STDP:
        rule = LearningRule.stdp()

        # Default 3-factor:
        rule = LearningRule.three_factor()

        # Custom from instructions:
        rule = LearningRule.from_instructions(ltd_program, ltp_program)

        # Custom from assembly text:
        rule = LearningRule()
        rule.assemble_ltd("SHR R5, R0, 3\\nSKIP_Z R5\\nSUB R2, R2, R5\\nSTORE_W R2\\nHALT")
        rule.assemble_ltp("SHR R5, R0, 3\\nSKIP_Z R5\\nADD R2, R2, R5\\nSTORE_W R2\\nHALT")
    """

    def __init__(self):
        self._program = [encode_instruction(OP_NOP)] * MICROCODE_DEPTH

    @classmethod
    def stdp(cls):
        """Factory: default 2-factor STDP rule."""
        rule = cls()
        rule._program = list(DEFAULT_STDP_PROGRAM)
        return rule

    @classmethod
    def three_factor(cls):
        """Factory: default 3-factor eligibility rule."""
        rule = cls()
        rule._program = list(DEFAULT_THREE_FACTOR_PROGRAM)
        return rule

    @classmethod
    def from_instructions(cls, ltd_instrs, ltp_instrs):
        """Build from lists of 32-bit instruction words.

        Args:
            ltd_instrs: List of up to 16 instruction words for LTD (addresses 0-15)
            ltp_instrs: List of up to 16 instruction words for LTP (addresses 16-31)
        """
        rule = cls()
        for i, instr in enumerate(ltd_instrs[:16]):
            rule._program[LTD_START + i] = instr
        for i, instr in enumerate(ltp_instrs[:16]):
            rule._program[LTP_START + i] = instr
        return rule

    def assemble_ltd(self, text):
        """Assemble LTD program from text mnemonics."""
        instrs = _assemble(text)
        for i, instr in enumerate(instrs[:16]):
            self._program[LTD_START + i] = instr

    def assemble_ltp(self, text):
        """Assemble LTP program from text mnemonics."""
        instrs = _assemble(text)
        for i, instr in enumerate(instrs[:16]):
            self._program[LTP_START + i] = instr

    def get_program(self):
        """Return the full 64-word microcode program."""
        return list(self._program)

    def get_ltd(self):
        """Return LTD region (addresses 0-15)."""
        return self._program[LTD_START:LTD_END + 1]

    def get_ltp(self):
        """Return LTP region (addresses 16-31)."""
        return self._program[LTP_START:LTP_END + 1]


def _parse_register(token):
    """Parse a register token like 'R0', 'R5', 'TRACE1', etc."""
    token = token.strip().rstrip(",").upper()
    if token in REGISTER_BY_NAME:
        return REGISTER_BY_NAME[token]
    raise ValueError(f"Unknown register: '{token}'")


def _assemble(text):
    """Assemble text mnemonics into instruction words.

    Format per line:
      OP DST, SRC_A, SRC_B [, SHIFT]
      OP DST, IMM                      (for LOADI)
      OP SRC_A                         (for SKIP_Z, SKIP_NZ, STORE_W, STORE_E)
      OP                               (for NOP, HALT)

    Lines starting with ';' or '#' are comments. Blank lines are skipped.

    Returns:
        List of 32-bit instruction words.
    """
    instructions = []
    for line in text.strip().split("\n"):
        line = line.strip()
        # Strip inline comments
        for ch in (';', '#'):
            if ch in line:
                line = line[:line.index(ch)].strip()
        if not line:
            continue

        parts = line.replace(",", " ").split()
        op_name = parts[0].upper()
        if op_name not in OPCODE_BY_NAME:
            raise ValueError(f"Unknown opcode: '{op_name}'")
        op = OPCODE_BY_NAME[op_name]

        dst = src_a = src_b = shift = 0
        imm = 0

        if op in (OP_NOP, OP_HALT):
            pass
        elif op == OP_LOADI:
            # LOADI DST, IMM
            dst = _parse_register(parts[1])
            imm = int(parts[2], 0)
        elif op in (OP_SKIP_Z, OP_SKIP_NZ, OP_STORE_W, OP_STORE_E):
            # OP SRC_A
            src_a = _parse_register(parts[1])
        elif op in (OP_SHR, OP_SHL):
            # OP DST, SRC_A, SHIFT
            dst = _parse_register(parts[1])
            src_a = _parse_register(parts[2])
            shift = int(parts[3])
        elif op == OP_MUL:
            # MUL DST, SRC_A, SRC_B [, SHIFT]
            dst = _parse_register(parts[1])
            src_a = _parse_register(parts[2])
            src_b = _parse_register(parts[3])
            if len(parts) > 4:
                shift = int(parts[4])
        else:
            # ADD, SUB, MAX, MIN: OP DST, SRC_A, SRC_B
            dst = _parse_register(parts[1])
            src_a = _parse_register(parts[2])
            src_b = _parse_register(parts[3])

        instructions.append(encode_instruction(op, dst, src_a, src_b, shift, imm))

    return instructions


def execute_program(program, pc_start, pc_end, regs):
    """Execute microcode instructions from pc_start to pc_end (or HALT).

    Args:
        program: List of 32-bit instruction words (full 64-word program)
        pc_start: Starting program counter
        pc_end: Maximum program counter (exclusive)
        regs: List of 8 register values [trace1, trace2, weight, elig, const, temp0, temp1, reward]

    Returns:
        dict with keys: weight, elig, weight_written, elig_written
    """
    pc = pc_start
    weight_written = False
    elig_written = False
    final_weight = regs[R_WEIGHT]
    final_elig = regs[R_ELIG]

    while pc < pc_end and pc < len(program):
        d = decode_instruction(program[pc])
        op = d["op"]

        if op == OP_NOP:
            pc += 1
        elif op == OP_ADD:
            regs[d["dst"]] = regs[d["src_a"]] + regs[d["src_b"]]
            pc += 1
        elif op == OP_SUB:
            regs[d["dst"]] = regs[d["src_a"]] - regs[d["src_b"]]
            pc += 1
        elif op == OP_MUL:
            regs[d["dst"]] = (regs[d["src_a"]] * regs[d["src_b"]]) >> d["shift"]
            pc += 1
        elif op == OP_SHR:
            val = regs[d["src_a"]]
            regs[d["dst"]] = val >> d["shift"] if val >= 0 else -((-val) >> d["shift"])
            pc += 1
        elif op == OP_SHL:
            regs[d["dst"]] = regs[d["src_a"]] << d["shift"]
            pc += 1
        elif op == OP_MAX:
            regs[d["dst"]] = max(regs[d["src_a"]], regs[d["src_b"]])
            pc += 1
        elif op == OP_MIN:
            regs[d["dst"]] = min(regs[d["src_a"]], regs[d["src_b"]])
            pc += 1
        elif op == OP_LOADI:
            regs[d["dst"]] = d["imm"]
            pc += 1
        elif op == OP_STORE_W:
            final_weight = regs[d["src_a"]]
            weight_written = True
            pc += 1
        elif op == OP_STORE_E:
            final_elig = regs[d["src_a"]]
            elig_written = True
            pc += 1
        elif op == OP_SKIP_Z:
            if regs[d["src_a"]] == 0:
                pc += 2  # skip next
            else:
                pc += 1
        elif op == OP_SKIP_NZ:
            if regs[d["src_a"]] != 0:
                pc += 2
            else:
                pc += 1
        elif op == OP_HALT:
            break
        else:
            pc += 1  # unknown op -> skip

    return {
        "weight": final_weight,
        "elig": final_elig,
        "weight_written": weight_written,
        "elig_written": elig_written,
    }
