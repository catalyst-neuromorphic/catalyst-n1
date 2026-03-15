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
REGISTER_BY_NAME.update({
    "TRACE1": R_TRACE1, "TRACE2": R_TRACE2, "WEIGHT": R_WEIGHT,
    "ELIG": R_ELIG, "CONST": R_CONST, "TEMP0": R_TEMP0,
    "TEMP1": R_TEMP1, "REWARD": R_REWARD,
})

MICROCODE_DEPTH = 64
LTD_START = 0
LTD_END   = 15
LTP_START = 16
LTP_END   = 31

def encode_instruction(op, dst=0, src_a=0, src_b=0, shift=0, imm=0):
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
    word = word & 0xFFFFFFFF
    op = (word >> 28) & 0xF
    dst = (word >> 25) & 0x7
    src_a = (word >> 22) & 0x7
    src_b = (word >> 19) & 0x7
    shift = (word >> 16) & 0x7
    imm = word & 0xFFFF
    if imm >= 0x8000:
        imm -= 0x10000
    return {
        "op": op, "dst": dst, "src_a": src_a, "src_b": src_b,
        "shift": shift, "imm": imm,
        "op_name": OPCODE_NAMES.get(op, f"UNKNOWN({op})"),
    }

def _default_stdp_program():
    program = [encode_instruction(OP_NOP)] * MICROCODE_DEPTH

    program[0] = encode_instruction(OP_SHR, dst=R_TEMP0, src_a=R_TRACE1, shift=3)
    program[1] = encode_instruction(OP_SKIP_Z, src_a=R_TEMP0)
    program[2] = encode_instruction(OP_SUB, dst=R_WEIGHT, src_a=R_WEIGHT, src_b=R_TEMP0)
    program[3] = encode_instruction(OP_STORE_W, src_a=R_WEIGHT)
    program[4] = encode_instruction(OP_HALT)

    program[16] = encode_instruction(OP_SHR, dst=R_TEMP0, src_a=R_TRACE1, shift=3)
    program[17] = encode_instruction(OP_SKIP_Z, src_a=R_TEMP0)
    program[18] = encode_instruction(OP_ADD, dst=R_WEIGHT, src_a=R_WEIGHT, src_b=R_TEMP0)
    program[19] = encode_instruction(OP_STORE_W, src_a=R_WEIGHT)
    program[20] = encode_instruction(OP_HALT)

    return program

def _default_three_factor_program():
    program = [encode_instruction(OP_NOP)] * MICROCODE_DEPTH

    program[0] = encode_instruction(OP_SHR, dst=R_TEMP0, src_a=R_TRACE1, shift=3)
    program[1] = encode_instruction(OP_SKIP_Z, src_a=R_TEMP0)
    program[2] = encode_instruction(OP_SUB, dst=R_ELIG, src_a=R_ELIG, src_b=R_TEMP0)
    program[3] = encode_instruction(OP_STORE_E, src_a=R_ELIG)
    program[4] = encode_instruction(OP_HALT)

    program[16] = encode_instruction(OP_SHR, dst=R_TEMP0, src_a=R_TRACE1, shift=3)
    program[17] = encode_instruction(OP_SKIP_Z, src_a=R_TEMP0)
    program[18] = encode_instruction(OP_ADD, dst=R_ELIG, src_a=R_ELIG, src_b=R_TEMP0)
    program[19] = encode_instruction(OP_STORE_E, src_a=R_ELIG)
    program[20] = encode_instruction(OP_HALT)

    return program

DEFAULT_STDP_PROGRAM = _default_stdp_program()
DEFAULT_THREE_FACTOR_PROGRAM = _default_three_factor_program()

class LearningRule:

    def __init__(self):
        self._program = [encode_instruction(OP_NOP)] * MICROCODE_DEPTH

    @classmethod
    def stdp(cls):
        rule = cls()
        rule._program = list(DEFAULT_STDP_PROGRAM)
        return rule

    @classmethod
    def three_factor(cls):
        rule = cls()
        rule._program = list(DEFAULT_THREE_FACTOR_PROGRAM)
        return rule

    @classmethod
    def from_instructions(cls, ltd_instrs, ltp_instrs):
        rule = cls()
        for i, instr in enumerate(ltd_instrs[:16]):
            rule._program[LTD_START + i] = instr
        for i, instr in enumerate(ltp_instrs[:16]):
            rule._program[LTP_START + i] = instr
        return rule

    def assemble_ltd(self, text):
        instrs = _assemble(text)
        for i, instr in enumerate(instrs[:16]):
            self._program[LTD_START + i] = instr

    def assemble_ltp(self, text):
        instrs = _assemble(text)
        for i, instr in enumerate(instrs[:16]):
            self._program[LTP_START + i] = instr

    def get_program(self):
        return list(self._program)

    def get_ltd(self):
        return self._program[LTD_START:LTD_END + 1]

    def get_ltp(self):
        return self._program[LTP_START:LTP_END + 1]

def _parse_register(token):
    token = token.strip().rstrip(",").upper()
    if token in REGISTER_BY_NAME:
        return REGISTER_BY_NAME[token]
    raise ValueError(f"Unknown register: '{token}'")

def _assemble(text):
    instructions = []
    for line in text.strip().split("\n"):
        line = line.strip()
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
            dst = _parse_register(parts[1])
            imm = int(parts[2], 0)
        elif op in (OP_SKIP_Z, OP_SKIP_NZ, OP_STORE_W, OP_STORE_E):
            src_a = _parse_register(parts[1])
        elif op in (OP_SHR, OP_SHL):
            dst = _parse_register(parts[1])
            src_a = _parse_register(parts[2])
            shift = int(parts[3])
        elif op == OP_MUL:
            dst = _parse_register(parts[1])
            src_a = _parse_register(parts[2])
            src_b = _parse_register(parts[3])
            if len(parts) > 4:
                shift = int(parts[4])
        else:
            dst = _parse_register(parts[1])
            src_a = _parse_register(parts[2])
            src_b = _parse_register(parts[3])

        instructions.append(encode_instruction(op, dst, src_a, src_b, shift, imm))

    return instructions

def execute_program(program, pc_start, pc_end, regs):
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
                pc += 2
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
            pc += 1

    return {
        "weight": final_weight,
        "elig": final_elig,
        "weight_written": weight_written,
        "elig_written": elig_written,
    }
