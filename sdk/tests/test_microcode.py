import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import neurocore as nc
from neurocore.microcode import (
    encode_instruction, decode_instruction, execute_program,
    LearningRule, _assemble,
    OP_NOP, OP_ADD, OP_SUB, OP_MUL, OP_SHR, OP_SHL,
    OP_MAX, OP_MIN, OP_LOADI, OP_STORE_W, OP_STORE_E,
    OP_SKIP_Z, OP_SKIP_NZ, OP_HALT,
    R_TRACE1, R_TRACE2, R_WEIGHT, R_ELIG, R_CONST,
    R_TEMP0, R_TEMP1, R_REWARD,
    LTD_START, LTD_END, LTP_START, LTP_END,
    MICROCODE_DEPTH,
)
from neurocore.constants import NEURONS_PER_CORE, WEIGHT_MAX_STDP, WEIGHT_MIN_STDP

class TestEncoding:
    def test_encode_decode_roundtrip(self):
        word = encode_instruction(OP_ADD, dst=R_WEIGHT, src_a=R_TRACE1, src_b=R_TEMP0)
        d = decode_instruction(word)
        assert d["op"] == OP_ADD
        assert d["dst"] == R_WEIGHT
        assert d["src_a"] == R_TRACE1
        assert d["src_b"] == R_TEMP0
        assert d["op_name"] == "ADD"

    def test_all_opcodes_valid(self):
        for op in range(14):
            word = encode_instruction(op)
            assert 0 <= word <= 0xFFFFFFFF
            d = decode_instruction(word)
            assert d["op"] == op

    def test_shift_encoding(self):
        for shift in range(8):
            word = encode_instruction(OP_SHR, dst=R_TEMP0, src_a=R_TRACE1, shift=shift)
            d = decode_instruction(word)
            assert d["shift"] == shift

    def test_immediate_encoding(self):
        for imm in [0, 1, -1, 32767, -32768, 100, -100]:
            word = encode_instruction(OP_LOADI, dst=R_CONST, imm=imm)
            d = decode_instruction(word)
            assert d["imm"] == imm

    def test_invalid_opcode_raises(self):
        with pytest.raises(ValueError):
            encode_instruction(14)
        with pytest.raises(ValueError):
            encode_instruction(-1)

    def test_invalid_register_raises(self):
        with pytest.raises(ValueError):
            encode_instruction(OP_ADD, dst=8)

class TestExecution:
    def test_add(self):
        prog = [encode_instruction(OP_NOP)] * MICROCODE_DEPTH
        prog[0] = encode_instruction(OP_ADD, dst=R_TEMP0, src_a=R_TRACE1, src_b=R_WEIGHT)
        prog[1] = encode_instruction(OP_HALT)
        regs = [10, 0, 20, 0, 0, 0, 0, 0]
        result = execute_program(prog, 0, 16, regs)
        assert regs[R_TEMP0] == 30

    def test_sub(self):
        prog = [encode_instruction(OP_NOP)] * MICROCODE_DEPTH
        prog[0] = encode_instruction(OP_SUB, dst=R_TEMP0, src_a=R_WEIGHT, src_b=R_TRACE1)
        prog[1] = encode_instruction(OP_HALT)
        regs = [30, 0, 100, 0, 0, 0, 0, 0]
        execute_program(prog, 0, 16, regs)
        assert regs[R_TEMP0] == 70

    def test_shr(self):
        prog = [encode_instruction(OP_NOP)] * MICROCODE_DEPTH
        prog[0] = encode_instruction(OP_SHR, dst=R_TEMP0, src_a=R_TRACE1, shift=3)
        prog[1] = encode_instruction(OP_HALT)
        regs = [100, 0, 0, 0, 0, 0, 0, 0]
        execute_program(prog, 0, 16, regs)
        assert regs[R_TEMP0] == 12

    def test_shl(self):
        prog = [encode_instruction(OP_NOP)] * MICROCODE_DEPTH
        prog[0] = encode_instruction(OP_SHL, dst=R_TEMP0, src_a=R_TRACE1, shift=2)
        prog[1] = encode_instruction(OP_HALT)
        regs = [5, 0, 0, 0, 0, 0, 0, 0]
        execute_program(prog, 0, 16, regs)
        assert regs[R_TEMP0] == 20

    def test_max_min(self):
        prog = [encode_instruction(OP_NOP)] * MICROCODE_DEPTH
        prog[0] = encode_instruction(OP_MAX, dst=R_TEMP0, src_a=R_TRACE1, src_b=R_WEIGHT)
        prog[1] = encode_instruction(OP_MIN, dst=R_TEMP1, src_a=R_TRACE1, src_b=R_WEIGHT)
        prog[2] = encode_instruction(OP_HALT)
        regs = [30, 0, 100, 0, 0, 0, 0, 0]
        execute_program(prog, 0, 16, regs)
        assert regs[R_TEMP0] == 100  # max(30, 100)
        assert regs[R_TEMP1] == 30   # min(30, 100)

    def test_loadi(self):
        prog = [encode_instruction(OP_NOP)] * MICROCODE_DEPTH
        prog[0] = encode_instruction(OP_LOADI, dst=R_CONST, imm=42)
        prog[1] = encode_instruction(OP_HALT)
        regs = [0] * 8
        execute_program(prog, 0, 16, regs)
        assert regs[R_CONST] == 42

    def test_skip_z(self):
        prog = [encode_instruction(OP_NOP)] * MICROCODE_DEPTH
        prog[0] = encode_instruction(OP_SKIP_Z, src_a=R_TRACE1)  # R0=0, skip
        prog[1] = encode_instruction(OP_LOADI, dst=R_TEMP0, imm=99)  # skipped
        prog[2] = encode_instruction(OP_LOADI, dst=R_TEMP1, imm=42)  # executed
        prog[3] = encode_instruction(OP_HALT)
        regs = [0] * 8
        execute_program(prog, 0, 16, regs)
        assert regs[R_TEMP0] == 0   # skipped
        assert regs[R_TEMP1] == 42  # executed

    def test_store_w(self):
        prog = [encode_instruction(OP_NOP)] * MICROCODE_DEPTH
        prog[0] = encode_instruction(OP_LOADI, dst=R_WEIGHT, imm=999)
        prog[1] = encode_instruction(OP_STORE_W, src_a=R_WEIGHT)
        prog[2] = encode_instruction(OP_HALT)
        regs = [0, 0, 500, 0, 0, 0, 0, 0]
        result = execute_program(prog, 0, 16, regs)
        assert result["weight_written"] is True
        assert result["weight"] == 999

    def test_store_e(self):
        prog = [encode_instruction(OP_NOP)] * MICROCODE_DEPTH
        prog[0] = encode_instruction(OP_LOADI, dst=R_ELIG, imm=-50)
        prog[1] = encode_instruction(OP_STORE_E, src_a=R_ELIG)
        prog[2] = encode_instruction(OP_HALT)
        regs = [0] * 8
        result = execute_program(prog, 0, 16, regs)
        assert result["elig_written"] is True
        assert result["elig"] == -50

class TestAssembler:
    def test_basic_assembly(self):
        text = """
        SHR R5, R0, 3
        SKIP_Z R5
        SUB R2, R2, R5
        STORE_W R2
        HALT
        """
        instrs = _assemble(text)
        assert len(instrs) == 5
        d = decode_instruction(instrs[0])
        assert d["op_name"] == "SHR"
        assert d["dst"] == R_TEMP0
        assert d["src_a"] == R_TRACE1
        assert d["shift"] == 3

    def test_comments_stripped(self):
        text = """
        ; This is a comment
        NOP
        HALT
        """
        instrs = _assemble(text)
        assert len(instrs) == 2

    def test_loadi_assembly(self):
        text = "LOADI R4, 0xFF"
        instrs = _assemble(text)
        d = decode_instruction(instrs[0])
        assert d["op"] == OP_LOADI
        assert d["imm"] == 255

class TestLearningRule:
    def test_stdp_factory(self):
        rule = LearningRule.stdp()
        prog = rule.get_program()
        assert len(prog) == MICROCODE_DEPTH
        ltd = rule.get_ltd()
        assert any(decode_instruction(w)["op"] != OP_NOP for w in ltd)

    def test_three_factor_factory(self):
        rule = LearningRule.three_factor()
        ltd = rule.get_ltd()
        has_store_e = any(decode_instruction(w)["op"] == OP_STORE_E for w in ltd)
        has_store_w = any(decode_instruction(w)["op"] == OP_STORE_W for w in ltd)
        assert has_store_e
        assert not has_store_w

    def test_from_instructions(self):
        ltd = [encode_instruction(OP_HALT)]
        ltp = [encode_instruction(OP_HALT)]
        rule = LearningRule.from_instructions(ltd, ltp)
        prog = rule.get_program()
        assert decode_instruction(prog[0])["op"] == OP_HALT
        assert decode_instruction(prog[16])["op"] == OP_HALT

    def test_assemble_ltd_ltp(self):
        rule = LearningRule()
        rule.assemble_ltd("SHR R5, R0, 3\nSKIP_Z R5\nSUB R2, R2, R5\nSTORE_W R2\nHALT")
        rule.assemble_ltp("SHR R5, R0, 3\nSKIP_Z R5\nADD R2, R2, R5\nSTORE_W R2\nHALT")
        prog = rule.get_program()
        assert decode_instruction(prog[0])["op"] == OP_SHR
        assert decode_instruction(prog[16])["op"] == OP_SHR

class TestMicrocodeSTDP:

    def test_default_microcode_stdp_weight_change(self):
        net = nc.Network()
        src = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
        tgt = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
        net.connect(src, tgt, topology="all_to_all", weight=500)
        net.set_learning_rule(LearningRule.stdp())

        sim = nc.Simulator()
        sim.deploy(net)
        sim.set_learning(learn=True)

        sim.inject(src, current=200)
        sim.run(1)  # src spikes at t=0
        sim.run(1)  # tgt receives input, spikes at t=1 -> LTP

        adj = sim._adjacency
        for targets in adj.values():
            for entry in targets:
                w = entry[1]
                assert w > 500, f"Expected LTP increase, got {w}"

    def test_default_microcode_three_factor(self):
        net = nc.Network()
        src = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
        tgt = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
        net.connect(src, tgt, topology="all_to_all", weight=500)
        net.set_learning_rule(LearningRule.three_factor())

        sim = nc.Simulator()
        sim.deploy(net)
        sim.set_learning(learn=True, three_factor=True)

        sim.inject(src, current=200)
        sim.inject(tgt, current=200)
        sim.run(3)

        assert len(sim._eligibility) > 0

        for targets in sim._adjacency.values():
            for entry in targets:
                assert entry[1] == 500

    def test_anti_stdp_custom_rule(self):
        rule = LearningRule()
        rule.assemble_ltd(
            "SHR R5, R0, 3\n"
            "SKIP_Z R5\n"
            "ADD R2, R2, R5\n"
            "STORE_W R2\n"
            "HALT"
        )
        rule.assemble_ltp(
            "SHR R5, R0, 3\n"
            "SKIP_Z R5\n"
            "SUB R2, R2, R5\n"
            "STORE_W R2\n"
            "HALT"
        )

        net = nc.Network()
        src = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
        tgt = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
        net.connect(src, tgt, topology="all_to_all", weight=500)
        net.set_learning_rule(rule)

        sim = nc.Simulator()
        sim.deploy(net)
        sim.set_learning(learn=True)

        sim.inject(src, current=200)
        sim.run(1)
        sim.run(1)

        adj = sim._adjacency
        for targets in adj.values():
            for entry in targets:
                w = entry[1]
                assert w < 500, f"Anti-STDP should decrease weight, got {w}"

    def test_compiler_generates_learn_cmds(self):
        from neurocore.compiler import Compiler

        net = nc.Network()
        src = net.population(2)
        tgt = net.population(2)
        net.connect(src, tgt, topology="all_to_all", weight=200)
        net.set_learning_rule(LearningRule.stdp())

        compiled = Compiler().compile(net)
        assert len(compiled.prog_learn_cmds) > 0
        for cmd in compiled.prog_learn_cmds:
            assert "core" in cmd
            assert "addr" in cmd
            assert "instr" in cmd
