"""Unit tests of the NQX-S1 bit-accurate model and its float reference."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from nqx_s1 import Csr, Op, S1Core, S1Params, host
from nqx_s1 import cordic, quant
from nqx_s1 import reference as ref
from nqx_s1.params import CHIP_ID, INC_DEFAULT, PHI

P = S1Params()
NQX_CORE = Path(__file__).resolve().parents[3] / "nqx-core"


def rand_vec(rng: random.Random, dim: int = P.dim, lim: int = 32767) -> list[int]:
    return [rng.randint(-lim - 1, lim) for _ in range(dim)]


# ---------------------------------------------------------------- params


def test_phase_increments_are_golden_ratio_constants():
    assert INC_DEFAULT == (0x61C88647, 0x9E3779B9, 0)
    assert abs(INC_DEFAULT[0] / 2**32 - 1 / PHI**2) < 2**-32
    assert abs(INC_DEFAULT[1] / 2**32 - 1 / PHI) < 2**-32
    assert (INC_DEFAULT[0] + INC_DEFAULT[1]) % 2**32 == 0


def test_derived_widths():
    assert (P.iw, P.dw, P.cw, P.hb) == (20, 24, 28, 3)
    assert P.packet_bytes == 70
    assert P.cordic_latency == P.n_iter + 2
    assert S1Params(dim=16).dw == 22


@pytest.mark.parametrize("dim", [8, 16, 32, 64, 128, 256])
def test_layer_pairs_are_disjoint_and_match_reference(dim):
    p = S1Params(dim=dim)
    for layer in (0, 1, 2):
        idx = [e for i, j, _ in p.layer_pairs(layer) for e in (i, j)]
        assert len(idx) == len(set(idx))
        assert all(0 <= e < dim for e in idx)
    if NQX_CORE.exists():
        sys.path.insert(0, str(NQX_CORE))
        from nqx.constants import NQXConfig
        from nqx.lut import GoldenAngleLUT

        lut = GoldenAngleLUT(NQXConfig(dim=dim))
        for layer in (0, 1, 2):
            ours = [(i, j) for i, j, _ in p.layer_pairs(layer)]
            assert ours == list(lut.layer(layer).pairs)
            angles = [a for _, _, a in ref.layer_angles(p, layer)]
            assert np.allclose(angles, lut.layer(layer).angles, rtol=0, atol=1e-12)


def test_reference_layer3_is_identity():
    x = np.random.default_rng(0).standard_normal((4, P.dim))
    y = ref._apply(P, x, 2, inverse=False)
    assert np.max(np.abs(y - x)) < 1e-10


def test_layer2_angles_are_negated_layer1_angles():
    for k in range(P.dim // 2 - 1):
        a1 = (k + 1) * INC_DEFAULT[0] % 2**32
        a2 = (k + 1) * INC_DEFAULT[1] % 2**32
        assert (a1 + a2) % 2**32 == 0


# ---------------------------------------------------------------- CORDIC


def test_cordic_rotation_accuracy():
    # Error budget in RF units (2^-FW of an int16 LSB): a few LSBs of datapath
    # rounding plus a relative term dominated by the rounded atan table
    # (~3e-5 worst case, i.e. -90 dB; the forward/inverse pair cancels it).
    rng = random.Random(1)
    worst = 0.0
    for _ in range(2000):
        a, b = rng.randint(-(2**22), 2**22), rng.randint(-(2**22), 2**22)
        z = rng.randrange(1 << P.zw)
        ang = z / (1 << P.zw) * 2 * math.pi
        x, y, sat = cordic.rotate(P, a, b, z)
        assert not sat
        ex = a * math.cos(ang) - b * math.sin(ang)
        ey = a * math.sin(ang) + b * math.cos(ang)
        bound = 4 + 4e-5 * math.hypot(a, b)
        worst = max(worst, abs(x - ex) / bound, abs(y - ey) / bound)
    assert worst < 1, worst


def test_cordic_vectoring_accuracy():
    rng = random.Random(2)
    for _ in range(2000):
        a, b = rng.randint(-(2**22), 2**22), rng.randint(-(2**22), 2**22)
        r, t, sat = cordic.vector(P, a, b)
        assert not sat
        assert abs(r - math.hypot(a, b)) < 4 + 4e-6 * math.hypot(a, b)
        if math.hypot(a, b) > 1000:
            err = (t / (1 << P.zw) * 2 * math.pi - math.atan2(b, a) + math.pi) % (2 * math.pi) - math.pi
            assert abs(err) < 1e-4


def test_cordic_saturates_instead_of_wrapping():
    hi = (1 << (P.dw - 1)) - 1
    x, y, sat = cordic.rotate(P, hi, hi, 1 << (P.zw - 3))
    assert sat and x <= hi and y == hi


# ---------------------------------------------------------------- quantizer


def test_radius_code_is_rounding():
    rmin, rmax = 1000, 8000
    for r in range(rmin, rmax + 1, 37):
        q, s = quant.radius_code(r, rmin, rmax)
        u = (r - rmin) * 7 / (rmax - rmin)
        assert abs(q - u) <= 0.5 + 1e-9
        assert s == (1 if u >= q else 0)


def test_angle_code_is_circular():
    zw = P.zw
    assert quant.angle_code(P, 0)[0] == 0
    assert quant.angle_code(P, (1 << zw) - 1)[0] == 0
    assert quant.angle_code(P, 1 << (zw - 1))[0] == 4
    assert quant.angle_code(P, (1 << (zw - 3)) - 1)[0] == 1


def test_decode_inverts_encode_within_quarter_step():
    rng = random.Random(3)
    rmin, rmax = 500, 90000
    step = quant.radius_step(P, rmin, rmax)
    for _ in range(500):
        r = rng.randint(rmin, rmax)
        t = rng.randrange(1 << P.zw)
        b = quant.encode_pair(P, r, t, rmin, rmax)
        rh, th, sat = quant.decode_pair(P, b, rmin, step, refine=True)
        assert not sat
        assert abs(rh - r) <= (rmax - rmin) / 28 + 2
        dt = (th - t + (1 << (P.zw - 1))) % (1 << P.zw) - (1 << (P.zw - 1))
        assert abs(dt) <= (1 << (P.zw - 5)) + 1


# ---------------------------------------------------------------- core


def test_csr_identity_and_scratch():
    out = S1Core().run(host.csrr(Csr.ID) + host.csrw(Csr.SCRATCH, 0x1234) + host.csrr(Csr.SCRATCH))
    assert int.from_bytes(out[:4], "little") == CHIP_ID
    assert int.from_bytes(out[4:8], "little") == 0x1234


def test_rotation_roundtrip_is_within_one_lsb():
    rng = random.Random(4)
    for _ in range(20):
        v = rand_vec(rng)
        prog = host.ldv(v) + b"".join(host.gvns(layer) for layer in (0, 1, 2))
        prog += b"".join(host.gvns(layer, True) for layer in (2, 1, 0)) + host.stv()
        out = host.unpack16(S1Core().run(prog))
        assert max(abs(a - b) for a, b in zip(out, v)) <= 1


def test_rotation_matches_float_reference():
    rng = random.Random(5)
    for _ in range(10):
        v = rand_vec(rng, lim=8000)
        c = S1Core()
        c.run(host.ldv(v) + host.gvns(0) + host.gvns(1) + host.gvns(2))
        fixed = np.array(c.rf) / 2**P.fw
        assert np.max(np.abs(fixed - ref.rotate(P, np.array(v, float)))) < 0.5


def test_encode_decode_matches_float_reference():
    rng = random.Random(6)
    x = np.array([rand_vec(rng, lim=16000) for _ in range(16)])
    out = []
    for v in x:
        pkt = S1Core().run(host.enc(list(v)))
        assert len(pkt) == P.packet_bytes
        out.append(host.unpack16(S1Core().run(host.dec(pkt))))
    err_fixed = np.sqrt(np.mean((np.array(out) - x) ** 2))
    err_float = np.sqrt(np.mean((ref.encode(P, x.astype(float)) - x) ** 2))
    assert abs(err_fixed - err_float) / err_float < 0.01


def test_zero_increment_bypasses_layer():
    v = rand_vec(random.Random(7))
    c = S1Core()
    c.run(host.ldv(v) + host.gvns(2))
    assert c.rf == [e << P.fw for e in v] and c.cordic_ops == 0


def test_illegal_opcode_and_layer_set_status():
    c = S1Core()
    c.run(bytes([0x7F, Op.GVNS, 3]))
    assert c.status == 1
    c.run(host.csrw(Csr.STATUS, 1))
    assert c.status == 0


def test_decode_saturation_flag():
    c = S1Core()
    big = ((1 << P.dw) - 1).to_bytes(P.hb, "little")
    c.run(host.dequant(bytes(P.hb) + big + bytes(P.n_pairs)))
    assert c.status & 2
