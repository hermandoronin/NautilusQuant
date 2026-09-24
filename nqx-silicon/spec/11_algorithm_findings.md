# Algorithm findings from the hardware mapping

**Document status:** Revision 1.0, 2026-09-24.

Turning the reference into fixed-point hardware exposed four properties of
the NautilusQuant v2 rotation (`NautilusQuant_Model.md`,
`nqx-core/nqx/lut.py`). They matter for anyone who reuses the algorithm, in
silicon or software.

## F1. Layer 3 does not rotate

The reference angle of layer 3 is

    θ₃(k) = golden_angle · (k+1) · φ²  =  (2π/φ²) · (k+1) · φ²  =  2π · (k+1)

which is a whole number of turns: cos = 1, sin = 0. In `lut.py` the float
evaluation leaves |sin| ≤ 1.3·10⁻¹³ (DIM=128). Layer 3, the "butterfly"
that is meant to mix distant coordinates, is therefore an identity. Its 64
Givens pairs cost compute in the emulator and add no mixing.

- **In NQX-S1.** INC2 resets to 0. The sequencer skips a zero-increment
  layer, so encoding costs two layers, not three. The result equals the
  reference to within the reference's own float noise.
- **Possible fix.** Use a scale that is not a power of φ, for example
  θ₃(k) = (k+1)·2π·frac(√2), or replace φ² by φ³. Because INC2 is a
  register, this fix can be evaluated on the fabricated chip without a
  respin (see [`03_programming_model.md`](03_programming_model.md) §5.3).
  The effect on quality should be measured on real KV caches before the
  reference changes.

Tests: `model/tests/test_model.py::test_reference_layer3_is_identity`.

## F2. Layer 2 angles are the negated layer 1 angles

1/φ + 1/φ² = 1, so (k+1)·2π/φ ≡ −(k+1)·2π/φ² (mod 2π). Pair k of L2 rotates
by exactly minus the angle of pair k of L1. They act on different
coordinate pairs, so the transform is not trivial, but the angle sets are
identical. In hardware the two increments are `0x9E3779B9` and its two's
complement `0x61C88647`. These are the Fibonacci-hashing constants
(Knuth, TAOCP vol. 3), which is a neat way to explain the design.

Test: `test_layer2_angles_are_negated_layer1_angles`.

## F3. The "rotation state" is 12 bytes, not 1.9 KB

`nqx-core` stores cos/sin for 191 pairs (1 910 bytes at DIM=128). The angles
are (k+1)·c mod 2π, a linear phase ramp. A 32-bit phase accumulator per
layer regenerates them exactly (to 5·10⁻⁸ rad), and a CORDIC consumes
angles directly. The claim "the rotation collapses into a 1.9 KB ROM"
understates the advantage: it collapses into two constants. This is what
NQX-S1 implements.

## F4. The quantizer matters more than the rotation

Measured on the bit-accurate model
([`04_numerics_results.md`](04_numerics_results.md), KV-like data,
4 bits/value):

| | relative RMSE |
|---|---|
| nqx-core pipeline (golden rotation, per-feature batch range, sign bit unused on decode) | 0.316 |
| golden rotation + S1 quantizer | 0.147 |
| random orthogonal rotation + S1 quantizer | 0.122 |
| no rotation + S1 quantizer | 0.161 |

Consequences for the project:

1. **Change the quantizer in nqx-core.** Adopting the S1 quantizer rules
   (per-vector radius range, circular angle code, use the residual sign on
   decode) halves the error with no extra bits. This is the largest
   improvement available.
2. **The golden angle helps, but less than a random rotation.** It beats no
   rotation by about 9 % on outlier-heavy data and trails a random rotation
   by about 17 %. `nqx-core/bench/phi_vs_random.md` reports the same
   direction. The defensible hardware argument is zero rotation state and
   bit-exact determinism, not better quality.
3. **Validate on real KV caches.** All numbers here use synthetic data.
   Llama/Qwen KV tensors (for example via `validate_real_kv.py` in the
   repository root) should decide F1's fix and F4's trade-off before an
   expensive production tape-out.
