"""NQX-S1 host driver and golden-vector runner.

Runs unchanged on CPython and on MicroPython (RP2040 / RP2350 boards such as
the Tiny Tapeout demo board or a Raspberry Pi Pico wired to the chip).

    transport = PinTransport(...)            # on the MCU, see nqx_pins.py
    chip = NQXS1(transport)
    print(hex(chip.csr_read(0x00)))          # 0x4e515831 = "NQX1"
    ok, failed = run_vector_file(chip, "04_encode_decode.txt")

On a PC, ModelTransport runs the same driver against the bit-accurate model,
which checks the vector files and this driver in CI.
"""

CSR_ID = 0x00
CSR_STATUS = 0x04
OP_ENC = 0x60
OP_DEC = 0x61
OP_SYNC = 0x70
OP_CSRW = 0x71
OP_CSRR = 0x72


class Transport:
    """Byte pipe to the chip: write() never blocks on output, read(n) waits."""

    def write(self, data):
        raise NotImplementedError

    def read(self, n):
        raise NotImplementedError


class ModelTransport(Transport):
    """CPython only: loop the byte stream through model/nqx_s1 (S1Core)."""

    def __init__(self, core=None):
        from nqx_s1 import S1Core

        self.core = core or S1Core()
        self.pending = bytearray()

    def write(self, data):
        self.pending += self.core.run(bytes(data))

    def read(self, n):
        out = bytes(self.pending[:n])
        del self.pending[:n]
        if len(out) != n:
            raise TimeoutError("model produced %d of %d bytes" % (len(out), n))
        return out


class NQXS1:
    def __init__(self, transport):
        self.t = transport

    def transact(self, data, n_out):
        self.t.write(data)
        return self.t.read(n_out) if n_out else b""

    def csr_read(self, addr):
        b = self.transact(bytes([OP_CSRR, addr]), 4)
        return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)

    def csr_write(self, addr, value):
        self.t.write(bytes([OP_CSRW, addr, value & 0xFF, (value >> 8) & 0xFF,
                            (value >> 16) & 0xFF, (value >> 24) & 0xFF]))

    def sync(self):
        return self.transact(bytes([OP_SYNC]), 1) == b"\xa5"

    def encode(self, vector, packet_bytes=70):
        data = bytearray([OP_ENC])
        for v in vector:
            data.append(v & 0xFF)
            data.append((v >> 8) & 0xFF)
        return self.transact(bytes(data), packet_bytes)

    def decode(self, packet, dim=128):
        raw = self.transact(bytes([OP_DEC]) + bytes(packet), 2 * dim)
        out = []
        for i in range(0, len(raw), 2):
            v = raw[i] | (raw[i + 1] << 8)
            out.append(v - 0x10000 if v & 0x8000 else v)
        return out


def _parse(lines):
    w, r = bytearray(), bytearray()
    for line in lines:
        line = line.strip()
        if not line or line[0] == "#":
            continue
        if line == ".":
            yield bytes(w), bytes(r)
            w, r = bytearray(), bytearray()
        elif line[0] == "W":
            w += bytes.fromhex(line[2:])
        elif line[0] == "R":
            r += bytes.fromhex(line[2:])


def run_vector_file(chip, path, verbose=False):
    """Replay one golden-vector file. Returns (passed, failed) transaction counts.

    The chip must be freshly reset before each file.
    """
    passed = failed = 0
    with open(path) as f:
        for n, (w, r) in enumerate(_parse(f)):
            got = chip.transact(w, len(r))
            if got == r:
                passed += 1
            else:
                failed += 1
                if verbose:
                    first = next(i for i in range(len(r)) if got[i] != r[i])
                    print("tx %d: mismatch at byte %d: got %s want %s"
                          % (n, first, got[first:first + 8].hex(), r[first:first + 8].hex()))
    return passed, failed
