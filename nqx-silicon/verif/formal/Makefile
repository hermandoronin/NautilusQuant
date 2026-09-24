# Formal verification with SymbiYosys (yosys-smtbmc + yices).
#   hsio.sby  4-phase handshake <-> stream bridge: protocol, data integrity,
#             bounded response. Unbounded proof (k-induction).
#   core.sby  core byte-stream valid/ready compliance, idle => ready.
#             Unbounded proof (k-induction) + reachability covers.

SBY ?= sby

.PHONY: all hsio core clean
all: hsio core

hsio:
	$(SBY) -f hsio.sby

core:
	$(SBY) -f core.sby

clean:
	rm -rf hsio_prove hsio_cover core_prove core_cover
