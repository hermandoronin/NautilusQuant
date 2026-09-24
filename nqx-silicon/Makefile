# NQX-S1 silicon: model, RTL checks, simulation, vectors and GDSII builds.

PY  ?= python3
RTL := rtl/nqx_s1_top.sv rtl/nqx_s1_hsio.sv rtl/nqx_s1_core.sv rtl/nqx_s1_cordic.sv rtl/nqx_s1_cordic_iter.sv \
       rtl/nqx_s1_rcode.sv rtl/nqx_s1_rsign.sv rtl/nqx_s1_tcode.sv rtl/nqx_s1_qdec.sv \
       rtl/nqx_s1_div28.sv rtl/nqx_s1_rf.sv

.PHONY: all params check-params lint yosys-check test-model test-rtl test formal vectors \
        gds-ihp gds-gf180 clean

all: check-params lint yosys-check test formal

params:            ## regenerate rtl/nqx_s1_params.vh and rtl/nqx_s1_atan.vh
	$(PY) tools/gen_params.py

check-params:
	$(PY) tools/gen_params.py --check

lint:              ## Verilator -Wall, must be clean
	verilator --lint-only -Wall -Irtl $(RTL) --top-module nqx_s1_top

yosys-check:       ## elaborate with Yosys and fail on any netlist problem
	yosys -q -p "read_verilog -sv -Irtl $(RTL); hierarchy -check -top nqx_s1_top; proc; check -assert"

test-model:        ## unit tests of the bit-accurate model
	$(PY) -m pytest model/tests -q

test-rtl:          ## cocotb: CORDIC unit, core random, pin-level (Icarus by default)
	$(PY) -m pytest verif/test_rtl.py -q

test: test-model test-rtl

formal:            ## SymbiYosys proofs (handshake bridge, core stream protocol)
	$(MAKE) -C verif/formal

vectors:           ## golden test vectors for silicon bring-up
	$(PY) tools/gen_vectors.py --out vectors

gds-ihp:           ## full chip, IHP SG13G2 (LibreLane Chip flow)
	$(MAKE) -C flow/ihp-sg13g2 gds

gds-gf180:         ## full chip, wafer.space GF180MCU slot 0p5x0p5
	$(MAKE) -C flow/wafer-space-gf180 librelane SLOT=0p5x0p5

clean:
	rm -rf verif/sim_build verif/cocotb/results.xml .pytest_cache
	$(MAKE) -C verif/formal clean
