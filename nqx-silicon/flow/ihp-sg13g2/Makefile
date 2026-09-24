# NQX-S1 full chip on IHP SG13G2.
#
#   make gds          # PDK via ciel (if missing) + LibreLane Chip flow -> final/
#   make gds PDK_ROOT=/path/to/IHP-Open-PDK   # use an existing PDK checkout
#
# The PDK revision is the one LibreLane 3.0.14 is qualified with.

PDK        ?= ihp-sg13g2
PDK_COMMIT ?= c4b8b4e5e7a05f375cca3815d51b3a37721fbf5c
PDK_ROOT   ?= $(HOME)/.ciel
LIBRELANE  ?= librelane
RUN_TAG    ?= $(shell date +%Y%m%d_%H%M%S)

.PHONY: pdk gds clean

pdk:
	@if [ ! -d "$(PDK_ROOT)/$(PDK)" ]; then \
	    ciel enable --pdk-root "$(PDK_ROOT)" --pdk-family ihp-sg13 $(PDK_COMMIT); \
	fi

gds: pdk
	KLAYOUT_PATH="$(PDK_ROOT)/$(PDK)/libs.tech/klayout" \
	$(LIBRELANE) --manual-pdk --pdk-root "$(PDK_ROOT)" --pdk $(PDK) \
	    --run-tag $(RUN_TAG) --save-views-to final config.yaml

clean:
	rm -rf runs final
