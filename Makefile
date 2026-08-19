PY ?= python
CONFIG ?= config/v3_core.yaml
OUT ?= results/v3_single.jsonl
RESULTS ?= results/v3_single.jsonl
DATA ?= data

.PHONY: help install test gen run analyze paper check-macros verify all clean reproduce

help:
	@echo "make install    install dependencies"
	@echo "make test       run the test suite"
	@echo "make gen        materialise the synthetic workloads"
	@echo "make run        execute the experiment grid"
	@echo "make analyze    build tables, figures and paper macros"
	@echo "make paper      compile the manuscript to PDF"
	@echo "make check-macros verify every number the paper cites exists"
	@echo "make verify     tests + macro check"
	@echo "make reproduce  gen + run + analyze + paper, end to end"

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

test:
	$(PY) -m pytest -q

gen:
	PYTHONPATH=src $(PY) -m skewbench.cli gen --config $(CONFIG) --data-root $(DATA)

run:
	PYTHONPATH=src $(PY) -m skewbench.cli run --config $(CONFIG) --out $(OUT) --data-root $(DATA)

analyze:
	PYTHONPATH=src:analysis $(PY) analysis/analyze.py $(OUT) results/analysis
	@mkdir -p paper/figures
	@cp -f results/analysis/figures/*.pdf paper/figures/ 2>/dev/null || true

paper: check-macros
	cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
	cd paper && bibtex main >/dev/null 2>&1 || true
	cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
	cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
	@echo "-> paper/main.pdf"

reproduce: gen run analyze paper

check-macros:
	$(PY) analysis/check_macros.py paper

verify: test check-macros
	$(PY) analysis/verify_numbers.py $(RESULTS) paper/numbers.tex
	@echo "-> tests pass, every cited number is defined AND matches the data"

clean:
	rm -rf results/eventlogs results/analysis paper/*.aux paper/*.log \
	       paper/*.bbl paper/*.blg paper/*.out __pycache__
