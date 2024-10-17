.PHONY: clean check check_standalone check_python all

all: pycv_here

pycv_here: src/*.cpp src/*.h src/pycv/*.py
	@python3 -m pip install .
	@which python3 > $@

clean:
	@python3 -m pip uninstall pycv

check_standalone:
	$(MAKE) -C regtest testclean
	$(MAKE) -C regtest checkfail

check_python:
	python3 -m pytest

check: check_standalone check_python
