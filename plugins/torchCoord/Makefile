SOEXT := so

all: TorchCoordination.$(SOEXT)

%.$(SOEXT): %.cpp
	plumed mklib $<

clean:
	@rm -fv TorchCoordination.$(SOEXT)

check: all
	$(MAKE) -C regtest
	$(MAKE) -C regtest checkfail
