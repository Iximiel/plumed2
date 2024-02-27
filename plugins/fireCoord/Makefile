SOEXT := so

all: FireCoordination.$(SOEXT)

%.$(SOEXT): %.cpp
	plumed mklib $<

clean:
	@rm -fv FireCoordination.$(SOEXT)

check: all
	$(MAKE) -C regtest
	$(MAKE) -C regtest checkfail
