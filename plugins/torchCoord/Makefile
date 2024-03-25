SOEXT := so

all: TorchCoordination.$(SOEXT)

#-w suppresses the warnings
%.$(SOEXT): %.cpp
	PLUMED_MKLIB_CFLAGS="-w --std=c++17" PLUMED_MKLIB_LDFLAGS="-lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda" plumed mklib $<

clean:
	@rm -fv TorchCoordination.$(SOEXT)

check: all
	$(MAKE) -C regtest
	$(MAKE) -C regtest checkfail
