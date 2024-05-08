include ./Makefile.conf

#Dependency tracking based on https://make.mad-scientist.net/papers/advanced-auto-dependency-generation/#tldr
#this assumes gcc
DEPDIR := .deps
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d
NVCXX = nvc++

NVCCCFLAGS =  -acc -acc=gpu -gpu=cc75 --c++17 -Minfo=all -fast
NVCCLDFLAGS = -shared -acc -acc=gpu -gpu=cc75
OBJS = CoordinationACC.o
ACCOBJS = fastCoord.o

ADDCPPFLAGS=$(PLUMED_INCLUDE)
ADDCLDFLAGS=
CXX=g++

ifeq ($(SOEXT),dylib)
  SONAME_OPTION:=-Xlinker -install_name
elsemake
  SONAME_OPTION:=-Xlinker -soname
endif

all: ACCCoordination.$(SOEXT)

%.o: %.cxx $(DEPDIR)/%.d | $(DEPDIR)
	@echo Compiling object $@
	$(NVCXX) $(NVCCCFLAGS) -c  $(DEPFLAGS) -fPIC $< -o $@

%.o: %.cpp $(DEPDIR)/%.d | $(DEPDIR)
	@echo Compiling object $@
	$(CXX) -c $(PLUMED_INCLUDE)  $(DEPFLAGS) $(CPPFLAGS) $(ADDCPPFLAGS) $(CXXFLAGS) $< -o $@

$(DEPDIR): ; @mkdir -p $@

DEPFILES := $(OBJS:%.o=$(DEPDIR)/%.d) $(ACCOBJS:%.o=$(DEPDIR)/%.d)

$(DEPFILES):
include $(wildcard $(DEPFILES))


libswitchACC.$(SOEXT): $(ACCOBJS)
	@echo Linking $@
	# $(NVCXX) $(NVCCLDFLAGS) $(ADDCLDFLAGS) -Wl,-soname,"$(notdir $@)" -o $@ $^
	$(NVCXX) $(NVCCLDFLAGS) $(ADDCLDFLAGS) -o $@ $^

ACCCoordination.$(SOEXT): $(OBJS) libswitchACC.$(SOEXT)
	@echo Linking $@
	$(CXX) -shared $(ADDCLDFLAGS) -Wl,-soname,"$(notdir $@)" -o $@ $< $(DYNAMIC_LIBS) /u/d/drapetti/scratch/repos/plumed2-dev/plugins/openACCCoord/libswitchACC.so $(PLUMED_KERNEL) 

clean:
	@rm -fv $(ACCOBJS) $(OBJS) ACCCoordination.$(SOEXT) libswitchACC.$(SOEXT)

veryclean: clean
	@echo removing dependency tracking
	@rm -rf .deps

check: all
	$(MAKE) -C regtest
	$(MAKE) -C regtest checkfail
