include ./Makefile.conf

#Dependency tracking based on https://make.mad-scientist.net/papers/advanced-auto-dependency-generation/#tldr
#this assumes gcc
DEPDIR := .deps
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d
NVCXX = nvc++
ACCFLAGS=-acc -acc=gpu -gpu=cc75
NVCCCFLAGS =  $(ACCFLAGS) --c++17 -Minfo=all -fast -Werror --diag_suppress probable_guiding_friend
NVCCLDFLAGS = -shared $(ACCFLAGS)
OBJS = CoordinationACC.o
ACCOBJS = 

ADDCPPFLAGS=$(PLUMED_INCLUDE)
ADDCLDFLAGS=
CXX=g++

ifeq ($(SOEXT),dylib)
  SONAME_OPTION:=-Xlinker -install_name
elsemake
  SONAME_OPTION:=-Xlinker -soname
endif

all: ACCCoordination.$(SOEXT)

$(DEPDIR): ; @mkdir -p $@

DEPFILES := $(OBJS:%.o=$(DEPDIR)/%.d) $(ACCOBJS:%.o=$(DEPDIR)/%.d)

$(DEPFILES):
include $(wildcard $(DEPFILES))

%.o: %.cpp $(DEPDIR)/%.d | $(DEPDIR)
	@echo Compiling object $@
	$(NVCXX) $(NVCCCFLAGS) -c $(PLUMED_INCLUDE) $(DEPFLAGS) $(CPPFLAGS) $(ADDCPPFLAGS) $(CXXFLAGS) -fPIC $< -o $@

ACCCoordination.$(SOEXT): $(OBJS)
	@echo Linking $@
	$(NVCXX) $(NVCCLDFLAGS) $(ADDCLDFLAGS) -shared -Wl,-soname,"$(notdir $@)" -o $@ $< $(NVCCDYNAMIC_LIBS) $(PLUMED_KERNEL) 

clean:
	@rm -fv $(ACCOBJS) $(OBJS) ACCCoordination.$(SOEXT)

veryclean: clean
	@echo removing dependency tracking
	@rm -rf $(DEPDIR)

check: all
	$(MAKE) -C regtest
	$(MAKE) -C regtest checkfail
