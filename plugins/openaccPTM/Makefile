include ./Makefile.conf

#Dependency tracking based on https://make.mad-scientist.net/papers/advanced-auto-dependency-generation/#tldr
#this assumes gcc
DEPDIR := .deps
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d
#tested with nvcc with :"Build cuda_11.7.r11.7/compiler.31442593_0"
#-dc adds relocatable device code 
#-dlto Perform link-time optimization of device code. 
OBJS = Distance.o

ADDCPPFLAGS=$(PLUMED_INCLUDE) -D__PLUMED_HAS_OPENACC=1 -acc -acc="gpu"
ADDCLDFLAGS=-acc -static-nvidia

ifeq ($(SOEXT),dylib)
  SONAME_OPTION:=-Xlinker -install_name
else
  SONAME_OPTION:=-Xlinker -soname
endif

PLUGIN=plumedOpenACC.$(SOEXT)

NVCXX ?= nvc++
NVLDSHARED = $(LDSHARED:$(CXX)=$(NVCXX))
all: $(PLUGIN)

%.o: %.cpp $(DEPDIR)/%.d | $(DEPDIR)
	@echo Compiling object $@
	$(NVCXX) -c $(DEPFLAGS) $(CPPFLAGS) $(ADDCPPFLAGS) $(CXXFLAGS) $< -o $@

$(DEPDIR): ; @mkdir -p $@

DEPFILES := $(OBJS:%.o=$(DEPDIR)/%.d)
$(DEPFILES):
include $(wildcard $(DEPFILES))

$(PLUGIN): $(OBJS)
	@echo Linking $@
	$(NVLDSHARED)  $(ADDCLDFLAGS) $(SONAME_OPTION),"$(notdir $@)" $(DYNAMIC_LIBS) $(PLUMED_KERNEL) -o $@ $^

clean:
	@rm -fv $(OBJS) $(PLUGIN)
	@rm -rf .deps

check: all
	$(MAKE) -C regtest
	$(MAKE) -C regtest checkfail
