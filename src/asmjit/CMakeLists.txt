#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "asmjit")
#Note that the macros here require this directory added as a subdir of plumed/src
set(module_${module_name} ON CACHE INTERNAL "always active module ${module_name}")
ADDMODULETOKERNEL(${module_name}
SOURCES
arch.cpp
assembler.cpp
codebuilder.cpp
codecompiler.cpp
codeemitter.cpp
codeholder.cpp
constpool.cpp
cpuinfo.cpp
func.cpp
globals.cpp
inst.cpp
logging.cpp
operand.cpp
osutils.cpp
regalloc.cpp
runtime.cpp
string.cpp
utils.cpp
vmem.cpp
x86assembler.cpp
x86builder.cpp
x86compiler.cpp
x86inst.cpp
x86instimpl.cpp
x86internal.cpp
x86logging.cpp
x86operand.cpp
x86operand_regs.cpp
x86regalloc.cpp
zone.cpp
EXTRA_HEADERS
arm.h
asmjit_apibegin.h
asmjit_apiend.h
asmjit_build.h
asmjit.h
base.h
misc_p.h
moved_string.h
regalloc_p.h
simdtypes.h
x86emitter.h
x86globals.h
x86.h
x86instimpl_p.h
x86internal_p.h
x86logging_p.h
x86misc.h
x86regalloc_p.h
)
ADDMODULENEEDS(${module_name}
	
)
ADDMODULEDEPENDENCIES(${module_name}
	
)
