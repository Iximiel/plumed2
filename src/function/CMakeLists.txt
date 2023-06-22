#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "function")
set(module_default_${module_name} ON CACHE INTERNAL "default status of the module ${module_name}")
#Note that the macros here require this directory added as a subdir of plumed/src
option(module_${module_name} "activate module ${module_name}" ON)
ADDMODULETOKERNEL(${module_name}
SOURCES
Combine.cpp
Custom.cpp
Ensemble.cpp
FuncPathGeneral.cpp
FuncPathMSD.cpp
FuncSumHills.cpp
Function.cpp
LocalEnsemble.cpp
Piecewise.cpp
Sort.cpp
Stats.cpp
Target.cpp
NEEDS
	core reference tools lepton
DEPENDS
	reference #tools core
	#tools #lepton
)
