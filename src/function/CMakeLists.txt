#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "function")
#Note that the macros here require this directory added as a subdir of plumed/src
DECLAREPLUMEDMODULE(${module_name}
ON
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
