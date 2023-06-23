#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "opes")
#Note that the macros here require this directory added as a subdir of plumed/src
DECLAREPLUMEDMODULE(${module_name}
OFF
SOURCES
ECVcustom.cpp
ECVlinear.cpp
ECVmultiThermalBaric.cpp
ECVmultiThermal.cpp
ECVumbrellasFile.cpp
ECVumbrellasLine.cpp
ExpansionCVs.cpp
OPESexpanded.cpp
OPESmetad.cpp
NEEDS
	bias core tools
DEPENDS
	bias #tools core
)
