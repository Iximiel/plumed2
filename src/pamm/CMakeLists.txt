#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "pamm")
#Note that the macros here require this directory added as a subdir of plumed/src
DECLAREPLUMEDMODULE(${module_name}
OFF
SOURCES
HBPammHydrogens.cpp
HBPammMatrix.cpp
HBPammObject.cpp
PAMM.cpp
PammObject.cpp
NEEDS
	core tools vesselbase multicolvar adjmat
DEPENDS
	adjmat #multicolvar vesselbase bias tools core
)
