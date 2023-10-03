#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "annfunc")
#Note that the macros here require this directory added as a subdir of plumed/src
DECLAREPLUMEDMODULE(${module_name}
OFF
SOURCES
ANN.cpp
NEEDS
	core function
DEPENDS
	function #reference tools core lepton
)
