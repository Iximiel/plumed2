#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "annfunc")
#Note that the macros here require this directory added as a subdir of plumed/src
option(module_${module_name} "activate module ${module_name}" OFF)
ADDMODULETOKERNEL(${module_name}
SOURCES
ANN.cpp
NEEDS
	core function
DEPENDS
	function #reference tools core lepton
)
