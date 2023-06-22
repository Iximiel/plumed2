#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "mapping")
set(module_default_${module_name} ON CACHE INTERNAL "default status of the module ${module_name}")
#Note that the macros here require this directory added as a subdir of plumed/src
option(module_${module_name} "activate module ${module_name}" ON)
ADDMODULETOKERNEL(${module_name}
SOURCES
AdaptivePath.cpp
Mapping.cpp
PathBase.cpp
Path.cpp
PathReparameterization.cpp
PathTools.cpp
PCAVars.cpp
PropertyMap.cpp
SpathVessel.cpp
TrigonometricPathVessel.cpp
ZpathVessel.cpp
NEEDS
	core tools cltools reference vesselbase
DEPENDS
	cltools #tools core
	reference #tools core
	vesselbase #tools core
)
