#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "mapping")
#Note that the macros here require this directory added as a subdir of plumed/src
DECLAREPLUMEDMODULE(${module_name}
ON
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
