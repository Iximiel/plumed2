#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "bias")
#Note that the macros here require this directory added as a subdir of plumed/src
DECLAREPLUMEDMODULE(${module_name}
ON
SOURCES
ABMD.cpp
Bias.cpp
BiasValue.cpp
ExtendedLagrangian.cpp
External.cpp
LWalls.cpp
MaxEnt.cpp
MetaD.cpp
MovingRestraint.cpp
PBMetaD.cpp
Restraint.cpp
ReweightBase.cpp
ReweightBias.cpp
ReweightMetad.cpp
ReweightTemperaturePressure.cpp
ReweightWham.cpp
UWalls.cpp
NEEDS
	core tools
DEPENDS
	core #tools
)
