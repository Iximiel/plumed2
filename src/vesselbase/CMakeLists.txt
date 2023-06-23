#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "vesselbase")
#Note that the macros here require this directory added as a subdir of plumed/src
DECLAREPLUMEDMODULE(${module_name}
ON
SOURCES
ActionWithAveraging.cpp
ActionWithInputVessel.cpp
ActionWithVessel.cpp
AltMin.cpp
AveragingVessel.cpp
Between.cpp
BridgeVessel.cpp
FunctionVessel.cpp
Highest.cpp
Histogram.cpp
LessThan.cpp
Lowest.cpp
Max.cpp
Mean.cpp
Min.cpp
Moments.cpp
MoreThan.cpp
OrderingVessel.cpp
ShortcutVessel.cpp
StoreDataVessel.cpp
Sum.cpp
ValueVessel.cpp
Vessel.cpp
VesselRegister.cpp
NEEDS
	core tools bias analysis
DEPENDS
	analysis #bias tools core
)
