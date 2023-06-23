#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "reference")
#Note that the macros here require this directory added as a subdir of plumed/src
DECLAREPLUMEDMODULE(${module_name}
ON
SOURCES
ArgumentOnlyDistance.cpp
Direction.cpp
DotProductDistance.cpp
DRMSD.cpp
EuclideanDistance.cpp
FakeFrame.cpp
IntermolecularDRMSD.cpp
IntramolecularDRMSD.cpp
MahalanobisDistance.cpp
MetricRegister.cpp
MultiDomainRMSD.cpp
NormalizedEuclideanDistance.cpp
OptimalRMSD.cpp
ReferenceArguments.cpp
ReferenceAtoms.cpp
ReferenceConfiguration.cpp
ReferenceValuePack.cpp
RMSDBase.cpp
SimpleRMSD.cpp
SingleDomainRMSD.cpp
NEEDS
	core tools
DEPENDS
	core #tools
)
