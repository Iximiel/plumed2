#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "crystallization")
#Note that the macros here require this directory added as a subdir of plumed/src
DECLAREPLUMEDMODULE(${module_name}
OFF
SOURCES
BondOrientation.cpp
CubicHarmonicBase.cpp
EnvironmentSimilarity.cpp
Fccubic.cpp
Gradient.cpp
GradientVessel.cpp
InterMolecularTorsions.cpp
LocalSteinhardt.cpp
MoleculeOrientation.cpp
MoleculePlane.cpp
OrientationSphere.cpp
PolymerAngles.cpp
Q3.cpp
Q4.cpp
Q6.cpp
SimpleCubic.cpp
SMAC.cpp
Steinhardt.cpp
Tetrahedral.cpp
VectorMean.cpp
VectorMultiColvar.cpp
VectorSum.cpp
NEEDS
	core tools vesselbase multicolvar
DEPENDS
	multicolvar #vesselbase tools core
)
