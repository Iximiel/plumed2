#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "dimred")
#Note that the macros here require this directory added as a subdir of plumed/src
DECLAREPLUMEDMODULE(${module_name}
OFF
SOURCES
ClassicalMultiDimensionalScaling.cpp
DimensionalityReductionBase.cpp
OutputPCAProjections.cpp
PCA.cpp
ProjectNonLandmarkPoints.cpp
SketchMapBase.cpp
SketchMapConjGrad.cpp
SketchMap.cpp
SketchMapPointwise.cpp
SketchMapRead.cpp
SketchMapSmacof.cpp
SMACOF.cpp
SmacoffMDS.cpp
NEEDS
	core tools reference gridtools analysis
DEPENDS
	#core #tools 
	gridtools #analysis bias reference tools core
)
