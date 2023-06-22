#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "adjmat")
set(module_default_${module_name} OFF CACHE INTERNAL "default status of the module ${module_name}")
#Note that the macros here require this directory added as a subdir of plumed/src
option(module_${module_name} "activate module ${module_name}" OFF)
ADDMODULETOKERNEL(${module_name}
SOURCES
ActionWithInputMatrix.cpp
AdjacencyMatrixBase.cpp
AdjacencyMatrixVessel.cpp
AlignedMatrixBase.cpp
ClusterAnalysisBase.cpp
ClusterDiameter.cpp
ClusterDistribution.cpp
ClusteringBase.cpp
ClusterProperties.cpp
ClusterSize.cpp
ClusterWithSurface.cpp
ContactAlignedMatrix.cpp
ContactMatrix.cpp
DFSClustering.cpp
DumpGraph.cpp
HbondMatrix.cpp
MatrixColumnSums.cpp
MatrixRowSums.cpp
OutputCluster.cpp
SMACMatrix.cpp
Sprint.cpp
TopologyMatrix.cpp
NEEDS
	core tools vesselbase multicolvar
DEPENDS
	#core #tools depends on
	#tools #vesselbase depends on
	#vesselbase # multicolvar depends on
	multicolvar
)
