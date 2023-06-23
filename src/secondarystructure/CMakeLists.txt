#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "secondarystructure")
#Note that the macros here require this directory added as a subdir of plumed/src
DECLAREPLUMEDMODULE(${module_name}
ON
SOURCES
AlphaRMSD.cpp
AntibetaRMSD.cpp
ParabetaRMSD.cpp
SecondaryStructureRMSD.cpp
NEEDS
	core tools reference vesselbase
DEPENDS
	#reference#tools core
	vesselbase #analisys tools core reference
)
