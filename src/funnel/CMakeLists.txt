#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "funnel")
#Note that the macros here require this directory added as a subdir of plumed/src
DECLAREPLUMEDMODULE(${module_name}
OFF
SOURCES
FPS.cpp
Funnel.cpp
NEEDS
	core tools colvar bias
DEPENDS
	bias #tools core
	colvar#tools core
)
