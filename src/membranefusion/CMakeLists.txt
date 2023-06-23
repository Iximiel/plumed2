#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "membranefusion")
#Note that the macros here require this directory added as a subdir of plumed/src
DECLAREPLUMEDMODULE(${module_name}
OFF
SOURCES
FusionPoreExpansionP.cpp
FusionPoreNucleationP.cpp
MemFusionP.cpp
NEEDS
	core reference tools colvar
DEPENDS
	colvar #reference tools core
)
