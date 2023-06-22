#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "membranefusion")
set(module_default_${module_name} OFF CACHE INTERNAL "default status of the module ${module_name}")
#Note that the macros here require this directory added as a subdir of plumed/src
option(module_${module_name} "activate module ${module_name}" OFF)
ADDMODULETOKERNEL(${module_name}
SOURCES
FusionPoreExpansionP.cpp
FusionPoreNucleationP.cpp
MemFusionP.cpp
NEEDS
	core reference tools colvar
DEPENDS
	colvar #reference tools core
)
