#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "vatom")
set(module_default_${module_name} ON CACHE INTERNAL "default status of the module ${module_name}")
#Note that the macros here require this directory added as a subdir of plumed/src
option(module_${module_name} "activate module ${module_name}" ON)
ADDMODULETOKERNEL(${module_name}
SOURCES
ActionWithVirtualAtom.cpp
Center.cpp
FixedAtom.cpp
Ghost.cpp
NEEDS
	core tools
DEPENDS
	core #tools
)
