#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "s2cm")
#Note that the macros here require this directory added as a subdir of plumed/src
DECLAREPLUMEDMODULE(${module_name}
OFF
SOURCES
S2ContactModel.cpp
NEEDS
	core tools colvar
DEPENDS
	colvar #tools  core
)
