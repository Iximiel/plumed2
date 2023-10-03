#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "piv")
#Note that the macros here require this directory added as a subdir of plumed/src
DECLAREPLUMEDMODULE(${module_name}
OFF
SOURCES
PIV.cpp
NEEDS
	core tools bias colvar
DEPENDS
	colvar #tools core
	bias #tools core
)
