#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "eds")
#Note that the macros here require this directory added as a subdir of plumed/src
DECLAREPLUMEDMODULE(${module_name}
OFF
SOURCES
EDS.cpp
NEEDS
	core tools bias
DEPENDS
	bias #tools core
)
