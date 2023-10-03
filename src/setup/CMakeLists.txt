#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "setup")
#Note that the macros here require this directory added as a subdir of plumed/src
DECLAREPLUMEDMODULE(${module_name}
ON
SOURCES
Load.cpp
Restart.cpp
Units.cpp
NEEDS
	core tools
DEPENDS
	core #tools
)
