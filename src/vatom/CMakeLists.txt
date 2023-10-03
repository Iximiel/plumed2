#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "vatom")
#Note that the macros here require this directory added as a subdir of plumed/src
DECLAREPLUMEDMODULE(${module_name}
ON
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
