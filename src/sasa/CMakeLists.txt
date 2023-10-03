#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "sasa")
#Note that the macros here require this directory added as a subdir of plumed/src
DECLAREPLUMEDMODULE(${module_name}
OFF
SOURCES
Sasa.cpp
sasa_HASEL.cpp
sasa_LCPO.cpp
NEEDS
	core tools config  
DEPENDS
	core #tools #config  
)
