#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "fisst")
#Note that the macros here require this directory added as a subdir of plumed/src
DECLAREPLUMEDMODULE(${module_name}
OFF
SOURCES
FISST.cpp
legendre_rule_fast.cpp
NEEDS
	core tools bias
DEPENDS
	bias #tools core
)
