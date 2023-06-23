#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "drr")
#Note that the macros here require this directory added as a subdir of plumed/src
DECLAREPLUMEDMODULE(${module_name}
OFF
SOURCES
colvar_UIestimator.cpp
DRR.cpp
drrtool.cpp
DynamicReferenceRestraining.cpp
NEEDS
	core tools bias cltools config
DEPENDS
	bias# tools core
	cltools #config
)
