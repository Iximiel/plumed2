#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "isdb")
#Note that the macros here require this directory added as a subdir of plumed/src
DECLAREPLUMEDMODULE(${module_name}
ON
SOURCES
Caliber.cpp
CS2Backbone.cpp
EMMI.cpp
FretEfficiency.cpp
Jcoupling.cpp
MetainferenceBase.cpp
Metainference.cpp
NOE.cpp
PRE.cpp
RDC.cpp
Rescale.cpp
SAXS.cpp
Select.cpp
Selector.cpp
NEEDS
	bias colvar core reference tools function
DEPENDS
	bias #tools core
	colvar #reference tools
	function #reference
)
