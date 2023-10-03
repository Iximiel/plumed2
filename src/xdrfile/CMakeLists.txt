#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "xdrfile")
#Note that the macros here require this directory added as a subdir of plumed/src
DECLAREPLUMEDMODULE(${module_name}
ON
SOURCES
xdrfile.cpp
xdrfile_trr.cpp
xdrfile_xtc.cpp
)
