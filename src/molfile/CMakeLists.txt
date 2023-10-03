#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "molfile")
string(TOUPPER ${module_name} externalLib)
#Note that the macros here require this directory added as a subdir of plumed/src
option(useExternal_${module_name} "enable search for external ${module_name}, default OFF" OFF)
DECLAREPLUMEDMODULE(${module_name}
ON
SOURCES
crdplugin.cpp
dcdplugin.cpp
gromacsplugin.cpp
pdbplugin.cpp
EXTRA_HEADERS
endianswap.h
fastio.h
Gromacs.h
largefiles.h
libmolfile_plugin.h
molfile_plugin.h
periodic_table.h
readpdb.h
vmdplugin.h
)

#  --enable-external-molfile-plugins
#                          enable search for external molfile_plugins, default:
#                          yes
if (useExternal_${module_name})
    find_package(MolfilePlugins)
    if(MolfilePlugins_FOUND)
        target_compile_definitions(${module_name} PUBLIC __PLUMED_HAS_EXTERNAL_MOLFILE_PLUGINS=1)
        target_link_libraries(${module_name} PUBLIC MolfilePlugins:MolfilePlugins)
    endif()
endif (useExternal_${module_name})
#The following lines should mimic the following line
#CPPFLAGS+=-DSTATIC_PLUGIN  -DVMDPLUGIN=molfile_$(patsubst %.o,%,$@)
target_compile_definitions(${module_name} PRIVATE STATIC_PLUGIN)
get_target_property(MY_PROJECT_SOURCES ${module_name} SOURCES)
foreach(sourcefile ${MY_PROJECT_SOURCES})
    string(REPLACE ".cpp" "" simpleName ${sourcefile})
    set_source_files_properties(${sourcefile}
        PROPERTIES COMPILE_DEFINITIONS VMDPLUGIN=molfile_${simpleName})
    
endforeach(sourcefile in ${MY_PROJECT_SOURCES})

#this should propagate the compile definition to the main
target_compile_definitions(${module_name} PUBLIC __PLUMED_HAS_MOLFILE_PLUGINS=1)
