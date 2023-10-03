#[=======================================================================[.rst:
FindMolfilePlugins
-------

Finds the MolfilePlugins library.
Tries to use pkgconfig or search in the vmd installation folder (defined by the variable USR_VMD_DIR)
in the subdirectories  lib/plugins/include for molfile_plugin.h and lib/plugins/LINUXAMD64/molfile  for the
shared obects like pdbplugin.so, as now search only for linux ".so" 

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``MolfilePlugins::MolfilePlugins``
  The MolfilePlugins library
``MolfilePlugins::nameplugin``
  The various plugins


Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``MolfilePlugins_FOUND``
  True if the system has the MolfilePlugins library.
``MolfilePlugins_VERSION``
  The version of the MolfilePlugins library which was found.
``MolfilePlugins_INCLUDE_DIRS``
  Include directories needed to use MolfilePlugins.
``MolfilePlugins_LIBRARIES``
  Libraries needed to link to MolfilePlugins.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``MolfilePlugins_INCLUDE_DIR``
  The directory containing ``molfile_plugin.h``.
``MolfilePlugins_nameplugin_LIBRARY``
  The path to the the various plugins found in the plugin directory library.

#]=======================================================================]

find_package(PkgConfig)
pkg_check_modules(PC_Molfile QUIET MolfilePlugins)
#-D MOLFILE_INCLUDE_DIR=path

find_path(MolfilePlugins_INCLUDE_DIR
  NAMES molfile_plugin.h
  PATHS ${PC_MolfilePlugins_INCLUDE_DIRS} ${USR_MOLFILE_INCLUDE_DIR} ${USR_VMD_DIR}
  PATH_SUFFIXES
    include #for USR_MOLFILE_INCLUDE_DIR
    lib/plugins/include #for vmd path
)

find_path(LIBMolfilePlugins_INCLUDE_DIR
  NAMES libmolfile_plugin.h
  PATHS ${PC_MolfilePlugins_INCLUDE_DIRS} ${USR_MOLFILE_INCLUDE_DIR} ${USR_VMD_DIR}
  PATH_SUFFIXES
    include #for USR_MOLFILE_INCLUDE_DIR
    lib/plugins/include #for vmd path
)

#pdb is the most probable plugin to exist, this is  needed to get the others
find_library(MolfilePlugins_pdbplugin_LIBRARY
  NAMES pdbplugin.so
  PATHS ${PC_MolfilePlugins_LIBRARY_DIRS}  ${USR_VMD_DIR}
  PATH_SUFFIXES
    lib/plugins/LINUXAMD64/molfile #for vmd path
)

get_filename_component(MolfilePlugins_LIBRARY_PATH ${MolfilePlugins_pdbplugin_LIBRARY} DIRECTORY CACHE)
file(GLOB MolfilePlugins_PLUGIN_OBJECTS ${MolfilePlugins_LIBRARY_PATH}/*plugin.so)
foreach(singlePlugin ${MolfilePlugins_PLUGIN_OBJECTS})
    get_filename_component(pluginName ${singlePlugin} NAME_WE)
    find_library(MolfilePlugins_${pluginName}_LIBRARY
        NAMES ${pluginName}.so
        PATHS ${MolfilePlugins_LIBRARY_PATH}
    )
endforeach(singlePlugin ${MolfilePlugins_PLUGIN_OBJECTS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MolfilePlugins
  FOUND_VAR MolfilePlugins_FOUND
  REQUIRED_VARS
    MolfilePlugins_pdbplugin_LIBRARY
    MolfilePlugins_INCLUDE_DIR
    LIBMolfilePlugins_INCLUDE_DIR
  VERSION_VAR MolfilePlugins_VERSION
)

if(MolfilePlugins_FOUND AND NOT TARGET MolfilePlugins::MolfilePlugins)
  add_library(MolfilePlugins::MolfilePlugins INTERFACE IMPORTED)
  set_target_properties(MolfilePlugins::MolfilePlugins PROPERTIES
    INTERFACE_COMPILE_OPTIONS "${PC_MolfilePlugins_CFLAGS_OTHER}"
    INTERFACE_INCLUDE_DIRECTORIES "${MolfilePlugins_INCLUDE_DIR}" "${LIBMolfilePlugins_INCLUDE_DIR}"
  )
  foreach(singlePlugin ${MolfilePlugins_PLUGIN_OBJECTS})
    get_filename_component(pluginName ${singlePlugin} NAME_WE)
    add_library(MolfilePlugins::${pluginName} UNKNOWN IMPORTED)
    set_target_properties(MolfilePlugins::${pluginName} PROPERTIES
        IMPORTED_LOCATION  "${MolfilePlugins_${pluginName}_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${MolfilePlugins_INCLUDE_DIR}"
    )
    target_link_libraries(MolfilePlugins::MolfilePlugins INTERFACE MolfilePlugins::${pluginName})
    find_library(MolfilePlugins_${pluginName}_LIBRARY
        NAMES ${pluginName}.so
        PATHS ${MolfilePlugins_LIBRARY_PATH}
    )
    mark_as_advanced(MolfilePlugins_${pluginName}_LIBRARY)
  endforeach(singlePlugin ${MolfilePlugins_PLUGIN_OBJECTS})
endif()

mark_as_advanced(
  MolfilePlugins_INCLUDE_DIR
  MolfilePlugins_LIBRARY
)
