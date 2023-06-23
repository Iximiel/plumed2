cmake_minimum_required(VERSION 3.20.2)

project(Plumed2 LANGUAGES CXX)

set(PLUMED_VERSION_SHORT "2.10")
set(PLUMED_VERSION_LONG "2.10.0-dev")
#set(PLUMED_VERSION_GIT "23e754fd4-dirty")
execute_process(
        COMMAND git describe --long --dirty --always
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        OUTPUT_VARIABLE PLUMED_VERSION_GIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(PLUMED_VERSION_MAJOR 2)
set(PLUMED_VERSION_MINOR 10)
set(PLUMED_VERSION_PATCH 0)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS None Debug Release RelWithDebInfo MinSizeRel)

#as now CMake is only experimental, it can be useful if you use vscode or other
#IDE that have an interface with it, but we are not still confident that it is ready
#for installation purposes
#set(CMAKE_SKIP_INSTALL_RULES YES)
option(risk_accepted "CMake is experimental use at your own risk" OFF)
if(NOT risk_accepted)
   message(FATAL_ERROR "You must accept the risk to use the experimental cmake compilation: -Drisk_accepted=ON")
endif(NOT risk_accepted)
include(GNUInstallDirs)
set(CMAKE_INSTALL_LIBDIR lib CACHE PATH "Object code libraries")
#set(CMAKE_INSTALL_DOCDIR ${CMAKE_INSTALL_DATAROOTDIR}/doc/${program_name})


add_subdirectory(src)
add_subdirectory(scripts)
add_subdirectory(fortran)
#add_subdirectory(vim)

install(DIRECTORY patches
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/${program_name})
