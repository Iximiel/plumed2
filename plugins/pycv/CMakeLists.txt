cmake_minimum_required(VERSION 3.15...3.27)
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 17)

#Finding necessary packages
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(plumed REQUIRED plumedInternals)

#Finding optionals things
if("-D__PLUMED_HAS_MPI=1" IN_LIST plumed_CFLAGS)
    find_package(MPI REQUIRED)
endif()
if(MPI_CXX_FOUND)
    list(APPEND extraLibs MPI::MPI_CXX)
endif()

if("-fopenmp" IN_LIST plumed_STATIC_LDFLAGS_OTHER)
    find_package(OpenMP REQUIRED)
endif()
if(OpenMP_CXX_FOUND )
    list(APPEND extraLibs OpenMP::OpenMP_CXX)
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fno-gnu-unique USE_NO_GNU_UNIQUE)
if(USE_NO_GNU_UNIQUE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-gnu-unique")
endif()

# plumed_STATIC_LDFLAGS_OTHER:INTERNAL=-rdynamic;-Wl,-Bsymbolic;-fopenmp
#-rdynamic is automatically set by cmake, as -fPIC

################################################################################
################################the pycv library################################
################################################################################

add_library(PythonCVInterface SHARED src/ActionWithPython.cpp  src/PythonCVInterface.cpp  src/PythonFunction.cpp)
#public, so they trickle down to the python module
target_compile_definitions(PythonCVInterface PUBLIC ${plumed_CFLAGS})
target_compile_options(PythonCVInterface PUBLIC ${PLUMED_CXX_FLAGS})
target_include_directories(PythonCVInterface PUBLIC src ${plumed_INCLUDEDIR})
#######################################################################
# uncommenting this brings problems since some symbols here are needed
# by the python module even if it should be the correct setting...
# https://gcc.gnu.org/wiki/Visibility could be a starting point
#######################################################################
# target_compile_options(PythonCVInterface PRIVATE -fvisibility=hidden)
target_link_libraries(PythonCVInterface PRIVATE pybind11::embed)
target_link_libraries(PythonCVInterface PUBLIC plumedKernel ${extraLibs})
#this removes the "lib" prefix
set_target_properties(PythonCVInterface PROPERTIES PREFIX "")

install(TARGETS PythonCVInterface DESTINATION pycv)

################################################################################
############################The pvCV companion module###########################
################################################################################

pybind11_add_module(plumedCommunications src/PlumedPythonEmbeddedModule.cpp)
target_link_libraries(plumedCommunications PRIVATE pybind11::headers)
target_link_libraries(plumedCommunications PUBLIC PythonCVInterface)

# The install directory is the output (wheel) directory
install(TARGETS plumedCommunications DESTINATION .)
