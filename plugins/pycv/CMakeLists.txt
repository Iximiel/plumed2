cmake_minimum_required(VERSION 3.15...3.27)
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 17)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "pybind11 found: ${pybind11_VERSION}")

exec_program(plumed
ARGS info --include-dir
OUTPUT_VARIABLE PLUMED_INCLUDE_DIR
)
message(STATUS "Plumed include dir: ${PLUMED_INCLUDE_DIR}")

################################################################################
################################the pycv library################################
################################################################################
#TODO: removhe the "lib" prefix
add_library(PythonCVInterface SHARED src/ActionWithPython.cpp  src/PythonCVInterface.cpp  src/PythonFunction.cpp)
target_link_libraries(PythonCVInterface PRIVATE pybind11::embed)
target_link_libraries(PythonCVInterface PUBLIC plumedKernel)
target_include_directories(PythonCVInterface PUBLIC src ${PLUMED_INCLUDE_DIR}/plumed)

install(TARGETS PythonCVInterface DESTINATION .)

################################################################################
############################The pvCV companion module###########################
################################################################################

pybind11_add_module(plumedCommunications src/PlumedPythonEmbeddedModule.cpp)
#python_add_library(plumedCommunications MODULE PlumedPythonEmbeddedModule.cpp WITH_SOABI)
# target_link_libraries(plumedCommunications PRIVATE pybind11::headers)
target_link_libraries(plumedCommunications PUBLIC plumedKernel)
target_link_libraries(plumedCommunications PUBLIC PythonCVInterface)
target_include_directories(plumedCommunications PUBLIC src ${PLUMED_INCLUDE_DIR}/plumed)

# The install directory is the output (wheel) directory
install(TARGETS plumedCommunications DESTINATION .)
