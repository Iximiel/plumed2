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

python_add_library(_core MODULE PlumedPythonEmbeddedModule.cpp WITH_SOABI)
target_link_libraries(_core PRIVATE pybind11::headers)
target_link_libraries(_core PUBLIC plumedKernel)

# This is passing in the version as a define just as an example
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

# The install directory is the output (wheel) directory
install(TARGETS _core DESTINATION plumedCommunications)

#the pycv library
add_library(PythonCVInterface SHARED ActionWithPython.cpp  PythonCVInterface.cpp  PythonFunction.cpp)
target_link_libraries(PythonCVInterface PRIVATE pybind11::embed)
target_link_libraries(PythonCVInterface PUBLIC plumedKernel)

target_link_libraries(_core PUBLIC PythonCVInterface)

target_include_directories(PythonCVInterface PUBLIC ${PLUMED_INCLUDE_DIR}/plumed)
target_include_directories(_core PUBLIC ${PLUMED_INCLUDE_DIR}/plumed)

#TBD
#install(TARGETS PythonCVInterface DESTINATION )
