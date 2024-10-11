cmake_minimum_required(VERSION 3.20)
project(PYCV VERSION 0.1.0 LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 17)

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
message(STATUS "pybind11 found: ${pybind11_VERSION}")

exec_program(plumed
ARGS info --include-dir
OUTPUT_VARIABLE PLUMED_INCLUDE_DIR
)
message(STATUS "Plumed include dir: ${PLUMED_INCLUDE_DIR}")

pybind11_add_module(plumedCommunications PlumedPythonEmbeddedModule.cpp)
add_library(PythonCVInterface SHARED ActionWithPython.cpp  PythonCVInterface.cpp  PythonFunction.cpp)
target_link_libraries(PythonCVInterface PUBLIC pybind11::embed plumedKernel)
target_link_libraries(plumedCommunications PUBLIC plumedKernel PythonCVInterface)

target_include_directories(PythonCVInterface PUBLIC ${PLUMED_INCLUDE_DIR}/plumed)
target_include_directories(plumedCommunications PUBLIC ${PLUMED_INCLUDE_DIR}/plumed)