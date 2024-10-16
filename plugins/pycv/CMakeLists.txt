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

exec_program(plumed
ARGS info --configuration
OUTPUT_VARIABLE PLUMED_CONFIG
)
set(PLUMED_CXX_FLAGS "")
set(PLUMED_CPP_FLAGS "")
set(PLUMED_DYNAMIC_LIBS "")

string(REPLACE "\n" ";" ProcessFile_LINES "${PLUMED_CONFIG}")
foreach(_line ${ProcessFile_LINES})
    # message(STATUS "Found PLUMED :${_line}")
  if (${_line} MATCHES "CXXFLAGS=.*")
    set(PLUMED_CXX_FLAGS ${_line})
    string(REGEX REPLACE "CXXFLAGS= *" "" PLUMED_CXX_FLAGS ${PLUMED_CXX_FLAGS})
    string(REPLACE " " ";" PLUMED_CXX_FLAGS ${PLUMED_CXX_FLAGS})
    # message(STATUS "Found PLUMED CXX_FLAGS: \"${PLUMED_CXX_FLAGS}\"")
    message(STATUS "Found PLUMED CXX_FLAGS: ")
    foreach (_flag ${PLUMED_CXX_FLAGS})
        message(STATUS "   \"${_flag}\"")
    endforeach()
  endif()
  if (${_line} MATCHES "CPPFLAGS=.*")
    set(PLUMED_CPP_FLAGS ${_line})
    string(REGEX REPLACE "CPPFLAGS= *" "" PLUMED_CPP_FLAGS ${PLUMED_CPP_FLAGS})
    string(REPLACE "\\" "" PLUMED_CPP_FLAGS ${PLUMED_CPP_FLAGS})
    string(REPLACE "-D" ";" PLUMED_CPP_FLAGS ${PLUMED_CPP_FLAGS})
    # message(STATUS "Found PLUMED CPP_FLAGS: \"${PLUMED_CPP_FLAGS}\"")
    message(STATUS "Found PLUMED CPP_FLAGS:")
    foreach(_flag ${PLUMED_CPP_FLAGS})
        message(STATUS "   \"${_flag}\"")
    endforeach()
    
  endif()
  if (${_line} MATCHES "DYNAMIC_LIBS=.*")
    set(PLUMED_DYNAMIC_LIBS ${_line})
    string(REGEX REPLACE "DYNAMIC_LIBS= *" "" PLUMED_DYNAMIC_LIBS ${PLUMED_DYNAMIC_LIBS})
    string(REPLACE " " ";" PLUMED_DYNAMIC_LIBS ${PLUMED_DYNAMIC_LIBS})
    # message(STATUS "Found PLUMED DYNAMIC_LIBS: \"${PLUMED_DYNAMIC_LIBS}\"")
    message(STATUS "Found PLUMED DYNAMIC_LIBS:")
    foreach(_flag ${PLUMED_DYNAMIC_LIBS})
        message(STATUS "   \"${_flag}\"")
    endforeach()
  endif()
  
endforeach()

message(STATUS "Plumed include dir: ${PLUMED_INCLUDE_DIR}")
set(CMAke__extra)
################################################################################
################################the pycv library################################
################################################################################
#TODO: remove the "lib" prefix
add_library(PythonCVInterface SHARED src/ActionWithPython.cpp  src/PythonCVInterface.cpp  src/PythonFunction.cpp)
#public, so they trickle down to the python module
target_compile_definitions(PythonCVInterface PUBLIC ${PLUMED_CPP_FLAGS})
get_target_property(COMPILE_DEFINITIONS PythonCVInterface COMPILE_DEFINITIONS)
message(STATUS "COMPILE_DEFINITIONS ${COMPILE_DEFINITIONS}")
target_compile_options(PythonCVInterface PUBLIC ${PLUMED_CXX_FLAGS})
target_compile_options(PythonCVInterface PRIVATE -fvisibility=hidden)
target_link_libraries(PythonCVInterface PRIVATE pybind11::embed)
target_link_libraries(PythonCVInterface PUBLIC plumedKernel ${PLUMED_DYNAMIC_LIBS})
target_include_directories(PythonCVInterface PUBLIC src ${PLUMED_INCLUDE_DIR})
#this removes the "lib" prefix
set_target_properties(PythonCVInterface PROPERTIES PREFIX "")

install(TARGETS PythonCVInterface DESTINATION pycv)

################################################################################
############################The pvCV companion module###########################
################################################################################

pybind11_add_module(plumedCommunications src/PlumedPythonEmbeddedModule.cpp)
#python_add_library(plumedCommunications MODULE PlumedPythonEmbeddedModule.cpp WITH_SOABI)
 target_link_libraries(plumedCommunications PRIVATE pybind11::headers)
target_link_libraries(plumedCommunications PUBLIC plumedKernel ${PLUMED_DYNAMIC_LIBS})
target_link_libraries(plumedCommunications PUBLIC PythonCVInterface)
target_include_directories(plumedCommunications PUBLIC src ${PLUMED_INCLUDE_DIR})

# The install directory is the output (wheel) directory
install(TARGETS plumedCommunications DESTINATION .)
