cmake_minimum_required(VERSION 3.20.2)
#Thi directory can be compiled as an autonomos project
install( FILES plumed_f08.f90  plumed.f90
    DESTINATION ${DIRS_EXTRA_INSTALL}/fortran
    PERMISSIONS
        OWNER_WRITE
        OWNER_READ GROUP_READ WORLD_READ
)

file(COPY plumed_f08.f90  plumed.f90
DESTINATION ${CMAKE_BINARY_DIR}/lib/plumed/fortran
USE_SOURCE_PERMISSIONS
)
include(CheckLanguage)
check_language(Fortran)
if(NOT CMAKE_Fortran_COMPILER)
  message(STATUS "Fortran cannot be compiled")
  return()
endif()

#project(Plumed2Fortran LANGUAGES CXX Fortran)
