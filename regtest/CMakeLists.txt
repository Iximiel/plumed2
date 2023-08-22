cmake_minimum_required(VERSION 3.20.2)
#This cmake project can be prepared on its own for testing another plumed 
#cmake installation, remember to add the directory with the Plumed2Config.cmake 
#to the CMAKE_PREFIX_PATH
project(Plumed2Tests LANGUAGES CXX)

enable_testing()

if(NOT TARGET Plumed2::plumed_bin)
    find_package(Plumed2 REQUIRED)
endif()

set(run_ctest
  ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_ctest
)

#set (failRegex "FAILURE;ERROR;WARNING")
set (failRegex "FAILURE;ERROR")

set(PLUMED_TEST_KNOWN_TYPES
  #simplemd
  driver
  #sum_hills
  #make
  #plumed
  #python
)

function(PLUMED_TEST)
#I will then separate the different test types in driver, compile ecc...
  set(options "")
  set(oneValueArgs "NAME;ARGS;TYPE;MPIPROCS")
  set(multiValueArgs "EXTRAFILES;NEEDS;LABELS;MODULES")
  cmake_parse_arguments(PARSE_ARGV 0 PLUMED_TEST "${options}" "${oneValueArgs}"
    "${multiValueArgs}" )
#   cmake_parse_arguments(PLUMED_TEST "${options}" "${oneValueArgs}"
# "${multiValueArgs}" "${ARGN}" )
  
  set(args ${PLUMED_TEST_ARGS})
  set(testname ${TEST_DIR}-${PLUMED_TEST_NAME})
  set(type ${PLUMED_TEST_TYPE})
  #message("Test ${testname}::extra:${PLUMED_TEST_EXTRAFILES}")
  
  if(NOT PLUMED_TEST_TYPE)
    message(FATAL_ERROR "Test ${testname}::Test need TYPE to be indicated")
  endif()

  list(FIND PLUMED_TEST_KNOWN_TYPES ${type} TYPE_KNOWN)
  if(TYPE_KNOWN EQUAL -1)
    message(WARNING #FATAL_ERROR
    "Test ${testname}::Test TYPE \"${type}\" is not in the"
    "supported list \"${PLUMED_TEST_KNOWN_TYPES}\"")
    return()
  endif()

  if(PLUMED_TEST_MPIPROCS)
    set(PLUMED_TEST_MPIPROCS "${PLUMED_TEST_MPIPROCS}")
    set(popt "-p")
  endif()
  if(PLUMED_TEST_EXTRAFILES)
    list(JOIN PLUMED_TEST_EXTRAFILES " " t)
    set (PLUMED_TEST_EXTRAFILES  ${t})
    set (eopt "-e")
  endif()
  if(PLUMED_TEST_NEEDS)
    list(JOIN PLUMED_TEST_NEEDS " " t)
    set (PLUMED_TEST_NEEDS  ${t})
    set (nopt "-n")
  endif()

  if(PLUMED_TEST_MODULES)
    list(JOIN PLUMED_TEST_MODULES " " t)
    set (PLUMED_TEST_MODULES  ${t})
    set (mopt "-n")
  endif()

  add_test(NAME ${testname} COMMAND ${run_ctest}
    $<TARGET_FILE:Plumed2::plumed_bin>
    -a "${type} ${args}"
    -d ${CMAKE_CURRENT_SOURCE_DIR}
    -s ${CMAKE_SOURCE_DIR}
    ${popt} ${PLUMED_TEST_MPIPROCS}
    ${eopt} ${PLUMED_TEST_EXTRAFILES}
    ${nopt} ${PLUMED_TEST_NEEDS}
    ${mopt} ${PLUMED_TEST_MODULES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}

  )
  message("Test ${testname}::adding test")
  
  LIST(APPEND PLUMED_TEST_LABELS ${TEST_DIR})

  set_tests_properties(${testname} PROPERTIES
    FAIL_REGULAR_EXPRESSION "${failRegex}"
    SKIP_RETURN_CODE 127
    LABELS "${PLUMED_TEST_LABELS}")
endfunction(PLUMED_TEST)


add_subdirectory(basic)