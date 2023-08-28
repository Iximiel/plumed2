cmake_minimum_required(VERSION 3.20.2)
#This cmake project can be prepared on its own for testing another plumed 
#cmake installation, remember to add the directory with the Plumed2Config.cmake 
#to the CMAKE_PREFIX_PATH
project(Plumed2Tests LANGUAGES CXX)

enable_testing()

if(NOT TARGET Plumed2::plumed_bin)
  find_package(Plumed2 REQUIRED)
  #This sets up the environmental variable CMAKE_PREFIX_PATH in run_ctest 
  #when testing in the build environment
  set(STANDALONE_TESTS ON)
endif()

set(PLUMED_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(run_ctest ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_ctest)

#set (failRegex "FAILURE;ERROR;WARNING")
set (failRegex "FAILURE;ERROR")

set(PLUMED_TEST_KNOWN_TYPES
  simplemd
  driver
  sum_hills
  make
  plumed
  #python
)

function(PLUMED_TEST test_name)
  #use: PLUMED_TEST(<test_name> TYPE <test_type>
  #[ARGS <string>]
  #[MPIPROCS <number>]
  #[EXTRAFILES path1 path2 ...]
  #[EXPORTVARIABLES variable1 value1 variable2 value2 ...]
  #[NEEDS component1 component2 ...]
  #[MODULES module1 module2 ...]
  #[LABELS label1 label2 ...]
  #)

  #PLUMED_TEST will set up a call for run_ctest with the settings given by the user
  #test_type should be one of 
  #  - "plumed" will invoke plumed
  #    - "simplemd" will invoke plumed with simplemd
  #    - "driver" will invoke plumed with driver
  #    - "sum_hills" will invoke plumed with sum_hills
  #  - "make" will build a custom executable with the files in the test directory
  #  - "python" 
  # the "make" test will compile with cmake using a list that contain the following instructions:
  #```
  # find_package(Plumed2 REQUIRED)
  # add_executable(exe ${files})
  # target_link_libraries(exe PUBLIC
  #   Plumed2::sharedplumedMain
  # )
  #```
  #
  #the arguments to pass to plumed (used with "driver", "plumed", "simplemd", 
  # "sum_hills"), must be specified in a single string, such as 
  #`ARGS "--plumed plumed.dat --trajectory-stride 10 --timestep 0.005 --ixyz trajectory.xyz --dump-forces forces --dump-forces-fmt=%8.4f"`.
  #
  #If MPIPROCS>1 the test will be skipped if mpi is not avaiable.
  #
  #With EXTRAFILES the user can specify a list of paths to files to be copied in the test directory
  #
  #
  #With NEEDS the user can specify a list needed components, if the specified 
  #componets are not found by `plumed config has componentname` the test will be skipped.
  #
  #With MODULES the user can specify a list needed modules, if the specified 
  #componets are not found by `plumed config module modulename` the test will be skipped.
  #
  #With LABELS the user can specify a collection of keyword that can be use to 
  #limit the number of test to be runned with ctest. The type and the home 
  #directory of the test are automatically set up as labels
  #
  #By specifying one of "simplemd", "driver", "sum_hills" as type, the keyword 
  # will be automatically prepended to the specified arguments

  set(options "")
  set(oneValueArgs "TYPE;ARGS;MPIPROCS")
  set(multiValueArgs "EXTRAFILES;EXPORTVARIABLES;NEEDS;LABELS;MODULES")
  cmake_parse_arguments(PARSE_ARGV 1 PLUMED_TEST "${options}" "${oneValueArgs}"
    "${multiValueArgs}" )
  
  set(args ${PLUMED_TEST_ARGS})
  set(testname ${TEST_DIR}-${test_name})
  set(type ${PLUMED_TEST_TYPE})
  #message("Test ${testname}::extra:${PLUMED_TEST_EXTRAFILES}")
  
  if(NOT PLUMED_TEST_TYPE)
    message(FATAL_ERROR "Test ${testname}::Test need TYPE to be indicated")
  endif()

  list(FIND PLUMED_TEST_KNOWN_TYPES ${type} TYPE_KNOWN)
  if(TYPE_KNOWN EQUAL -1)
    # message(WARNING #FATAL_ERROR
    # "Test ${testname}::Test TYPE \"${type}\" is not in the "
    # "supported list \"${PLUMED_TEST_KNOWN_TYPES}\"")
    return()
  endif()
#TODO: export PLUMED_STACK_TRACE=no the export!!!!
  if(PLUMED_TEST_MPIPROCS)
    set(PLUMED_TEST_MPIPROCS "${PLUMED_TEST_MPIPROCS}")
    set(popt "-p")
  endif()
  if(PLUMED_TEST_EXTRAFILES)
    list(JOIN PLUMED_TEST_EXTRAFILES " " t)
    set (PLUMED_TEST_EXTRAFILES  ${t})
    set (eopt "-f")
  endif()
  if(PLUMED_TEST_NEEDS)
    list(JOIN PLUMED_TEST_NEEDS " " t)
    set (PLUMED_TEST_NEEDS  ${t})
    set (nopt "-n")
  endif()

  if(PLUMED_TEST_MODULES)
    list(JOIN PLUMED_TEST_MODULES " " t)
    set (PLUMED_TEST_MODULES  ${t})
    set (mopt "-m")
  endif()

  if(PLUMED_TEST_EXPORTVARIABLES)
    set (t "")
    set (l)
    set (sep "-e")
    foreach(var IN LISTS PLUMED_TEST_EXPORTVARIABLES)
      string(FIND ${var} " " therearespaces)
      STRING(APPEND t "${sep}${var}" )
      if(sep STREQUAL "=")
        set (sep "-e")
        list(APPEND l ${t})
        unset(t)
      else()
        set (sep "=")
      endif()
    endforeach()
    set (PLUMED_TEST_EXPORTVARIABLES  ${l})
    unset(l)
    unset(sep)
  endif()

  if(NOT STANDALONE_TESTS)
    list(APPEND PLUMED_TEST_EXPORTVARIABLES "-ePlumed2_DIR=${CMAKE_BINARY_DIR}/src/cmake")
    list(APPEND PLUMED_TEST_EXPORTVARIABLES "-ePLUMED_ROOT=${CMAKE_BINARY_DIR}")
    set(PLUMED_TEST_NOT_STANDALONELIB "-l${CMAKE_BINARY_DIR}/src")
  endif()

  add_test(NAME ${testname} COMMAND ${run_ctest}
    $<TARGET_FILE:Plumed2::plumed_bin>
    -t "${type}"
    -a "${args}"
    -d ${CMAKE_CURRENT_SOURCE_DIR}
    -s ${CMAKE_SOURCE_DIR}
    ${popt} ${PLUMED_TEST_MPIPROCS}
    ${eopt} ${PLUMED_TEST_EXTRAFILES}
    ${nopt} ${PLUMED_TEST_NEEDS}
    ${mopt} ${PLUMED_TEST_MODULES}
    ${PLUMED_TEST_EXPORTVARIABLES}
    ${PLUMED_TEST_NOT_STANDALONE}
    ${PLUMED_TEST_NOT_STANDALONELIB}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}

  )
  message("Test ${testname}::adding test")
  
  LIST(APPEND PLUMED_TEST_LABELS ${TEST_DIR} ${type})

  set_tests_properties(${testname} PROPERTIES
    FAIL_REGULAR_EXPRESSION "${failRegex}"
    SKIP_RETURN_CODE 125
    LABELS "${PLUMED_TEST_LABELS}")
endfunction(PLUMED_TEST)

#This directory ensures that the test procedure works as intended
add_subdirectory(testIntegrity)
#standard tests
add_subdirectory(basic)


