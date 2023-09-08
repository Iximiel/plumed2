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
  else()
  set (Plumed2_DIR "${CMAKE_BINARY_DIR}/lib/cmake/plumed")
endif()

set(PLUMED_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(run_ctest ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_ctest)

#set (failRegex "FAILURE;ERROR;WARNING")
set (failRegex "FAILURE;ERROR")
set (skipRegex "NOT_APPLICABLE")

set(PLUMED_TEST_KNOWN_TYPES
  simplemd
  driver
  sum_hills
  make
  plumed
  nothing#special comand to not make plumed run (and use only plumed_regtest_before/after and plumed_custom_skip)
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
  #[COMPILER_LANGUAGE C11 or Fortran08]
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
  #With NEEDS the user can specify a list needed components, if the specified 
  #componets are not found by `plumed config has componentname` the test will be skipped.
  #
  #With MODULES the user can specify a list needed modules, if the specified 
  #componets are not found by `plumed config module modulename` the test will be skipped.
  #
  #With COMPILER_LANGUAGE you can specify Fortan08 or C11 and the make test will be skipped if the compiler is not compatible (works ONLY 
  #for make tests), note the argments are case insensitive
  #
  #With LABELS the user can specify a collection of keyword that can be use to 
  #limit the number of test to be runned with ctest. The type and the home 
  #directory of the test are automatically set up as labels
  #
  #By specifying one of "simplemd", "driver", "sum_hills" as type, the keyword 
  # will be automatically prepended to the specified arguments

  set(options "")
  set(oneValueArgs "TYPE;ARGS;MPIPROCS;COMPILER_LANGUAGE")
  set(multiValueArgs "EXTRAFILES;EXPORTVARIABLES;NEEDS;LABELS;MODULES")
  cmake_parse_arguments(PARSE_ARGV 1 PLUMED_TEST "${options}" "${oneValueArgs}"
    "${multiValueArgs}" )
  
  if(PLUMED_TEST_ARGS)
    set(PLUMED_TEST_ARGS "-a${PLUMED_TEST_ARGS}")
  endif(PLUMED_TEST_ARGS)
  set(testname ${TEST_DIR}-${test_name})
  #message("Test ${testname}::extra:${PLUMED_TEST_EXTRAFILES}")
  
  if(NOT PLUMED_TEST_TYPE)
    message(FATAL_ERROR "Test ${testname}::Test need TYPE to be indicated")
  endif()

  list(FIND PLUMED_TEST_KNOWN_TYPES ${PLUMED_TEST_TYPE} TYPE_KNOWN)
  if(TYPE_KNOWN EQUAL -1)
    # message(WARNING #FATAL_ERROR
    # "Test ${testname}::Test TYPE \"${PLUMED_TEST_TYPE}\" is not in the "
    # "supported list \"${PLUMED_TEST_KNOWN_TYPES}\"")
    return()
  endif()
#TODO: export PLUMED_STACK_TRACE=no the export!!!!
  if(PLUMED_TEST_MPIPROCS)
    set(PLUMED_TEST_MPIPROCS "-p${PLUMED_TEST_MPIPROCS}")
  endif()
  
  if(PLUMED_TEST_COMPILER_LANGUAGE)
    string(TOLOWER ${PLUMED_TEST_COMPILER_LANGUAGE} PLUMED_TEST_COMPILER_LANGUAGE)
    set(PLUMED_TEST_COMPILER_LANGUAGE "-c${PLUMED_TEST_COMPILER_LANGUAGE}")
  endif()


  if(PLUMED_TEST_EXTRAFILES)
    set (t)
    foreach(extra IN LISTS PLUMED_TEST_EXTRAFILES)
        list(APPEND t "-f${extra}" )  
    endforeach(extra)
    set (PLUMED_TEST_EXTRAFILES  ${t})
  endif()
  
  if(PLUMED_TEST_NEEDS)
    list(JOIN PLUMED_TEST_NEEDS " " t)
    set (PLUMED_TEST_NEEDS  -n${t})
  endif()

  if(PLUMED_TEST_MODULES)
    list(JOIN PLUMED_TEST_MODULES " " t)
    set (PLUMED_TEST_MODULES  -m${t})
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
    list(APPEND PLUMED_TEST_EXPORTVARIABLES
      "-ePLUMED_ROOT=${CMAKE_BINARY_DIR}")
    set(PLUMED_TEST_NOT_STANDALONELIB "-l${CMAKE_BINARY_DIR}/src")
  endif()
  
  list(APPEND PLUMED_TEST_EXPORTVARIABLES 
    "-ePlumed2_DIR=${Plumed2_DIR}"
    "-ePLUMED_KERNEL=$<TARGET_LINKER_FILE:Plumed2::sharedplumedKernel>"
    "-ePLUMED_LIB=$<TARGET_LINKER_FILE:Plumed2::sharedplumedMain>"
    )

  add_test(NAME ${testname} COMMAND ${run_ctest}
    ${PLUMED_TEST_ARGS}
    -d${CMAKE_CURRENT_SOURCE_DIR}
    ${PLUMED_TEST_MPIPROCS}
    ${PLUMED_TEST_EXTRAFILES}
    ${PLUMED_TEST_COMPILER_LANGUAGE}
    ${nopt} ${PLUMED_TEST_NEEDS}
    ${mopt} ${PLUMED_TEST_MODULES}
    ${PLUMED_TEST_EXPORTVARIABLES}
    ${PLUMED_TEST_NOT_STANDALONE}
    ${PLUMED_TEST_NOT_STANDALONELIB}
    $<TARGET_FILE:Plumed2::plumed_bin>
    ${PLUMED_TEST_TYPE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  if(VERBOSE)
    message("Adding test \"${testname}\"")
  endif(VERBOSE)
  
  LIST(APPEND PLUMED_TEST_LABELS dir-${TEST_DIR} type-${PLUMED_TEST_TYPE})

  set_tests_properties(${testname} PROPERTIES
    FAIL_REGULAR_EXPRESSION "${failRegex}"
    SKIP_REGULAR_EXPRESSION "${skipRegex}"
    SKIP_RETURN_CODE 125
    LABELS "${PLUMED_TEST_LABELS}")
endfunction(PLUMED_TEST)

#This directory ensures that the test procedure works as intended
add_subdirectory(testIntegrity)
#basic test tests
add_subdirectory(basic)

foreach(dir adjmat analysis annfunc core crystallization dimred drr 
        eds fisst funnel isdb logmfd manyrestraints mapping maze membranefusion 
        multicolvar opes pamm piv python pytorch s2cm sasa secondarystructure 
        ves)
        add_subdirectory(${dir})
endforeach(dir)

