enable_testing()

if(NOT TARGET PLUMED2::plumed)
    find_package(PLUMED2 REQUIRED)
endif()

#set (failRegex "FAILURE;ERROR;WARNING")
set (failRegex "FAILURE;ERROR")

function(PLUMED_TEST)
#I will then separate the different test types in driver, compile ecc...
  set(options "")
  set(oneValueArgs "NAME;ARGS;TYPE;MPIPROCS")
  set(multiValueArgs "EXTRAFILES;NEEDS")
  cmake_parse_arguments(PARSE_ARGV 0 PLUMED_TEST "${options}" "${oneValueArgs}"
    "${multiValueArgs}" )
#   cmake_parse_arguments(PLUMED_TEST "${options}" "${oneValueArgs}"
# "${multiValueArgs}" "${ARGN}" )
  
  set(args ${PLUMED_TEST_ARGS})
  set(testname ${PLUMED_TEST_NAME})
  set(type ${PLUMED_TEST_TYPE})
  message("adding test ${testname}")
  message("extra:${PLUMED_TEST_EXTRAFILES}")
  
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

  add_test(NAME ${testname} COMMAND ${CMAKE_SOURCE_DIR}/regtest/scripts/run_ctest
    $<TARGET_FILE:PLUMED2::plumed>
    -a "${type} ${args}"
    -d ${CMAKE_CURRENT_SOURCE_DIR}
    -s ${CMAKE_SOURCE_DIR}
    ${popt} ${PLUMED_TEST_MPIPROCS}
    ${eopt} ${PLUMED_TEST_EXTRAFILES}
    ${nopt} ${PLUMED_TEST_NEEDS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  
  set_tests_properties(${testname} PROPERTIES
    FAIL_REGULAR_EXPRESSION "${failRegex}"
    SKIP_RETURN_CODE 127)
endfunction(PLUMED_TEST)


add_subdirectory(basic)