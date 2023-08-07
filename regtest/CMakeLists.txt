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
  set(multiValueArgs "EXTRAFILES")
  cmake_parse_arguments(PARSE_ARGV 0 PLUMED_TEST "${options}" "${oneValueArgs}"
    "${multiValueArgs}" )
#   cmake_parse_arguments(PLUMED_TEST "${options}" "${oneValueArgs}"
# "${multiValueArgs}" "${ARGN}" )
  
  set(args ${PLUMED_TEST_ARGS})
  set(testname ${PLUMED_TEST_NAME})
  set(type ${PLUMED_TEST_TYPE})
  message("adding test ${testname}")
  add_test(NAME ${testname} COMMAND ${CMAKE_SOURCE_DIR}/regtest/scripts/run_ctest
    $<TARGET_FILE:PLUMED2::plumed>
    -a "${type} ${args}"
    -d ${CMAKE_CURRENT_SOURCE_DIR}
    -s ${CMAKE_SOURCE_DIR}
    $< $<BOOL:${PLUMED_TEST_MPIPROCS}> : -p ${PLUMED_TEST_MPIPROCS} >
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  
  set_tests_properties(${testname} PROPERTIES
    FAIL_REGULAR_EXPRESSION "${failRegex}"
    SKIP_RETURN_CODE 127)
endfunction(PLUMED_TEST)


add_subdirectory(basic)