enable_testing()

set(type "driver")
set (plmdarg --plumed plumed.dat --ixyz trajectory.xyz --dump-forces ff --dump-forces-fmt=%10.5f --dump-full-virial)

PLUMEDTEST(NAME rt42 COMMAND ${CMAKE_SOURCE_DIR}/regtest/scripts/run_ctest setup
  -d ${CMAKE_CURRENT_SOURCE_DIR}
  -s ${CMAKE_SOURCE_DIR}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

PLUMEDTEST(NAME rt42rt COMMAND plumed_bin_runtime driver
  ${plmdarg}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tmp
)

PLUMEDTEST(NAME rt42post COMMAND ${CMAKE_SOURCE_DIR}/regtest/scripts/run_ctest check
  -d ${CMAKE_CURRENT_SOURCE_DIR}
  -s ${CMAKE_SOURCE_DIR}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tmp
)
set_tests_properties(rt42post PROPERTIES DEPENDS "rt42rt")
set_tests_properties(rt42rt PROPERTIES DEPENDS "rt42")