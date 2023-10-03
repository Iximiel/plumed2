enable_testing()

project(regtest-eds)
set(TEST_DIR "eds")
FILE(GLOB children RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
FOREACH(child ${children})  
  IF(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${child} AND
      EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${child}/CMakeLists.txt)
    add_subdirectory(${child})
  ENDIF()
ENDFOREACH()
