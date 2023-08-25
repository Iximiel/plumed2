enable_testing()
#this subdir is needed for checking that the test facility works as intended

project(testIntegrity)
set(TEST_DIR "testIntegrity")
FILE(GLOB children RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
FOREACH(child ${children})  
  IF(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${child})
    add_subdirectory(${child})
  ENDIF()
ENDFOREACH()
