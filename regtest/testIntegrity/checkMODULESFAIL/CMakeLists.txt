PLUMED_TEST(checkMODULESFAIL TYPE plumed
#using help to have a dummy output
ARGS help
#these are a recipe of always on modules, so it can not fail
MODULES
    notEXISTENTmodule
)

#Removing the skip on non found component
set_tests_properties(${TEST_DIR}-checkMODULESFAIL PROPERTIES
    WILL_FAIL ON
    )
set_property(TEST ${TEST_DIR}-checkMODULESFAIL PROPERTY SKIP_RETURN_CODE)
