PLUMED_TEST(checkMODULES_SHOULD_FAIL TYPE nothing
#these are a recipe of always on modules, so it can not fail
MODULES
    notEXISTENTmodule
)

#Removing the skip on non found component
set_tests_properties(${TEST_DIR}-checkMODULES_SHOULD_FAIL PROPERTIES
    WILL_FAIL ON
    )
set_property(TEST ${TEST_DIR}-checkMODULES_SHOULD_FAIL PROPERTY SKIP_RETURN_CODE)
