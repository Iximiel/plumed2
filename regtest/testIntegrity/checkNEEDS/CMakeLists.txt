PLUMED_TEST(checkNEEDS_SHOULD_FAIL TYPE plumed
#using help to have a dummy output
ARGS help
NEEDS notExpectedcomponent
)
#Removing the skip on non found component
set_tests_properties(${TEST_DIR}-checkNEEDS_SHOULD_FAIL PROPERTIES
    WILL_FAIL ON
    )
set_property(TEST ${TEST_DIR}-checkNEEDS_SHOULD_FAIL PROPERTY SKIP_RETURN_CODE)
