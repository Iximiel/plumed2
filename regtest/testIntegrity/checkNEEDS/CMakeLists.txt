PLUMED_TEST(checkNEEDSFailing TYPE plumed
#using help to have a dummy output
ARGS help
NEEDS notExpectedcomponent
)
#Removing the skip on non found component
set_tests_properties(${TEST_DIR}-checkNEEDSFailing PROPERTIES
    WILL_FAIL ON
    )
set_property(TEST ${TEST_DIR}-checkNEEDSFailing PROPERTY SKIP_RETURN_CODE)
