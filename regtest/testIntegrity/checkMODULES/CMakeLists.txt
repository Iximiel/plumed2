PLUMED_TEST(checkMODULES TYPE plumed
#using help to have a dummy output
ARGS help
#these are a recipe of always on modules, so it can not fail
MODULES
    asmjit
    blas
    core
    lapack
    lepton
    tools
)

#Removing the skip on non found component
# set_tests_properties(${TEST_DIR}-checkMODULES PROPERTIES
#     SKIP_RETURN_CODE
#     )
set_property(TEST ${TEST_DIR}-checkMODULES PROPERTY SKIP_RETURN_CODE)
