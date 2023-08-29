PLUMED_TEST(checkMODULES TYPE nothing
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
