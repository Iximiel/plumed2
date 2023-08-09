PLUMED_TEST(NAME rt-errormessages TYPE make
)
#the following variable is the original config file
set(originalConfig [[
type=make
# this is needed because we are explicitly checking some exception message
export PLUMED_STACK_TRACE=no
]] )
