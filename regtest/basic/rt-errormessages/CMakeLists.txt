PLUMED_TEST(rt-errormessages TYPE make
EXPORTVARIABLES
PLUMED_STACK_TRACE "no"
)
#the following variable is the original config file
set(originalConfig [[
type=make
# this is needed because we are explicitly checking some exception message
export PLUMED_STACK_TRACE=no
]] )
