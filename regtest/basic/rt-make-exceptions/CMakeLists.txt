PLUMED_TEST(rt-make-exceptions TYPE make
NEEDS dlopen cregex
EXPORTVARIABLES
PLUMED_STACK_TRACE "no"
)
#the following variable is the original config file
set(originalConfig [[
type=make
plumed_needs="dlopen cregex"
# this is needed because we are explicitly checking some exception message
export PLUMED_STACK_TRACE=no
]] )
