PLUMED_TEST(rt-make-c-c++ TYPE make
EXPORTVARIABLES
PLUMED_STACK_TRACE "no"
)
#the following variable is the original config file
set(originalConfig [[
type=make
export PLUMED_STACK_TRACE=no
]] )
