PLUMED_TEST(NAME rt-make-c11 TYPE make
)
#the following variable is the original config file
set(originalConfig [[
type=make
plumed_language=c11
# this is needed because we are explicitly checking some exception message
export PLUMED_STACK_TRACE=no

]] )
