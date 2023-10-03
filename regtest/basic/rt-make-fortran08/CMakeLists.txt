PLUMED_TEST(rt-make-fortran08 TYPE make
EXPORTVARIABLES
PLUMED_STACK_TRACE "no"
COMPILER_LANGUAGE Fortran08
)
#the following variable is the original config file
set(originalConfig [[
type=make
export PLUMED_STACK_TRACE=no
plumed_language=fortran08

function plumed_regtest_after(){
# this should be empty since _ and __ names are not used
nm *.o | grep "plumed_f_[A-Za-z0-9]*_$" > names.dat
}

]] )
