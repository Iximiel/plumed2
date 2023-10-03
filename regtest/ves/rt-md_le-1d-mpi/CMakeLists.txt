PLUMED_TEST(rt-md_le-1d-mpi TYPE plumed
ARGS "ves_md_linearexpansion input"
MPIPROCS 2
MODULES ves
)
#the following variable is the original config file
set(originalConfig [[
plumed_modules=ves
mpiprocs=2
type=plumed
arg="ves_md_linearexpansion input"

]] )
