PLUMED_TEST(rt-md_le-1d-mpi-5-replicas-diff-asymmetric-potentials TYPE plumed
ARGS "ves_md_linearexpansion input"
MPIPROCS 5
MODULES ves
)
#the following variable is the original config file
set(originalConfig [[
plumed_modules=ves
mpiprocs=5
type=plumed
arg="ves_md_linearexpansion input"

]] )
