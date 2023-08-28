PLUMED_TEST(rt-md_le-1d-mpi-4-replicas-diff-prefactors TYPE plumed
ARGS "ves_md_linearexpansion input"
MPIPROCS 4
MODULES ves
)
#the following variable is the original config file
set(originalConfig [[
plumed_modules=ves
mpiprocs=4
type=plumed
arg="ves_md_linearexpansion input"

]] )
