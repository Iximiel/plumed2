PLUMED_TEST(NAME rt-plumed-mpi2 TYPE driver
ARGS "--plumed=plumedx.dat --timestep=0.05 --ixyz trajectory.xyz --dump-forces ff --pdb test.pdb --multi 3
     --debug-dd yes --debug-grex 2 --debug-grex-log log.grex"
MPIPROCS 6
EXTRAFILES ../rt-mpi3/traj* ../rt-mpi3/ff*.reference ../rt-mpi3/deriv*reference
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=6
type=driver
arg="--plumed=plumedx.dat --timestep=0.05 --ixyz trajectory.xyz --dump-forces ff --pdb test.pdb --multi 3
     --debug-dd yes --debug-grex 2 --debug-grex-log log.grex"
extra_files="../rt-mpi3/traj* ../rt-mpi3/ff*.reference ../rt-mpi3/deriv*reference"
]] )
