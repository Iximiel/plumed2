PLUMED_TEST(rt-emmi-metad-mpi TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride=1 --timestep=0.002 --multi=2 --mf_xtc traj.xtc"
MPIPROCS 2
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=2
type=driver
arg="--plumed plumed.dat --trajectory-stride=1 --timestep=0.002 --multi=2 --mf_xtc traj.xtc"
]] )
