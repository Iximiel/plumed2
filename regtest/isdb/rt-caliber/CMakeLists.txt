PLUMED_TEST(rt-caliber TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 1 --timestep 0.005 --mf_xtc trajout.xtc --multi 4 --dump-forces allforce --dump-forces-fmt %.3f"
MPIPROCS 4
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=4
type=driver
arg="--plumed plumed.dat --trajectory-stride 1 --timestep 0.005 --mf_xtc trajout.xtc --multi 4 --dump-forces allforce --dump-forces-fmt %.3f"
]] )
