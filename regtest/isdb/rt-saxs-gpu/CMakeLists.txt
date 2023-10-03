PLUMED_TEST(rt-saxs-gpu TYPE driver
ARGS "--plumed plumed.dat --timestep 0.005 --mf_pdb template.pdb --dump-forces ff"
MPIPROCS 2
NEEDS arrayfire
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=2
type=driver
arg="--plumed plumed.dat --timestep 0.005 --mf_pdb template.pdb --dump-forces ff"
plumed_needs="arrayfire"
PLUMED_ALLOW_SKIP_ON_TRAVIS=yes
]] )
