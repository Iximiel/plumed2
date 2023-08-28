PLUMED_TEST(rt-wham TYPE driver
ARGS "--mf_xtc alltraj.xtc --multi 6"
MPIPROCS 6
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=6
type=driver
arg="--mf_xtc alltraj.xtc --multi 6" 
]] )
