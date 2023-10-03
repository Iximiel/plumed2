PLUMED_TEST(rt-dlp4 TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 1 --idlp4 HISTORY --dump-forces forces --dump-forces-fmt=%10.6f"
MPIPROCS 2
EXTRAFILES ../../trajectories/dlp4/HISTORY
)
#the following variable is the original config file
set(originalConfig [[
mpiprocs=2 
type=driver
arg="--plumed plumed.dat --trajectory-stride 1 --idlp4 HISTORY --dump-forces forces --dump-forces-fmt=%10.6f"
extra_files="../../trajectories/dlp4/HISTORY"
]] )
