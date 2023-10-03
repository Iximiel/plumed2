PLUMED_TEST(rt-molfile-2 TYPE driver
ARGS "--plumed plumed.dat --trajectory-stride 1 --timestep 0.005 --pdb diala.pdb --mf_dcd traj.dcd --dump-forces forces --dump-forces-fmt=%10.6f"
NEEDS molfile_plugins
EXTRAFILES ../../trajectories/molfile_plugin/traj.dcd ../../trajectories/molfile_plugin/test0.pdb ../../trajectories/molfile_plugin/test1.pdb ../../trajectories/molfile_plugin/diala.pdb 
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_needs=molfile_plugins
arg="--plumed plumed.dat --trajectory-stride 1 --timestep 0.005 --pdb diala.pdb --mf_dcd traj.dcd --dump-forces forces --dump-forces-fmt=%10.6f"
extra_files="../../trajectories/molfile_plugin/traj.dcd ../../trajectories/molfile_plugin/test0.pdb ../../trajectories/molfile_plugin/test1.pdb ../../trajectories/molfile_plugin/diala.pdb "
]] )
