PLUMED_TEST(checkEXTRAFILES TYPE nothing
EXTRAFILES 
${PLUMED_TEST_DIR}/trajectories/molfile_plugin/traj.dcd 
${PLUMED_TEST_DIR}/trajectories/molfile_plugin/test0.pdb 
../../trajectories/molfile_plugin/test1.pdb
../../trajectories/molfile_plugin/diala.pdb
#these commands checks that the globs work
../../trajectories/checkStarFiles/relativeGlob_*
${PLUMED_TEST_DIR}/trajectories/checkStarFiles/absoluteGlob_*
)
