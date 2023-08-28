PLUMED_TEST(rt-sean-marks TYPE driver
ARGS "--timestep 0.002 --trajectory-stride 500 --dump-forces forces_bias_lambdav.out --mf_xtc some_frames.xtc --dump-forces-fmt %8.4f"
MODULES crystallization
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules=crystallization
arg="--timestep 0.002 --trajectory-stride 500 --dump-forces forces_bias_lambdav.out --mf_xtc some_frames.xtc --dump-forces-fmt %8.4f"
]] )
