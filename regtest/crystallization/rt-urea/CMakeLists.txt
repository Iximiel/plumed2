PLUMED_TEST(rt-urea TYPE driver
ARGS "--ixyz traj.xyz --box 26.699,26.717,28.643"
MODULES crystallization adjmat
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules="crystallization adjmat"
# this is to test a different name
arg="--ixyz traj.xyz --box 26.699,26.717,28.643"
]] )
