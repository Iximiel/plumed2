PLUMED_TEST(rt-q6-lq6-around TYPE driver
ARGS "--ixyz lq6_test.xyz --box 61.8440,71.5360,150.00000"
MODULES crystallization
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules=crystallization
# this is to test a different name
arg="--ixyz lq6_test.xyz --box 61.8440,71.5360,150.00000"
]] )
