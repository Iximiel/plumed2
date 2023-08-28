PLUMED_TEST(rt-memfusionp TYPE driver
ARGS "--plumed plumed.dat --mf_xtc MemFusionP.xtc"
MODULES membranefusion
)
#the following variable is the original config file
set(originalConfig [[
type=driver
plumed_modules=membranefusion
# this is to test a different name
arg="--plumed plumed.dat --mf_xtc MemFusionP.xtc"
]] )
