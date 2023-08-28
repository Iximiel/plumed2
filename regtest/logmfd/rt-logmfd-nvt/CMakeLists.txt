PLUMED_TEST(rt-logmfd-nvt TYPE simplemd
ARGS "--plumed plumed.dat < in"
MODULES logmfd
EXPORTVARIABLES
PLUMED_NUM_THREADS "1"
EXTRAFILES ../input.xyz
)
#the following variable is the original config file
set(originalConfig [[
type=simplemd
plumed_modules=logmfd
arg="--plumed plumed.dat < in"
extra_files="../input.xyz"
export PLUMED_NUM_THREADS=1

]] )
