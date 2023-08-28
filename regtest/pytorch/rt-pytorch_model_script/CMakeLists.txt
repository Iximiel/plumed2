PLUMED_TEST(rt-pytorch_model_script TYPE driver
ARGS "--plumed plumed.dat --mf_xtc alanine.xtc"
MODULES pytorch
NEEDS libtorch
)
#the following variable is the original config file
set(originalConfig [[
plumed_needs=libtorch
plumed_modules=pytorch
type=driver
arg="--plumed plumed.dat --mf_xtc alanine.xtc"

# note: model has been previously created with create-pytorch-model.py
# the following crashes in CI
# function plumed_regtest_before(){
#  python create-pytorch-model.py
#}
]] )
