PLUMED_TEST(rt-ghbfix TYPE driver
ARGS "--plumed plumed.dat --ixyz traj.xyz"
)
#the following variable is the original config file
set(originalConfig [[
type=driver

arg="--plumed plumed.dat --ixyz traj.xyz"

function plumed_regtest_before(){
awk 'BEGIN{
  for(i=1;i<=150;i++) {
    print "2"
    print "100 100 100"
    print "X 0 0 0"
    print "X 0 0",i*0.002
  }
}' > traj.xyz
}
]] )
