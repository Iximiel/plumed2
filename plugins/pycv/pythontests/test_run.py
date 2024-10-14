import unittest
import numpy as np
from plumed import Plumed

import os
from contextlib import contextmanager

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(newdir)
    try:
        yield
    finally:
        os.chdir(prevdir)

def read_xyz(filename):
   xyz = open(filename)
   n_atoms = int(xyz.readline())
   title = xyz.readline()
   trajectory =[]
   while True :
      atom_type= np.zeros(n_atoms).astype(str)
      coordinates = np.zeros([n_atoms,3]) 
      for i in range(0,n_atoms) :
          line = xyz.readline()
          atom,x,y,z = line.split()
          atom_type[i]=atom
          coordinates[i,:]=np.array([x,y,z],dtype=np.float64)
      trajectory.append( coordinates )
      nextline = xyz.readline()
      if( nextline=="" ) : break
      c_atoms = int(nextline)
      if( c_atoms!=n_atoms ) : break 
      title = xyz.readline()
   xyz.close()
   return trajectory

def create_plumed_var( plmd, name, command ):
   plmd.cmd("readInputLine", name + ": " + command )
   shape = np.zeros( 1, dtype=np.int_ )
   plmd.cmd("getDataRank " + name, shape )
   data = np.zeros((1))
   plmd.cmd("setMemoryForData " + name, data )
   return data

class Test(unittest.TestCase):
  def runtest(self):
    from pycv import getPythonCVInterface
    os.system('rm -f bck.*')
    # Output to four decimal places only
    np.set_printoptions(precision=4)
    # Read trajectory
    traj = read_xyz("traj.xyz")
    num_frames = len(traj)
    num_atoms = traj[0].shape[0]
    
    # Create arrays for stuff
    box=np.diag(12.41642*np.ones(3,dtype=np.float64))
    virial=np.zeros((3,3),dtype=np.float64)
    masses=np.ones(num_atoms,dtype=np.float64)
    forces=np.random.rand(num_atoms,3)
    charges=np.zeros(num_atoms,dtype=np.float64)
    
    # Create PLUMED object and read input
    plmd = Plumed()

    # not really needed, used to check https://github.com/plumed/plumed2/issues/916
    plumed_version = np.zeros(1, dtype=np.intc)
    plmd.cmd( "getApiVersion", plumed_version)

    plmd.cmd("setMDEngine","python")
    plmd.cmd("setTimestep", 1.)
    plmd.cmd("setKbT", 1.)
    plmd.cmd("setNatoms",num_atoms)
    plmd.cmd("setLogFile","test.log")
    plmd.cmd("init")
    # plmd.cmd("readInputLine","LOAD FILE=./PythonCVInterface.so")
    plmd.cmd("readInputLine",f"LOAD FILE={getPythonCVInterface()}")
    cvPy = create_plumed_var( plmd, "cvPy", "PYCVINTERFACE IMPORT=mypycv")
    plmd.cmd("readInputLine","PRINT FILE=colvar.out ARG=*")
    # Open an output file
    with open("logfile", "w+") as of:
    
        # Now analyze the trajectory
        for step in range(0,num_frames) :
            of.write("RUNNING ANALYSIS FOR STEP " + str(step) + "\n" )
            plmd.cmd("setStep",step )
            plmd.cmd("setBox",box )
            plmd.cmd("setMasses", masses )
            plmd.cmd("setCharges", charges )
            plmd.cmd("setPositions", traj[step])
            plmd.cmd("setForces", forces )
            plmd.cmd("setVirial", virial )
            plmd.cmd("calc")
            
            self.assertEqual(cvPy,2)
    

  def test(self):
    
    self.runtest()

if __name__ == "__main__":
    unittest.main()

