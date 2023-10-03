#To the developer: use this simple makefile to generate the fortran interfaces
all: plumed.f90 plumed_f08.f90

plumed.f90: plumed.f90.template
	fypp plumed.f90.template > plumed.f90

plumed_f08.f90: plumed_f08.f90.template
	fypp plumed_f08.f90.template > plumed_f08.f90
