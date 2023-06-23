#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "colvar")
#Note that the macros here require this directory added as a subdir of plumed/src
DECLAREPLUMEDMODULE(${module_name}
ON
SOURCES
Angle.cpp
Cell.cpp
Colvar.cpp
Constant.cpp
ContactMap.cpp
CoordinationBase.cpp
Coordination.cpp
DHEnergy.cpp
Dimer.cpp
Dipole.cpp
Distance.cpp
DRMSD.cpp
EEFSolv.cpp
Energy.cpp
ERMSD.cpp
ExtraCV.cpp
Fake.cpp
GHBFIX.cpp
Gyration.cpp
MultiRMSD.cpp
PathMSDBase.cpp
PathMSD.cpp
PCARMSD.cpp
Position.cpp
ProjectionOnAxis.cpp
PropertyMap.cpp
Puckering.cpp
RMSD.cpp
Template.cpp
Torsion.cpp
Volume.cpp
NEEDS
	core reference tools
DEPENDS
	reference #tools core
)

#this installs the Template.cpp example colvar
install(FILES Template.cpp
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/${program_name}/src/colvar)
