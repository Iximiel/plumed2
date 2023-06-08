install( FILES plumed_f08.f90  plumed.f90
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/${program_name}/fortran
    PERMISSIONS
        OWNER_WRITE
        OWNER_READ GROUP_READ WORLD_READ
)