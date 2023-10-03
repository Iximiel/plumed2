#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "lapack")
#Note that the macros here require this directory added as a subdir of plumed/src
option(useExternal_LAPACK "enable search for external LAPACK, default ON" ON)
DECLAREPLUMEDMODULE(${module_name}
"always"
SOURCES
    lapack.cpp
EXTRA_HEADERS
    def_external.h
    def_internal.h
    real.h
    lapack_limits.h
NEEDS
    blas
DEPENDS
    blas
)

if (useExternal_LAPACK AND ${BLAS_FOUND})
    find_package(LAPACK)
endif (useExternal_LAPACK AND ${BLAS_FOUND})

if (LAPACK_FOUND)
    target_compile_definitions(${module_name}
        PUBLIC
        __PLUMED_HAS_EXTERNAL_LAPACK=1)
    target_compile_options(${module_name} PRIVATE ${LAPACK_LINKER_FLAGS})
    target_link_libraries(${module_name}
        PUBLIC
        LAPACK::LAPACK)
    list(APPEND PLUMED_DYNAMIC_LIBS ${LAPACK_LIBRARIES})
    list(APPEND PLUMED_LD_FLAGS ${LAPACK_LINKER_FLAGS})
else()
    set(LAPACK_FOUND OFF)
endif (LAPACK_FOUND)
set (LAPACK_FOUND LAPACK_FOUND PARENT_SCOPE)
set(PLUMED_LD_FLAGS ${PLUMED_LD_FLAGS} PARENT_SCOPE)
set(PLUMED_DYNAMIC_LIBS ${PLUMED_DYNAMIC_LIBS} PARENT_SCOPE)
# print_target_property(${module_name} COMPILE_OPTIONS)
# print_target_property(${module_name} COMPILE_DEFINITIONS)
# print_target_property(${module_name} INTERFACE_COMPILE_DEFINITIONS)
