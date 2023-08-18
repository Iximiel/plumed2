#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "blas")
#Note that the macros here require this directory added as a subdir of plumed/src
option(useExternal_BLAS "enable search for external BLAS, default ON" ON)
DECLAREPLUMEDMODULE(${module_name}
"always"
SOURCES
blas.cpp
EXTRA_HEADERS
def_external.h
def_internal.h
real.h
)

#include(FortranCInterface)
if (useExternal_BLAS)
    find_package(BLAS)
endif (useExternal_BLAS)

if (BLAS_FOUND)
    target_compile_definitions(${module_name}
        PUBLIC
        __PLUMED_HAS_EXTERNAL_BLAS=1)
    include(CheckCXXSourceCompiles)
    #From the manual:The check is only performed once, with the result cached in
    #the variable named by <resultVar>. Every subsequent CMake run will re-use
    #this cached value rather than performing the check again, even if the
    #<code> changes. In order to force the check to be re-evaluated, the
    #variable named by <resultVar> must be manually removed from the cache.
    set(CMAKE_REQUIRED_LIBRARIES BLAS::BLAS)
    check_cxx_source_compiles(
    [[extern "C"
        void srotg_(double*,double*,double*,double*);
        int main(void) {
            double a,b,c,s;
            srotg_(&a,&b,&c,&s);
        }
    ]]
    UnderscodeBlas)
    check_cxx_source_compiles(
    [[extern "C"
        void srotg(double*,double*,double*,double*);
        int main(void) {
            double a,b,c,s;
            srotg(&a,&b,&c,&s);
        }
    ]]
    noUnderscodeBlas
    )
    unset(CMAKE_REQUIRED_LIBRARIES)

    if(noUnderscodeBlas)
        target_compile_definitions(${module_name}
            PUBLIC
            F77_NO_UNDERSCORE)
    elseif(NOT noUnderscodeBlas AND NOT UnderscodeBlas)
        message(WARNING "both srtog_ and srtog are not linkable")
    # elseif(noUnderscodeBlas AND UnderscodeBlas)
    #     message(WARNING "both srtog_ and srtog are linkable")
    # if both can be linked, use the underscore version
    endif(noUnderscodeBlas)

    target_link_libraries(${module_name}
        PUBLIC
        BLAS::BLAS)
    list(APPEND PLUMED_DYNAMIC_LIBS ${BLAS_LIBRARIES})
    list(APPEND PLUMED_LD_FLAGS ${BLAS_LINKER_FLAGS})
    
else()
    set(BLAS_FOUND OFF)
endif (BLAS_FOUND)

set (BLAS_FOUND BLAS_FOUND PARENT_SCOPE)
set(PLUMED_LD_FLAGS ${PLUMED_LD_FLAGS} PARENT_SCOPE)
set(PLUMED_DYNAMIC_LIBS ${PLUMED_DYNAMIC_LIBS} PARENT_SCOPE)
# print_target_property(${module_name} COMPILE_OPTIONS)
# print_target_property(${module_name} COMPILE_DEFINITIONS)
# print_target_property(${module_name} INTERFACE_COMPILE_DEFINITIONS)
