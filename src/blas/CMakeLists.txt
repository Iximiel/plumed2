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
    include(CheckCXXSourceRuns)
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
    set(CMAKE_REQUIRED_DEFINITIONS -DSDOTtype=float)
    if(noUnderscodeBlas)
        target_compile_definitions(${module_name}
            PUBLIC
            F77_NO_UNDERSCORE)
        set(CMAKE_REQUIRED_DEFINITIONS -DF77_NO_UNDERSCORE -DSDOTtype=float)
    elseif(NOT noUnderscodeBlas AND NOT UnderscodeBlas)
        message(WARNING "both srtog_ and srtog are not linkable")
    # elseif(noUnderscodeBlas AND UnderscodeBlas)
        #     message(WARNING "both srtog_ and srtog are linkable")
        # if both can be linked, use the underscore version
    endif(noUnderscodeBlas)
    set (sdot_check_source     
        [[
#if ! defined(F77_NO_UNDERSCORE)
#define sdot sdot_
#endif
extern "C"{
    SDOTtype sdot(int *n, float *dx, int *incx, float *dy, int *incy);
}

int main(){
    int size=5;
    float af[5],bf[5];
    for(unsigned i=0;i<size;i++){
    af[i]=size;
    bf[i]=size-i;
    }
    int inca=1;
    int incb=1;
    SDOTtype f=sdot(&size,af,&inca,bf,&incb)-75;
    if(f*f<1e-10) return 0;
    else return 1;
}
        ]] )
    check_cxx_source_runs("${sdot_check_source}" sdot_returns_float)
    if(NOT sdot_returns_float)    
        set(CMAKE_REQUIRED_DEFINITIONS -DSDOTtype=double)
        if(noUnderscodeBlas)
            set(CMAKE_REQUIRED_DEFINITIONS -DF77_NO_UNDERSCORE -DSDOTtype=double)
        endif()
        check_cxx_source_runs("${sdot_check_source}" sdot_returns_double)
        if(sdot_returns_double)
            message([[Setting workaround for blas float functions returning double]])
            target_compile_definitions(${module_name}
                PUBLIC
                __PLUMED_BLAS_RETURNS_FLOAT=double)
            #   if test -n "$lapack_found" ; then
            message([[Setting workaround for lapack float functions returning double]])
            #this will be inherited by lapack
            target_compile_definitions(${module_name}
                    PUBLIC
                    __PLUMED_LAPACK_RETURNS_FLOAT=double)
        else()
  #the check is unnecessary
#else if test "$sdot_returns_float" = no && test "$sdot_returns_double" = no ;

  message(WARNING [[There is a problem with your blas implementation]])
endif()
    endif()


        
    unset(CMAKE_REQUIRED_LIBRARIES)
    unset(CMAKE_REQUIRED_DEFINITIONS)
    

    target_link_libraries(${module_name}
        PUBLIC
        BLAS::BLAS)
    list(APPEND PLUMED_DYNAMIC_LIBS ${BLAS_LIBRARIES})
    list(APPEND PLUMED_LD_FLAGS ${BLAS_LINKER_FLAGS})
    
else()
    set(BLAS_FOUND OFF)
endif (BLAS_FOUND)

set(BLAS_FOUND BLAS_FOUND PARENT_SCOPE)
set(PLUMED_LD_FLAGS ${PLUMED_LD_FLAGS} PARENT_SCOPE)
set(PLUMED_DYNAMIC_LIBS ${PLUMED_DYNAMIC_LIBS} PARENT_SCOPE)
# print_target_property(${module_name} COMPILE_OPTIONS)
# print_target_property(${module_name} COMPILE_DEFINITIONS)
# print_target_property(${module_name} INTERFACE_COMPILE_DEFINITIONS)
