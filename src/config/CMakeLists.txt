set(module_name config)
#this is not a module, this is set up to make easier the dependency interface
set(module_${module_name} ON CACHE INTERNAL
            "always active core module ${module_name}")
#set this before or maybe let only one config e
set(plumed_install_flag ON)
#SOEXT is set in ./src
if(plumed_install_flag)
set(ISINSTALLED "true")
#must go to /lib/plumed
    set(PLUMED_ROOT ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/${program_name})
    #CMAKE_INSTALL_DATAROOTDIR defaults to "share"
    #set (htmldir ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DOCDIR})
    set (htmldir ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATAROOTDIR}/doc/${program_name})
    set (includedir ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})
    set (program_name ${program_name})
    set (libdir ${libdir})
    configure_file(Config.inc.in ConfigInstall.inc)
    add_library(Config OBJECT ConfigInstall.cpp Makefile.conf.xxd)
else()
    set(ISINSTALLED "false")
    set (PLUMED_ROOT ${CMAKE_SOURCE_DIR})
    set (htmldir "xxxxNAxxxx")
    set (includedir "xxxxNAxxxx")
    set (program_name "xxxxNAxxxx")
    configure_file(Config.inc.in Config.inc)
    add_library(Config OBJECT Config.cpp Makefile.conf.xxd)
endif(plumed_install_flag)
#TODO: verify if the version number are not set
configure_file(version.h.in version.h)
################################################################################
#The config interface library holds a lot of options that must be inherited by #
#everithing, more specialized options will be found in the respective modules  #
# like blas and lapack
################################################################################
add_library(config INTERFACE)
################################################################################
#                     options and external libraries                           #
################################################################################
macro(PlumedLibsCheck Name)
    #this sets a check for installed libraries that will be used in the package 
    #installing procedure
    if(${Name}_FOUND)
        set(${Name}_FOUND ON PARENT_SCOPE)
        set(${Name}_VERSION ${${Name}_VERSION} PARENT_SCOPE)
        mark_as_advanced(${Name}_FOUND ${Name}_VERSION)
    endif()
endmacro(PlumedLibsCheck)
find_package(PkgConfig REQUIRED)
PlumedLibsCheck(PkgConfig)
include(CheckSourceCompiles)
include(CheckSourceRuns)
include(CheckCXXSymbolExists)
include(CMakePushCheckState)
#  --enable-mpi            enable search for mpi, default: yes
option(enable_mpi "enable search for mpi" ON)

if(enable_mpi)

    find_package(MPI)
    PlumedLibsCheck(MPI)
    
    CONFIGSETTINGS(config MPI_CXX_FOUND
        LINK_LIBRARIES MPI::MPI_CXX
        COMPILE_DEFINITIONS __PLUMED_HAS_MPI
    )
endif(enable_mpi)

option(enable_dlopen "enable search for dlopen" ON)
if(enable_dlopen)
    if(CMAKE_DL_LIBS )
        CONFIGSETTINGS(config CMAKE_DL_LIBS
            LINK_LIBRARIES "${CMAKE_DL_LIBS}"
            COMPILE_DEFINITIONS __PLUMED_HAS_DLOPEN
        )
    if(CMAKE_DL_LIBS STREQUAL "dl")
        list(APPEND PLUMED_LIBS -ldl)
        else()
        list(APPEND PLUMED_LIBS ${CMAKE_DL_LIBS})
    endif()
    endif(CMAKE_DL_LIBS)
endif(enable_dlopen)

#  --enable-rtld_default   enable search for RTLD_DEFAULT macro, default: yes
option(enable_rtld_default "enable search for RTLD_DEFAULT macro" ON)
if(enable_rtld_default)
#     cmake_push_check_state(RESET)
#     set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_DL_LIBS})
#     #this may be put inside of if(enable_dlopen) for consistency
#     check_source_compiles(CXX
# [[#include <dlfcn.h>
# int main (){
#   void* f=dlsym(RTLD_DEFAULT,"path");
#   return 0;
# }]]
#     HAVE_RTLD_DEFAULT
#     )
#     cmake_pop_check_state()
    check_cxx_symbol_exists (RTLD_DEFAULT dlfcn.h HAVE_RTLD_DEFAULT)
    CONFIGSETTINGS(config HAVE_RTLD_DEFAULT
        COMPILE_DEFINITIONS __PLUMED_HAS_RTLD_DEFAULT
        )
endif(enable_rtld_default)

option(enable_fftw "enable search for fftw3" ON)
if(enable_fftw)
    #consider to make this more elegant: see lammps or eigen or use
    # https://github.com/egpbos/findFFTW
    pkg_search_module(FFTW fftw3 IMPORTED_TARGET)
    PlumedLibsCheck(FFTW)
    CONFIGSETTINGS(config FFTW_FOUND
        LINK_LIBRARIES PkgConfig::FFTW
        COMPILE_DEFINITIONS __PLUMED_HAS_FFTW
        PLUMED_DYNAMIC_LIBS ${FFTW_LDFLAGS}
    )
endif(enable_fftw)

option(enable_gsl "enable search for gsl" ON)
if(enable_gsl)
    find_package(GSL)
    PlumedLibsCheck(GSL)
    CONFIGSETTINGS(config GSL_FOUND
            LINK_LIBRARIES GSL::gsl GSL::gslcblas
            COMPILE_DEFINITIONS __PLUMED_HAS_GSL
            PLUMED_DYNAMIC_LIBS ${GSL_LDFLAGS}
    )
endif(enable_gsl)

option(enable_zlib "enable search for zlib" ON)
if(enable_zlib)
    find_package(ZLIB)
    PlumedLibsCheck(ZLIB)

    CONFIGSETTINGS(config ZLIB_FOUND
            LINK_LIBRARIES ZLIB::ZLIB
            COMPILE_DEFINITIONS __PLUMED_HAS_ZLIB
            PLUMED_DYNAMIC_LIBS ${ZLIB_LIBRARY_RELEASE}
    )
endif(enable_zlib)

#  --enable-cregex         enable search for C regular expression, default: yes
option(enable_cregex "enable search for C regular expression" ON)
if(enable_cregex)
#     find_path(REGEX_HEADER_DIR NAMES regex.h)
#     if(REGEX_HEADER_DIR)
#         check_source_compiles(CXX
# [[extern "C"
#     char regcomp ();
#     int main(void) {
#         return regcomp ();
#     }]]
#     HAVE_REGCOMP
#         )
#     endif(REGEX_HEADER_DIR)
    check_cxx_symbol_exists (regcomp regex.h HAVE_REGCOMP)
    CONFIGSETTINGS(config HAVE_REGCOMP
        COMPILE_DEFINITIONS __PLUMED_HAS_CREGEX
    )
endif(enable_cregex)

#  --enable-chdir          enable search for chdir function, default: yes
option(enable_chdir  "enable search for chdir function" ON)
if(enable_chdir)
#     find_path(unistd_HEADER_DIR NAMES unistd.h)
#     if(unistd_HEADER_DIR)
#         check_source_compiles(CXX
# [[extern "C"
#     char chdir ();
#     int main(void) {
#         return chdir ();
#     }]]
#         HAVE_CHDIR
#         )
#     endif(unistd_HEADER_DIR)
    check_cxx_symbol_exists (chdir unistd.h HAVE_CHDIR)
    CONFIGSETTINGS(config HAVE_CHDIR
        COMPILE_DEFINITIONS __PLUMED_HAS_CHDIR
    )
endif(enable_chdir)

#  --enable-getcwd         enable search for getcwd function, default: yes
option(enable_getcwd   "enable search for getcwd function" ON)
if(enable_getcwd )
    # find_path(unistd_HEADER_DIR NAMES unistd.h)
    # if(unistd_HEADER_DIR)
    #     check_source_compiles(CXX
    # [[extern "C"
    # char getcwd  ();
    # int main(void) {
    #     return getcwd  ();
    # }]]
    #     COMPILES_getcwd 
    #     )
    # endif(unistd_HEADER_DIR)
    check_cxx_symbol_exists (getcwd  unistd.h HAVE_GETCWD)
    CONFIGSETTINGS(config HAVE_GETCWD 
        COMPILE_DEFINITIONS __PLUMED_HAS_GETCWD 
    )
endif(enable_getcwd )

#  --enable-popen          enable search for popen, default: yes
option(enable_popen  "enable search for popen" ON)
if(enable_popen)
    # find_path(cstdio_HEADER_DIR NAMES cstdio)
    # if(cstdio_HEADER_DIR)
    #     check_source_compiles(CXX
    # [[extern "C"
    # char popen ();
    # int main(void) {
    #     return popen ();
    # }]]
    # HAVE_POPEN
    #     )
    # endif(cstdio_HEADER_DIR)
    check_cxx_symbol_exists (popen  cstdio HAVE_POPEN)
    CONFIGSETTINGS(config HAVE_POPEN
        COMPILE_DEFINITIONS __PLUMED_HAS_POPEN
    )
endif(enable_popen)

#  --enable-execinfo       enable search for execinfo, default: yes
option(enable_execinfo  "enable search for execinfo" ON)
if(enable_execinfo)
    find_package(Backtrace)
    PlumedLibsCheck(Backtrace)
    CONFIGSETTINGS(config Backtrace_FOUND
        COMPILE_DEFINITIONS __PLUMED_HAS_EXECINFO
        LINK_LIBRARIES "${Backtrace_LIBRARY}"
    )
endif(enable_execinfo)

#  --enable-subprocess     enable search for functions needed to manage a
#                          subprocess, default: yes
option(enable_subprocess  "enable search for functions needed to manage a subprocess" ON)
if(enable_subprocess)
    check_source_compiles(CXX
[[#include <cstdio>
#include <unistd.h>
#include <csignal>

int main () {
// notice that [] is required to cheat autoconf
    int cp[2];
    FILE* fp;
    char* arr[3];
    arr[0]=NULL;
    arr[1]=NULL;
    arr[2]=NULL;
    arr[3]=NULL;
    if(pipe(cp)>=0)
    if(fork()>=0)
    if(close(1)>=0)
    if(dup(cp[1])>=0) {
        fp=fdopen(cp[0],"w");
        execv(arr[0],arr);
    }
    auto p=fork();
    kill(p,SIGINT);
    kill(p,SIGCONT);
    kill(p,SIGSTOP);
    return 0;
}]]
        HAVE_SUBPROCESS_MANAGEMENT
    )

    CONFIGSETTINGS(config HAVE_SUBPROCESS_MANAGEMENT
        COMPILE_DEFINITIONS __PLUMED_HAS_SUBPROCESS
        
    )
endif(enable_subprocess)
##############################################################################
#TODO:IMPLEMENT CORRECTLY
# This is not needed in the main plumed but only for compiling the pip interface
#  --enable-python         enable search for python, default: yes
# option(enable_execinfo  "enable search for python" ON)
# if(enable_execinfo)
#     find_package (Python COMPONENTS Interpreter Development)
# endif(enable_execinfo)
##############################################################################

################################################################################
#               default off options and external libraries                     #
################################################################################

#  --enable-boost_graph    enable search for boost graph, default: no
option(enable_boost_graph "enable search for boost graph" OFF)
if(enable_boost_graph)
    find_package(Boost
        COMPONENTS graph
    )
    PlumedLibsCheck(Boost_GRAPH)
    CONFIGSETTINGS(config Boost_GRAPH_FOUND
        LINK_LIBRARIES Boost::graph
        COMPILE_DEFINITIONS __PLUMED_HAS_BOOST_GRAPH
    )
endif(enable_boost_graph)

#  --enable-boost_serialization
#                          enable search for boost serialization, default: no
option(enable_boost_serialization "enable search for boost serialization" OFF)
if(enable_boost_serialization)
    find_package(Boost
        COMPONENTS serialization
    )
    PlumedLibsCheck(Boost_SERIALIZATION)
    CONFIGSETTINGS(config Boost_SERIALIZATION_FOUND
        LINK_LIBRARIES Boost::serialization
        COMPILE_DEFINITIONS __PLUMED_HAS_BOOST_SERIALIZATION
    )
endif(enable_boost_serialization)

#  --enable-readdir-r      enable search for readdir_r (threadsafe), default:
#                          no
option(enable_readdir_r "enable search for readdir_r (threadsafe)" OFF)
if(enable_readdir_r)
    #this check for the function in the dirent.h header,
    #(and also if the header is present)
    check_cxx_symbol_exists (readdir_r dirent.h HAVE_READDIR_R)
    
    CONFIGSETTINGS(config HAVE_READDIR_R
        COMPILE_DEFINITIONS __PLUMED_HAS_READDIR_R
    )
endif(enable_readdir_r)

##############################################################################
#TODO:IMPLEMENT CORRECTLY
#  --enable-absolute-soname
#                          enable store absolute soname (Linux only - this is
#                          the default behavior on OSX). Only enable for
#                          testing!, default: no
option(enable_absolute_soname
    "enable store absolute soname (Linux only - this is the default behavior 
    on OSX). Only enable for testing!, default: no"
    OFF)
##############################################################################

option(enable_bsymbolic "enable the flag -Wl,-Bsymbolic at link time" ON)
if(enable_bsymbolic)
    set(BsymbolicFLAGS "-Wl,-Bsymbolic")
    list(APPEND CMAKE_REQUIRED_LIBRARIES ${BsymbolicFLAGS})
	check_source_compiles(CXX
	[=[
	int main ()
	{
	  return 0;
	}
	]=]
	HAVE_BSYMBOLIC
	)
	list(REMOVE_ITEM CMAKE_REQUIRED_LIBRARIES ${BsymbolicFLAGS})
	if (HAVE_BSYMBOLIC)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${BsymbolicFLAGS}")
        list(APPEND PLUMED_DYNAMIC_LIBS ${BsymbolicFLAGS})
        else()
        message(WARNING "${BsymbolicFLAGS} cannot be enabled in the compiler")
    endif(HAVE_BSYMBOLIC)
endif(enable_bsymbolic)
#  --disable-openmp        do not use OpenMP
option(enable_openmp "enable search for openmp" ON)
if(enable_openmp)
    find_package(OpenMP)
    PlumedLibsCheck(OpenMP)
    CONFIGSETTINGS(config OpenMP_CXX_FOUND
        LINK_LIBRARIES OpenMP::OpenMP_CXX
        PLUMED_DYNAMIC_LIBS ${OpenMP_CXX_FLAGS}
    )
    #if(OpenMP_CXX_FOUND)
    #    target_link_libraries(config INTERFACE OpenMP::OpenMP_CXX)
    #endif(OpenMP_CXX_FOUND)
endif(enable_openmp)
################################################################################
#                            compilers                                         #
################################################################################
if (CMAKE_CXX_STANDARD GREATER_EQUAL 11)
    target_compile_definitions(config INTERFACE __PLUMED_LIBCXX11=1)
endif()

#some alwais on definitions
target_compile_definitions(config
      INTERFACE
      $<$<CONFIG:Release>:NDEBUG>
      $<$<CONFIG:MinSizeRel>:NDEBUG>
      # from the original config:
      # this is necessary in many MPI implementations
      # I leave it by default, since it seems harmless
      _REENTRANT=1
)

#the build_interface makes sure that everithing that depends on config can include to anythinf in builded plumed
#the install_interfaces does the same but with the installed plumed
target_include_directories(config INTERFACE
    # these are relative to ${CMAKE_INSTALL_PREFIX}, in this way config is relocatable after install
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>#THESE ARE THE INCLUDE PATH for the make tests and standard code
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/plumed>#THESE ARE THE INCLUDE PATH for PLUGINS

    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/includesource>#THESE ARE THE INCLUDE PATH for the make tests and standard code (this make possible the use of `#include "plumed/subdir/header.h"`)
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/includebinary>#THESE ARE THE INCLUDE PATH for the make tests and standard code (this make possible the use of `#include "plumed/subdir/header.h"`)
    $<BUILD_INTERFACE:${PLUMED_SOURCE_SRC}>#THESE ARE THE INCLUDE PATH for PLUGINS
    $<BUILD_INTERFACE:${PLUMED_BINARY_SRC}>#THESE ARE THE INCLUDE PATH for PLUGINS
    )

#This makes the `#include "plumed/subdir/header.h"` avaiable in a not instaled plumed 
file(MAKE_DIRECTORY
    ${CMAKE_BINARY_DIR}/includesource
    ${CMAKE_BINARY_DIR}/includebinary)

file(CREATE_LINK ${PLUMED_SOURCE_SRC} ${CMAKE_BINARY_DIR}/includesource/plumed
    SYMBOLIC)
file(CREATE_LINK ${PLUMED_BINARY_SRC} ${CMAKE_BINARY_DIR}/includebinary/plumed
    SYMBOLIC)
################################################################################

target_link_libraries(Config PUBLIC config)
target_include_directories(Config PRIVATE 
    ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

configure_file(Makefile.conf.xxd.compiler.sh.in
                 Makefile.conf.xxd.compiler.sh @ONLY)
configure_file(${PLUMED_MAIN_DIR}/Makefile.conf.in
                 Makefile.cmakeMocked.conf @ONLY)
add_custom_command(OUTPUT Makefile.conf.xxd
    DEPENDS ${PLUMED_MAIN_DIR}/Makefile.conf.in
    COMMAND bash Makefile.conf.xxd.compiler.sh
    COMMENT "Generationc completion.xxd"
    #WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_property(TARGET Config
    APPEND PROPERTY PUBLIC_HEADER
    ${CMAKE_CURRENT_BINARY_DIR}/version.h Config.h)
install (TARGETS Config config
    EXPORT Plumed2MODULES
    PUBLIC_HEADER
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${program_name}/config
)
#print_target_property(ConfigInstall INCLUDE_DIRECTORIES)

#adding -lstdc++ -lc++ is done automatically by cmake
#this is set up to replicate the behaviour of creating
#in the original make, for the various Plumed.inc
#  --enable-libsearch      enable search for libraries, default: yes
option(enable_libsearch "enable search for libraries" ON)
if(enable_libsearch)
    foreach(testlib "-lstdc++" "-lc++" "-lmpi_cxx")
    check_source_runs(C
"#include <stdlib.h>
void f(){}
int main(void) { f(); return 0; }"
    HAVE_${testlib})
    if (HAVE_${testlib})
        list(APPEND PLUMED_DYNAMIC_LIBS ${testlib})
        #adds only the first c++ main library
        break()
    endif (HAVE_${testlib})
    endforeach(testlib)
endif(enable_libsearch)

if(CMAKE_ENABLE_EXPORTS)
    #this is maybe a bit too crude
    list(APPEND PLUMED_DYNAMIC_LIBS -rdynamic)
endif(CMAKE_ENABLE_EXPORTS)
set(PLUMED_LD_FLAGS ${PLUMED_LD_FLAGS} PARENT_SCOPE)
set(PLUMED_LIBS ${PLUMED_LIBS} PARENT_SCOPE)
set(PLUMED_DYNAMIC_LIBS ${PLUMED_DYNAMIC_LIBS} PARENT_SCOPE)
set(CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS} PARENT_SCOPE)
