#here we redefine some variables,
#the scope of these variables is this file only
#the renaming is needed to align the procedure with the standard Makefile

set(_SOEXT_ ${SOEXT})
set(_program_name_ ${program_name})
set(_libdir_ ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
set(_bindir_ ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
set(_includedir_ ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})
if(enable_absolute_soname)
    set(_use_absolute_soname_ "yes")
else()
    set(_use_absolute_soname_ "no")
endif(enable_absolute_soname)
set(exec_prefix ${CMAKE_INSTALL_PREFIX})
configure_file(modulefile.in modulefile @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/modulefile
DESTINATION ${CMAKE_INSTALL_LIBDIR}/${program_name}
PERMISSIONS
 OWNER_WRITE
 OWNER_READ GROUP_READ WORLD_READ
)
set( _libs_ "")
#get_target_property(_private_libs_ plumed_bin COMPILE_DEFINITIONS)
                   # -D$<JOIN:$<TARGET_PROPERTY:INTERFACE_COMPILE_DEFINITIONS>, -D>
set( _private_libs_ "-l$<JOIN:$<TARGET_PROPERTY:COMPILE_FLAGS>, -l>")#(LIBS) $(DYNAMIC_LIBS))
#set( _private_libs_ "-l$<JOIN:$<TARGET_LINKER_FILE_NAME>, -l>")
#set( _private_libs_ $-D$<JOIN:$_private_libs_>, -D>)
set( _prefix_ ${CMAKE_INSTALL_PREFIX})
set( _exec_prefix_ ${exec_prefix})
set( _VERSION_ ${PLUMED_VERSION_LONG})
set( _conflicts_ "")
set( _cppflags_ "")
set( _extra_ "")

configure_file(pkgconfig.in ${program_name}.pc.in @ONLY)
file(GENERATE OUTPUT ${program_name}.pc
    INPUT ${CMAKE_CURRENT_BINARY_DIR}/${program_name}.pc.in TARGET lapack)


set(_libs_ $(LIBS_PLUMED_KERNEL))
set(_private_libs_ $(LIBS) $(DYNAMIC_LIBS))
set( _cppflags_ "-D$<JOIN:$<TARGET_PROPERTY:COMPILE_DEFINITIONS>, -D>")
set(_extra_  "(full version, including internals)")

configure_file(pkgconfig.in ${program_name}Internals.pc.in @ONLY)
file(GENERATE OUTPUT ${program_name}Internals.pc
    INPUT ${CMAKE_CURRENT_BINARY_DIR}/${program_name}Internals.pc.in TARGET plumed_bin)
set( t [[        
set(_program_name_ ${program_name}Wrapper)
set(_libs_ $(LIBS))
set(_private_libs_ "")
set(_cppflags_ "")
set(_extra_  "(wrapper-only version, set PLUMED_KERNEL at runtime)")
> install/pkgconfigWrapper

configure_file(pkgconfig.in ${program_name}Wrapper.pc @ONLY)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${program_name}.pc
    ${CMAKE_CURRENT_BINARY_DIR}/${program_name}Internals.pc
    ${CMAKE_CURRENT_BINARY_DIR}/${program_name}Wrapper.pc
DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
    PERMISSIONS
    OWNER_WRITE
    OWNER_READ GROUP_READ WORLD_READ
)
]])

set(listModules)
set(PLUMED_HAS)
set(PLUMED_HAS_wip)
#PLUMED_HAS_not is a wip, need to be more automated
set(PLUMED_HAS_not
__PLUMED_HAS_ARRAYFIRE_CUDA
__PLUMED_HAS_ARRAYFIRE_OCL
__PLUMED_HAS_ARRAYFIRE
__PLUMED_HAS_ASMJIT
__PLUMED_HAS_BOOST_GRAPH
__PLUMED_HAS_BOOST_SERIALIZATION
__PLUMED_HAS_CHDIR
__PLUMED_HAS_CREGEX
__PLUMED_HAS_DLOPEN
__PLUMED_HAS_EXECINFO
__PLUMED_HAS_EXTERNAL_BLAS
__PLUMED_HAS_EXTERNAL_LAPACK
__PLUMED_HAS_EXTERNAL_MOLFILE_PLUGINS
__PLUMED_HAS_FFTW
__PLUMED_HAS_GETCWD
__PLUMED_HAS_GSL
__PLUMED_HAS_LIBTORCH
__PLUMED_HAS_MOLFILE_PLUGINS
__PLUMED_HAS_MPI
__PLUMED_HAS_POPEN
__PLUMED_HAS_PYTHON
__PLUMED_HAS_READDIR_R
__PLUMED_HAS_RTLD_DEFAULT
__PLUMED_HAS_SUBPROCESS
__PLUMED_HAS_ZLIB)
macro(CHECKHASDEFINES module)
    #message("CHECKHASDEFINES ${module}")
    if(TARGET ${module})
        get_target_property(module_has ${module} COMPILE_DEFINITIONS)
        foreach(def ${module_has})
            string(REGEX MATCH "__PLUMED_HAS_[A-Z_]*" t ${def}) 
            list(APPEND PLUMED_HAS_wip ${t})
            list(REMOVE_ITEM PLUMED_HAS_not ${t})
        endforeach()
        get_target_property(module_has ${module} INTERFACE_COMPILE_DEFINITIONS)
        foreach(def ${module_has})
            string(REGEX MATCH "__PLUMED_HAS_[A-Z_]*" t ${def}) 
            list(APPEND PLUMED_HAS_wip ${t})
            list(REMOVE_ITEM PLUMED_HAS_not ${t})
        endforeach()
    endif(TARGET ${module})
endmacro(CHECKHASDEFINES)

foreach(dir ${module_dirs})
    if(${module_default_${dir}} STREQUAL "always")
        set (default "always")
    elseif(${module_default_${dir}})
        set (default "default-on")
    else()
        set (default "default-off")
    endif()
    set (temp "module ${dir} ${module_${dir}} (${default})")
    string(TOLOWER ${temp} temp)
    list(APPEND listModules ${temp})
    CHECKHASDEFINES(${dir})
endforeach(dir ${module_dirs})

CHECKHASDEFINES(config)
list(REMOVE_DUPLICATES listModules)
list(SORT listModules)
list(REMOVE_DUPLICATES PLUMED_HAS_wip)


foreach(t ${PLUMED_HAS_wip})
    string(REGEX REPLACE "__PLUMED_HAS_"  "" t ${t})
    string(TOLOWER ${t} t)
    LIST(APPEND PLUMED_HAS "has ${t} on")
endforeach(t ${PLUMED_HAS_wip})

foreach(t ${PLUMED_HAS_not})
    string(REGEX REPLACE "__PLUMED_HAS_"  "" t ${t})
    string(TOLOWER ${t} t)
    LIST(APPEND PLUMED_HAS "has ${t} off")
endforeach(t ${PLUMED_HAS_wip})
list(SORT PLUMED_HAS)
list(JOIN PLUMED_HAS "\n" PLUMED_HAS)

list(JOIN listModules "\n" CONTENTS)


if (MPIEXEC_EXECUTABLE)
    set (MPIEXEC_EXECUTABLE_name ${MPIEXEC_EXECUTABLE})
endif()
file(CONFIGURE OUTPUT config.txt
     CONTENT [[
# version strings
# syntax: version short/long/git number
version short @PLUMED_VERSION_SHORT@
version long @PLUMED_VERSION_LONG@
version git @PLUMED_VERSION_GIT@

# python executable
# syntax: python_bin executable
# empty string means that python has not been configured
python_bin @Python_EXECUTABLE@

# command to lauch mpi processes
# syntax: mpiexec command
# empty string means that mpiexec was not chosen at configure time
mpiexec @MPIEXEC_EXECUTABLE_name@

# list of 'has' options
# syntax: has name on/of
# if option xx is on then plumed has been compiled with -D__PLUMED_HAS_XX
@PLUMED_HAS@

# other defines
# syntax: define name=value
@PLUMED_DEFINES@

# list of modules
# syntax: module name on/off (default-on/default-off/always)
@CONTENTS@

# Makefile.conf file
# syntax: makefile_conf followed by a single space followed by a line from makefile_conf
]]
)
set(listHASModules)
#reconstrunct the module for the Config_lib.inc
foreach(dir ${module_dirs})
    if(${module_default_${dir}} STREQUAL "always")
        set (status "always")
    elseif(${module_${dir}})
        set (status "on")
    else()
        set (status "off")
    endif()
    string(TOLOWER ${dir} tmp)
    set (temp "if (moduleName == \"${tmp}\"){return presence::${status}")
    
    list(APPEND listHASModules ${temp})
endforeach(dir ${module_dirs})
message("${listHASModules}")
list(JOIN listHASModules ";}\n" CONTENTS)
file(CONFIGURE OUTPUT ../config/Config_lib.inc
     CONTENT [[
presence plumedHasFeature(std::string featureName){
    @listHASFeatures@
    return presence::notFound;
}
     
presence plumedHasModule(std::string moduleName){
@CONTENTS@;}
return presence::notFound;
}
std::string getMPI_EXEC(){
    return "@MPIEXEC_EXECUTABLE@";
}
]]
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/config.txt
DESTINATION ${CMAKE_INSTALL_LIBDIR}/${program_name}/src/config
PERMISSIONS
 OWNER_WRITE
 OWNER_READ GROUP_READ WORLD_READ
)

#the list are not modified in the parent scope
foreach(v PLUMED_LIBS  PLUMED_DYNAMIC_LIBS)
    set(t "")
    foreach(s ${${v}})
        string(REGEX REPLACE ".*lib" "-l" tt ${s})
        string(REGEX REPLACE "(\.so$|\.a$)" "" tt ${tt})
        list(APPEND t ${tt})
    endforeach()
    #list(SORT t)
    list(JOIN t " " ${v})

endforeach(v )

#list(JOIN PLUMED_LIBS " " PLUMED_LIBS)
#list(JOIN PLUMED_DYNAMIC_LIBS " " PLUMED_DYNAMIC_LIBS)
list(JOIN PLUMED_LD_FLAGS " " PLUMED_LD_FLAGS)
file(REAL_PATH ${CMAKE_INSTALL_PREFIX} CMAKE_INSTALL_PATH)
#$(LIBS) and $(LDFLAGS) are not so easy to get without using strange tricks
set (Plumed_inc "")
set (Plumed_inc_runtime [[
PLUMED_RUNTIME_LOAD= "@CMAKE_INSTALL_PATH@/@CMAKE_INSTALL_LIBDIR@/lib@program_name@Wrapper.a" @PLUMED_LIBS@ @PLUMED_LD_FLAGS@
PLUMED_RUNTIME_DEPENDENCIES= ""
]])
set (Plumed_inc_static [[
PLUMED_STATIC_LOAD= "@CMAKE_INSTALL_PATH@/@CMAKE_INSTALL_LIBDIR@/lib@program_name@.a" @PLUMED_LIBS@ @PLUMED_DYNAMIC_LIBS@ @PLUMED_LD_FLAGS@
PLUMED_STATIC_DEPENDENCIES= "@CMAKE_INSTALL_PATH@/@CMAKE_INSTALL_LIBDIR@/lib@program_name@.a"
]])
set (Plumed_inc_shared [[
PLUMED_SHARED_LOAD= "@CMAKE_INSTALL_PATH@/@CMAKE_INSTALL_LIBDIR@/lib@program_name@.@SOEXT@" @PLUMED_LIBS@ @PLUMED_LD_FLAGS@
PLUMED_SHARED_DEPENDENCIES= "@CMAKE_INSTALL_PATH@/@CMAKE_INSTALL_LIBDIR@/lib@program_name@.@SOEXT@"
]])
set (Plumed_inc_fortran [[
PLUMED_FORTRAN= "@CMAKE_INSTALL_PATH@/@CMAKE_INSTALL_LIBDIR@/@program_name@/fortran/plumed.f90"
PLUMED_FORTRAN08= "@CMAKE_INSTALL_PATH@/@CMAKE_INSTALL_LIBDIR@/@program_name@/fortran/plumed_f08.f90"
]])
###
###TODO: the toggle for the runtime plumed
###
if (ON)
    string (APPEND Plumed_inc ${Plumed_inc_runtime})
endif()
###
###TODO: the toggle for the static plumed
###
if (ON)
string (APPEND Plumed_inc ${Plumed_inc_static})
endif()
string (APPEND Plumed_inc ${Plumed_inc_shared} ${Plumed_inc_fortran})

file(CONFIGURE OUTPUT Plumed.inc
    CONTENT ${Plumed_inc}
)
foreach(type runtime shared static)
    set (Plumed_inc "")
    string (APPEND Plumed_inc ${Plumed_inc_${type}} ${Plumed_inc_fortran})
    file(CONFIGURE OUTPUT Plumed.inc.${type}
        CONTENT ${Plumed_inc}
    )    
endforeach(type runtime shared static)
