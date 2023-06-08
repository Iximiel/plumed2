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
foreach(dir ${toggabledirs} ${mayBeExternal})
    set (temp "module ${dir} ${module_${dir}} (default-?)")
    string(TOLOWER ${temp} temp)
    list(APPEND listModules ${temp})
endforeach(dir ${toggabledirs} ${mayBeExternal})
foreach(dir core tools lepton blas lapack asmjit)
    set (temp "module ${dir} ${module_${dir}} (always)")
    string(TOLOWER ${temp} temp)
    list(APPEND listModules ${temp})
endforeach(dir core tools lepton blas lapack asmjit)
list(SORT listModules)

SET(CONTENTS "")
FOREACH(line IN LISTS listModules)
   SET(CONTENTS "${CONTENTS}${line}\n")
ENDFOREACH(line)
#message(${CONTENTS})
#set(toggabledirs ${dirs})
#list(REMOVE_ITEM toggabledirs core tools lepton blas lapack asmjit)
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

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/config.txt
DESTINATION ${CMAKE_INSTALL_LIBDIR}/${program_name}/src/config
PERMISSIONS
 OWNER_WRITE
 OWNER_READ GROUP_READ WORLD_READ
)
