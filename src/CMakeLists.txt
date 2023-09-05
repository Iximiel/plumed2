cmake_minimum_required(VERSION 3.20.2)

project(Plumed2 LANGUAGES C CXX)

set (CMAKE_CXX_STANDARD 11)

include(GNUInstallDirs)

#some settingsvalues
set(program_name "plumed" CACHE STRING "the name of the main executable")
string(REPLACE "." "" soextension ${CMAKE_SHARED_LIBRARY_SUFFIX})

set(SOEXT ${soextension} CACHE INTERNAL "The extension of dynamic libraries (so/dylib)")
#set_property(CACHE SOEXT PROPERTY STRINGS "so" "dylib")

set (PLUMED_SOURCE_SRC ${CMAKE_CURRENT_SOURCE_DIR})
set (PLUMED_BINARY_SRC ${CMAKE_CURRENT_BINARY_DIR})
set (PLUMED_MAKETOOLS ${PLUMED_SOURCE_SRC}/maketools )
cmake_path (GET PLUMED_SOURCE_SRC PARENT_PATH PLUMED_MAIN_DIR)
set (PLUMED_SCRIPTS ${PLUMED_MAIN_DIR}/scripts)
set (PLUMED_PATCHES ${PLUMED_MAIN_DIR}/patches)

option(allmodules_activate "Activates all modules" OFF)
option(allmodules_default "Revert all module to their default state" OFF)

#this is a list of the activated modules that will be included into the kernel
set(modulesForKernel "")

################################################################################
###########################Setting up global options############################
################################################################################
#these are needed to help in reconstruct the pkgconfigs and the *.inc like in
#the original Makefile
set(PLUMED_LD_FLAGS "")
set(PLUMED_LIBS "")
set(PLUMED_DYNAMIC_LIBS "")

#this may be deactivated when we want to copile something more static
#This automatically adds the correct equivalent to -fPIC to the compiler
option(CMAKE_POSITION_INDEPENDENT_CODE "set the flags for position independent code" ON)
option(CMAKE_ENABLE_EXPORTS "Set the -rdynamic equivalent for your compiler" ON)
#  --enable-basic-warnings enable basic warnings, default: yes
option(enable_warnings_basic "enable basic warnings" ON)
if(enable_warnings_basic)
    if(MSVC)
      add_compile_options(/W4)
    else()
      add_compile_options(-Wall)
    endif(MSVC)
endif(enable_warnings_basic)
#  --enable-fussy          enable fussy warnings, default: no
option(enable_warnings_fussy "enable fussy warnings" OFF)
if(enable_warnings_fussy)
    if(MSVC)
      add_compile_options(/Wall)
    else()
      add_compile_options(        
        -Wall
        -Wextra # reasonable and standard
        -Wshadow # warn the user if a variable declaration shadows one from a# parent context
        -Wnon-virtual-dtor # warn the user if a class with virtual functions has a# non-virtual destructor.
        -Wold-style-cast # warn for c-style casts
        -Wcast-align # warn for potential performance problem casts
        -Wunused # warn on anything being unused
        -Woverloaded-virtual # warn if you overload (not override) a virtual func
        -Wpedantic # warn if non-standard C++ is used
        -Wconversion # warn on type conversions that may lose data
        -Wsign-conversion # warn on sign conversions
        -Wnull-dereference # warn if a null dereference is detected
        -Wdouble-promotion # warn if float is implicit promoted to double
        -Wformat=2 # warn on security issues around functions that format output# (ie printf)
        -Wduplicated-cond # warn if if / else chain has duplicated conditions
        -Wduplicated-branches # warn if if / else branches have duplicated code
        -Wlogical-op # warn about logical operations being used where bitwise were# probably wanted
        -Wuseless-cast # warn if you perform a cast to the same type
        -Wlifetime # ///
          )
    endif(MSVC)
endif(enable_warnings_fussy)
option(enable_warnings_errors "enable stop compilation on warnings" OFF)
if(enable_warnings_errors)
    if(MSVC)
      add_compile_options(/WX)
    else()
      add_compile_options(-Werror -Wfatal-errors)
    endif(MSVC)
endif(enable_warnings_errors)
#  --enable-fussy          enable fussy warnings, default: no


################################################################################
###############################Macro definitions################################
################################################################################
# support directory with .cmake files with useful macros
list(APPEND CMAKE_MODULE_PATH "${PLUMED_SOURCE_SRC}/cmake")
include(plumedCMakeMacros)

################################################################################
###############################Setting up Modules###############################
################################################################################

#getting the list of direcotries in src/
FILE(GLOB children RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
SET(alldirs "")
FOREACH(child ${children})
    IF(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${child})
        LIST(APPEND alldirs ${child})
    ENDIF()
ENDFOREACH()

set(module_dirs ${alldirs})# CACHE INTERNAL FORCE)
#directories that do not contain modules
set(utildirs lib include maketools cmake)
set(execDirs config wrapper) #removed main
#these are directory with codes that may be external
set(mayBeExternal blas lapack molfile)
list(REMOVE_ITEM module_dirs main ${utildirs} ${execDirs})

#the first subdir should is config
foreach(dir ${execDirs})
    add_subdirectory(${dir})
endforeach(dir ${execDirs})

if(DEFINED CACHE{module_core})
    #module_core will not be present on the first run:
    #allmodules_activate will act on the cache correctly during the module declaration
    if(allmodules_activate)
        message("All module will be forced ON")
        foreach(dir ${module_dirs})
            if(${module_default_${dir}} STREQUAL "always")
                continue()
                else()
                set(module_${dir} ON CACHE BOOL "activate module ${dir}" FORCE)
            endif()
        endforeach(dir ${module_dirs})
    endif()

    if(allmodules_default)
        message("All module will be forced to their default status")
        foreach(dir ${module_dirs})
            if(${module_default_${dir}} STREQUAL "always")
                continue()
                else()
                set(module_${dir} ${module_default_${dir}} CACHE BOOL "activate module ${dir}" FORCE)
            endif()
        endforeach(dir ${module_dirs})
    endif()
    else()
    if(allmodules_activate)
        message("All module will be activated")
    endif()
endif(DEFINED CACHE{module_core})

foreach(dir ${module_dirs})
    add_subdirectory(${dir})
endforeach(dir ${module_dirs})

#I set up the "auto switch off" for these options,
#so their won't be executed at each subsequent cmake run
#but they still appear in the cmake cache editor for the user
#(for example: the user may force default, and then activate only one module.
#In this way that module won't be switched off on the next cmake run)
set(allmodules_activate OFF CACHE BOOL "Activates all modules" FORCE)
set(allmodules_default OFF CACHE BOOL "Revert all module to their default state" FORCE)
#adding -lstdc++ -lc++ is done automatically by cmake

#options ## options are already done
##  --bindir=DIR            user executables [EPREFIX/bin]
##  --sbindir=DIR           system admin executables [EPREFIX/sbin]
# --libexecdir=DIR        program executables [EPREFIX/libexec]
##  --sysconfdir=DIR        read-only single-machine data [PREFIX/etc]
#  --sharedstatedir=DIR    modifiable architecture-independent data [PREFIX/com]
#  --localstatedir=DIR     modifiable single-machine data [PREFIX/var]
##  --libdir=DIR            object code libraries [EPREFIX/lib]
##  --includedir=DIR        C header files [PREFIX/include]
#  --oldincludedir=DIR     C header files for non-gcc [/usr/include]
#  --datarootdir=DIR       read-only arch.-independent data root [PREFIX/share]
#  --datadir=DIR           read-only architecture-independent data [DATAROOTDIR]
#  --infodir=DIR           info documentation [DATAROOTDIR/info]
#  --localedir=DIR         locale-dependent data [DATAROOTDIR/locale]
#  --mandir=DIR            man documentation [DATAROOTDIR/man]
#  --docdir=DIR            documentation root [DATAROOTDIR/doc/plumed]
#  --htmldir=DIR           html documentation [DOCDIR]
#  --dvidir=DIR            dvi documentation [DOCDIR]
#  --pdfdir=DIR            pdf documentation [DOCDIR]
#  --psdir=DIR             ps documentation [DOCDIR]
#
#Program names:
#  --program-prefix=PREFIX            prepend PREFIX to installed program names
#  --program-suffix=SUFFIX            append SUFFIX to installed program names
#  --program-transform-name=PROGRAM   run sed PROGRAM on installed program names
#
#Optional Features:
#  --disable-option-checking  ignore unrecognized --enable/--with options
#  --disable-FEATURE       do not include FEATURE (same as --enable-FEATURE=no)
#  --enable-FEATURE[=ARG]  include FEATURE [ARG=yes]
#  --enable-libsearch      enable search for libraries, default: yes
#  --enable-static-patch   enable allow statically linking plumed, default: yes
#  --enable-doc            enable documentation, default: yes
#  --enable-pdfdoc         enable pdf version of the manual, default: no
#  --enable-debug          enable debugging, default: no
#  --enable-gcov           enable gcov to estimate code coverage, default: no
#  --enable-cxx            11 or 14. To link libraries with headers that need
#                          this C++ level. Use --enable-cxx=none to remove
#                          -std=c++ flag
#  --enable-debug-glibcxx  enable enable boundary check, default: no
#  --enable-shared         enable shared libs, default: yes
#  --enable-rpath          enable store rpath, default: no
#  --enable-absolute-install-name
#                          enable store absolute relative (OSX only - disable
#                          to have a behavior similar to Linux). Only disable
#                          for testing!, default: yes
#  --enable-loader-path    enable use @loader_path to find
#                          libplumedKernel.dylib (OSX only), default: yes
#  --enable-ld-r           enable group object files, default: yes
#  --enable-ar-cr          enable use ar to build libplumedWrapper.a, default:
#                          yes
#  --enable-static-archive enable try to build libplumed.a for static linking,
#                          default: yes
#  --enable-python         enable search for python, default: yes
#  --enable-af_ocl         enable search for arrayfire_ocl, default: no
#  --enable-af_cuda        enable search for arrayfire_cuda, default: no
#  --enable-af_cpu         enable search for arrayfire_cpu, default: no
#  --enable-libtorch       enable search for libtorch, default: no


#CMAKE will do this for us :)
#  --enable-dependency-tracking
#                          enable dependency tracking, default: yes

################################################################################

#check module dependencies
foreach(dir ${module_dirs})
    if(${module_${dir}})
        set(notFound "")
        #message("${dir} ${moduleNeeds_${dir}}")
        foreach(otherModule ${moduleNeeds_${dir}})
            if (NOT module_${otherModule})
                list(FIND mayBeExternal ${otherModule} isExternal)
                if(${isExternal})
                    #this may be removed?
                    string(TOUPPER ${otherModule} capsName)
                    if(${capsName}_FOUND)
                        continue()
                    endif(${capsName}_FOUND)
                endif(${isExternal})
                list(APPEND notFound ${otherModule})
                set (modulesDependenciesProblems ON)
            endif(NOT module_${otherModule})
        endforeach(otherModule moduleNeeds_${dir})
        if (notFound)
            message(SEND_ERROR "Module \"${dir}\" needs the following modules active: ${notFound}")
        endif(notFound)
        unset(notFound)
    endif(${module_${dir}})
endforeach(dir ${module_dirs})
if (${modulesDependenciesProblems})
    message(FATAL_ERROR "Problems in module dependencies")
endif(${modulesDependenciesProblems})
################################################################################
#########################Setting up the main libraries##########################
################################################################################

set(KernelTargets "")
#In the original makefile:
#->#OBJ_DYNAMIC_WRAPPER := ../wrapper/Plumed.o (here is target PlumedDynamicWrapper)
#->#OBJ_WRAPPER := ../wrapper/PlumedStatic.o (here is target PlumedStaticWrapper)

foreach(activeModule ${modulesForKernel})
    list(APPEND KernelTargets $<TARGET_OBJECTS:${activeModule}>)
endforeach(activeModule ${modulesForKernel})

###############################libplumedKernel.so###############################
#sharedplumedKernel collects all the module objects in a single shared object
#->#PLUMED_KERNEL := $(PLUMED_LIBDIR)lib$(PLUMED_NAME)Kernel.$(SOEXT)
#the old makefile recipe is:
#->#$(PLUMED_KERNEL):$(OBJ_KERNEL)
#->#	$(LDSHARED) -o $@ $^ $(SONAME_OPTION),"$(SONAME_INSTALL_PATH)$(notdir $@)" $(DYNAMIC_LIBS)
#Cmake will automatically make the so relocatable on install
add_library(sharedplumedKernel SHARED)
set_target_properties(sharedplumedKernel PROPERTIES
    LIBRARY_OUTPUT_NAME ${program_name}Kernel)
target_link_libraries(sharedplumedKernel PUBLIC
     ${modulesForKernel} # OBJ_KERNEL without config
     Config #within OBJ_KERNEL
    )
#needed for tests:
add_library(Plumed2::sharedplumedKernel ALIAS sharedplumedKernel)

###############################libplumedWrapper.a###############################
#archiveplumedWrapper contains the dynamic wrapper
#->#PLUMED_WRAPPER_LIB := $(PLUMED_LIBDIR)lib$(PLUMED_NAME)Wrapper.a
#the old makefile recipe is:
#->#$(PLUMED_WRAPPER_LIB): $(OBJ_DYNAMIC_WRAPPER)
#->#	$(AR_CR) $@ $^
add_library(archiveplumedWrapper STATIC)
set_target_properties(archiveplumedWrapper PROPERTIES 
    ARCHIVE_OUTPUT_NAME ${program_name}Wrapper)
target_link_libraries(archiveplumedWrapper PUBLIC
    PlumedDynamicWrapper
    )

###################################libplumed.a##################################
#archiveplumedKernelWithMain contains the kernel objects ready to be statically linked
#it also contains the main function
#->#$(LD_RO) install/kernel.o $(OBJ_KERNEL)
#->#    $(LD_RO) install/kernel.o $(OBJ_KERNEL)
#the old makefile recipe is:
#->#install/lib$(program_name).a: install/kernel.o $(OBJ_WRAPPER)
#->#    $(AR_CR) $@ $^
add_library(archiveplumedKernelWithMain STATIC)
set_target_properties(archiveplumedKernelWithMain PROPERTIES
        ARCHIVE_OUTPUT_NAME ${program_name})
target_link_libraries(archiveplumedKernelWithMain PUBLIC
        ${modulesForKernel} # OBJ_KERNEL without config
        Config #within OBJ_KERNEL
        PlumedStaticWrapper
    )

##################################libplumed.so##################################
#sharedplumedMain contains the main function and need to be linked to the shared
#kernel
#->#PLUMED_SHARED_OBJ := $(PLUMED_LIBDIR)lib$(PLUMED_NAME).$(SOEXT)
#the old makefile recipe is:
#->#$(PLUMED_SHARED_OBJ): $(OBJ_WRAPPER) $(PLUMED_KERNEL)
#->#	$(LDSHARED) -o $@ $^ $(SONAME_OPTION),"$(SONAME_INSTALL_PATH)$(notdir $@)"
add_library(sharedplumedMain SHARED)
set_target_properties(sharedplumedMain PROPERTIES
        LIBRARY_OUTPUT_NAME ${program_name})
target_link_libraries(sharedplumedMain PUBLIC
    PlumedStaticWrapper #object containging the main
    sharedplumedKernel #the shared library with the kernel and the modules
)
#needed for tests:
add_library(Plumed2::sharedplumedMain ALIAS sharedplumedMain)

##########################end of libraries declaration##########################
#installing the libraries

install (TARGETS
    sharedplumedKernel archiveplumedKernelWithMain archiveplumedWrapper sharedplumedMain
    EXPORT Plumed2LIBS
    ARCHIVE DESTINATION ${DIRS_LIB_INSTALL}
    LIBRARY DESTINATION ${DIRS_LIB_INSTALL}
)

#these subdirectories contains informations for installing plumed
add_subdirectory(lib)
#the codes for the executable are within the main subdir
add_subdirectory(main)
#this subdirectory contains the informations for accessing the compiled plumed
#without installing it,
#AND installs the user macros for creating a plumed2 plugin (using LOAD FILE=file.cpp)
add_subdirectory(cmake)
################################################################################
#One can also use -DCMAKE_EXPORT_COMPILE_COMMANDS=ON in that case, such that we 
#have all the commands in the file compile_commands.json without actually 
#building the sources.
#-DCMAKE_EXPORT_COMPILE_COMMANDS=ON
#thanks https://stackoverflow.com/questions/2670121/using-cmake-with-gnu-make-how-can-i-see-the-exact-commands#comment112625585_2673355
################################################################################

file(CONFIGURE OUTPUT ${CMAKE_BINARY_DIR}/sourceme.sh
     CONTENT [[#by sourcing this file you will be able to run plumed from the build directory
export PATH="@CMAKE_BINARY_DIR@/src/main/:$PATH"
export LIBRARY_PATH="@CMAKE_BINARY_DIR@/src/:$LIBRARY_PATH"
export LD_LIBRARY_PATH="@CMAKE_BINARY_DIR@/src/:$LD_LIBRARY_PATH"
export PLUMED_KERNEL="@CMAKE_BINARY_DIR@/src/libplumedKernel.so"
export PLUMED_VIMPATH="@CMAKE_SOURCE_DIR@/vim"
export PYTHONPATH="@CMAKE_SOURCE_DIR@/python:$PYTHONPATH"
export PYTHONPATH="@CMAKE_SOURCE_DIR@/python:$PYTHONPATH"
export CMAKE_PREFIX_PATH="@CMAKE_BINARY_DIR@/src/cmake:$CMAKE_PREFIX_PATH"
export PLUMED_ROOT="@CMAKE_BINARY_DIR@"
]]
     #ESCAPE_QUOTES
     @ONLY
     )

set (TODO [[
     #TODO
-DSTDC_HEADERS=1
-DHAVE_SYS_TYPES_H=1
-DHAVE_SYS_STAT_H=1
-DHAVE_STDLIB_H=1
-DHAVE_STRING_H=1
-DHAVE_MEMORY_H=1
-DHAVE_STRINGS_H=1
-DHAVE_INTTYPES_H=1
-DHAVE_STDINT_H=1
-DHAVE_UNISTD_H=1

]])

if(SHOW_LINKING)
    set(
        CMAKE_ECHO_LINK_EXECUTABLE 
        "<CMAKE_COMMAND> -E echo \"Flags: <FLAGS>\""
        "<CMAKE_COMMAND> -E echo \"LinkFlags: <LINK_FLAGS>\""
        "echo \"LinkLibraries: <LINK_LIBRARIES>\" > <TARGET>"
    )
    set(CMAKE_ECHO_STANDARD_LIBRARIES ${CMAKE_CXX_STANDARD_LIBRARIES})
    set(CMAKE_ECHO_FLAGS ${CMAKE_CXX_FLAGS})
    set(CMAKE_ECHO_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS})
    set(CMAKE_ECHO_IMPLICIT_LINK_DIRECTORIES ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES})

    set(CMAKE_ECHO_COMPILE_OBJECT
    "<CMAKE_COMMAND> -E echo \"<DEFINES>\" >> <TARGET>")

    file(WRITE null.echo "")

    #add_executable(LinkLine "")
    foreach(linkelper Shared Static Runtime)
        add_executable(Link${linkelper}
            null.echo
        )
        set_target_properties(
            Link${linkelper}  
                PROPERTIES
                    LINKER_LANGUAGE ECHO
                    SUFFIX          ".txt"
        )
        add_custom_command(
            TARGET Link${linkelper}
            POST_BUILD
            # COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:Link${linkelper}> PackageCfg/$<TARGET_FILE_NAME:Link${linkelper}>
            COMMAND ${CMAKE_COMMAND} -E echo $<TARGET_FILE:Link${linkelper}> PackageCfg/$<TARGET_FILE_NAME:Link${linkelper}>
        )
    endforeach(linkelper )

    target_link_libraries(LinkShared sharedplumedKernel PlumedStaticWrapper)
    target_link_libraries(LinkStatic archiveplumedKernelWithMain PlumedStaticWrapper)
    target_link_libraries(LinkRuntime PlumedDynamicWrapper)
endif(SHOW_LINKING)
# include(CMakePrintHelpers)
# cmake_print_properties(TARGETS cltools plumed_bin
#     PROPERTIES COMPILE_DEFINITIONS)

# cmake_print_properties(TARGETS blas config PROPERTIES STATIC_LIBRARY_OPTIONS 
# INTERFACE_LINK_OPTIONS        LINK_OPTIONS
# INTERFACE_LINK_FLAGS          LINK_FLAGS 
# INTERFACE_COMPILE_DEFINITIONS COMPILE_DEFINITIONS
# INTERFACE_COMPILE_OPTIONS     COMPILE_OPTIONS 
# INTERFACE_LINK_LIBRARIES      LINK_LIBRARIES
# INTERFACE_INCLUDE_DIRECTORIES INCLUDE_DIRECTORIES )    

message("PLUMED_LIBS: ${PLUMED_LIBS}")
message("PLUMED_DYNAMIC_LIBS: ${PLUMED_DYNAMIC_LIBS}")
message("PLUMED_LD_FLAGS: ${PLUMED_LD_FLAGS}")
message("CMAKE_SHARED_LINKER_FLAGS: ${CMAKE_SHARED_LINKER_FLAGS}")
#LIST(REMOVE_DUPLICATES PLUMED_LD_FLAGS)
#message("PLUMED_LD_FLAGS: ${PLUMED_LD_FLAGS}")
