#this is not a module, this is set up to make easier the dependency interface
set(module_name main)
set(module_${module_name} ON CACHE INTERNAL "always active core module ${module_name}")

#In the original makefile:
#->#OBJ_DYNAMIC_WRAPPER := ../wrapper/Plumed.o (here is target PlumedDynamicWrapper)
#->#OBJ_WRAPPER := ../wrapper/PlumedStatic.o (here is target PlumedStaticWrapper)

#dynamically linked plumed
#original make recipe:
#->#$(PLUMED_MAIN_STATIC): $(OBJ_MAIN) $(OBJ_KERNEL) $(OBJ_WRAPPER)
#->#	$(LD) $(LDFLAGS) $^ -o $@ $(LIBS) $(DYNAMIC_LIBS)
add_executable(plumed_bin main.cpp)#OBJ_MAIN
target_link_libraries(plumed_bin PUBLIC
    sharedplumedKernel #PLUMED_KERNEL
    PlumedStaticWrapper #OBJ_WRAPPER
    )
set_target_properties(plumed_bin
    PROPERTIES RUNTIME_OUTPUT_NAME ${program_name}
    )

#statically linked plumed
#original make recipe (links the objects, same as the target archiveplumedKernelWithMain):
#->#$(PLUMED_MAIN_STATIC): $(OBJ_MAIN) $(OBJ_KERNEL) $(OBJ_WRAPPER)
#->#	$(LD) $(LDFLAGS) $^ -o $@ $(LIBS) $(DYNAMIC_LIBS)
add_executable(plumed_bin_static main.cpp) #$(OBJ_MAIN)
target_link_libraries(plumed_bin_static PUBLIC
    archiveplumedKernelWithMain
)
set_target_properties(plumed_bin_static
    PROPERTIES RUNTIME_OUTPUT_NAME ${program_name}-static
    )
#runtime linked plumed
#->#$(PLUMED_MAIN_RUNTIME): $(OBJ_MAIN) $(OBJ_DYNAMIC_WRAPPER)
#->#	$(LD) $(LDFLAGS) $^ -o $@ $(LIBS)
add_executable(plumed_bin_runtime main.cpp)
target_link_libraries(plumed_bin_runtime
    PUBLIC
    archiveplumedWrapper #OBJ_DYNAMIC_WRAPPER
    )
set_target_properties(plumed_bin_runtime
    PROPERTIES RUNTIME_OUTPUT_NAME ${program_name}-runtime
    )

install (TARGETS plumed_bin
    EXPORT Plumed2EXEC
    RUNTIME
    DESTINATION ${DIRS_BIN_INSTALL})

install (TARGETS plumed_bin_runtime
    EXPORT Plumed2EXEC
    RUNTIME
    DESTINATION ${DIRS_EXTRA_INSTALL})

#the executable alias are needed for testing in build-configuration
add_executable(Plumed2::plumed_bin ALIAS plumed_bin)
add_executable(Plumed2::plumed_bin_static ALIAS plumed_bin_static)
add_executable(Plumed2::plumed_bin_runtime ALIAS plumed_bin_runtime)

#consider adding install manifest:
#install(CODE "string(REPLACE \";\" \"\\n\" MY_CMAKE_INSTALL_MANIFEST_CONTENT \"\$\{CMAKE_INSTALL_MANIFEST_FILES\}\")\n\
#              file(WRITE my_install_manifest.txt \"\$\{MY_CMAKE_INSTALL_MANIFEST_CONTENT\}\")")
#thanks https://stackoverflow.com/a/66804151


#installation
#here there is an exampe ot multiple installation positions
#https://cmake.org/cmake/help/v3.2/command/install.html#installing-targets
