#this is not a module, this is set up to make easier the dependency interface
set(module_wrapper ON CACHE INTERNAL "always active core module wrapper")

add_library(PlumedStaticWrapper OBJECT PlumedStatic.cpp)
target_link_libraries(PlumedStaticWrapper PUBLIC config )
set_target_properties(PlumedStaticWrapper PROPERTIES PUBLIC_HEADER "Plumed.h")

add_library(PlumedDynamicWrapper OBJECT Plumed.c)
target_link_libraries(PlumedDynamicWrapper PUBLIC config )
set_target_properties(PlumedDynamicWrapper PROPERTIES PUBLIC_HEADER "Plumed.h")

install (TARGETS PlumedDynamicWrapper PlumedStaticWrapper
    EXPORT Plumed2WRAPPERS
    ARCHIVE DESTINATION ${DIRS_LIB_INSTALL}
    LIBRARY DESTINATION ${DIRS_LIB_INSTALL}
    PUBLIC_HEADER DESTINATION ${DIRS_INCLUDE_INSTALL}/wrapper
)

#add here the lib so and the lib static for Plumed.so and PlumedKernel.so
#find a way to add all the activated libraries here or add them to tha main CMakeList.txt?
#Then cmake should short all the mess for you, hopefully
