################################################################################
##########################Creating PlumedConfig.cmake###########################
################################################################################
include(CMakePackageConfigHelpers)
set (EXTERNAL_PACKAGES)
#when package are found in src/* the ${dep}_FOUND should be made PARENT_SCOPE

#export eventual helper variables options
foreach(opt
    #blas and lapack
    MKLROOT BLA_STATIC BLA_VENDOR BLA_F95 BLA_PREFER_PKGCONFIG
    #GSL options
    GSL_ROOT_DIR
    )
    if(${opt})
        list(APPEND EXTERNAL_PACKAGES "set(${opt} ${${opt}})")
    endif()
endforeach()

#these packages are called with find_package(name) without options
foreach(dep BLAS LAPACK OpenMP MPI GSL ZLIB Backtrace)
    if(${dep}_FOUND)
        list(APPEND EXTERNAL_PACKAGES "find_dependency(${dep})")
    endif()
endforeach()

if(PkgConfig_FOUND)
    list(APPEND EXTERNAL_PACKAGES "\nfind_dependency(PkgConfig)")
    #foreach(dep FFTW)
    if(FFTW_FOUND)
        list(APPEND EXTERNAL_PACKAGES 
        "pkg_search_module(FFTW fftw3 IMPORTED_TARGET)")
    endif()
    #endforeach()
endif()

#this adds the dependecies to Boost if they are required
set(BoostCompontents)
if(Boost_GRAPH_FOUND)
    list(APPEND BoostCompontents "graph")
endif()
if(Boost_SERIALIZATION_FOUND)
    list(APPEND BoostCompontents "serialization")
endif()

if(BoostCompontents)
    list(JOIN BoostCompontents " " BoostCompontents)
    list(APPEND EXTERNAL_PACKAGES 
    "find_dependency(Boost COMPONENTS ${BoostCompontents})")
endif()

list(JOIN EXTERNAL_PACKAGES "\n" EXTERNAL_PACKAGES)
#this is the installed version
configure_package_config_file(
    PlumedConfig.cmake.in
    "Plumed2Config.cmake"
    INSTALL_DESTINATION
      ${DIRS_CMAKE_INSTALL}
    PATH_VARS 
      DIRS_INCLUDE_INSTALL 
      DIRS_LIB_INSTALL
      DIRS_BIN_INSTALL
      DIRS_EXTRA_INSTALL
)
#This is the build version
configure_package_config_file(
    PlumedConfig.cmake.in
    "${CMAKE_BINARY_DIR}/lib/cmake/plumed/Plumed2Config.cmake"
    INSTALL_DESTINATION 
     ${CMAKE_BINARY_DIR}/lib/cmake/plumed/
    PATH_VARS 
      DIRS_INCLUDE_INSTALL 
      DIRS_LIB_INSTALL
      DIRS_BIN_INSTALL
      DIRS_EXTRA_INSTALL
    INSTALL_PREFIX
      ${CMAKE_BINARY_DIR}
)

################################################################################
######################Creating Plumed2ConfigVersion.cmake#######################
################################################################################
write_basic_package_version_file("${CMAKE_BINARY_DIR}/lib/cmake/plumed/Plumed2ConfigVersion.cmake"
  VERSION ${PLUMED_VERSION_MAJOR}.${PLUMED_VERSION_MINOR}.${PLUMED_VERSION_PATCH}
  #COMPATIBILITY <AnyNewerVersion|SameMajorVersion|SameMinorVersion|ExactVersion>
  COMPATIBILITY SameMinorVersion
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Plumed2Config.cmake
              ${CMAKE_BINARY_DIR}/lib/cmake/plumed/Plumed2ConfigVersion.cmake
              PlumedUserMacros.cmake #This installs the macros for the user
  DESTINATION ${DIRS_CMAKE_INSTALL}
)

#this is needed to have PlumedUserMacros.cmake avaiable in the build configuration
file(COPY PlumedUserMacros.cmake
  DESTINATION ${CMAKE_BINARY_DIR}/lib/cmake/plumed/
)

foreach(exportetTargets Plumed2MODULES Plumed2WRAPPERS Plumed2LIBS Plumed2EXEC)
  install(EXPORT ${exportetTargets}
    NAMESPACE Plumed2::
    DESTINATION ${DIRS_CMAKE_INSTALL}
  )
  export(EXPORT ${exportetTargets}
    NAMESPACE Plumed2::
    FILE "${CMAKE_BINARY_DIR}/lib/cmake/plumed/${exportetTargets}.cmake"
  )
endforeach(exportetTargets )

