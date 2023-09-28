#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "pytorch")
#Note that the macros here require this directory added as a subdir of plumed/src

DECLAREPLUMEDMODULE(${module_name}
OFF
SOURCES
PytorchModel.cpp
NEEDS
	core function
DEPENDS
	function #reference tools core lepton
)

if (module_pytorch)
	#if I use 'REQUIRED' the cmake process is broken if Torch is not found
	find_package(Torch)
	if (Torch_FOUND)
		#The commented code should be the correct one
		target_link_libraries(pytorch PRIVATE ${TORCH_LIBRARIES})
		target_compile_definitions(config INTERFACE "__PLUMED_HAS_LIBTORCH=1")
		if (${CMAKE_CXX_STANDARD} LESS 14)
			set_property(TARGET pytorch PROPERTY CXX_STANDARD 14)
		endif()
		# set(Torch_FOUND Torch_FOUND PARENT_SCOPE)
		# CONFIGSETTINGS(config Torch_FOUND
        # 	LINK_LIBRARIES ${TORCH_LIBRARIES}
        # 	COMPILE_DEFINITIONS __PLUMED_HAS_LIBTORCH
        # )
	else()
		if(option_relaxed_dependencies)
			#with relaxed
			message(WARNING "Torch not found, module_pytorch will be deactivated")
			set(module_pytorch OFF CACHE BOOL "activate module ${dir}" FORCE)
		else()
		#this does not break the cmake configuration, but stops the build
			message(SEND_ERROR "Torch not found, libtorch cannot be linked.\n"
				"Please follow the instruction of the previous warning or "
				"deactivate module_pytorch"
			)
		endif()
		
	endif()
endif()