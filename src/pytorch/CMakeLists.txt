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

if (module_${module_name})
	#if I use 'REQUIRED' the cmake process is broken if Torch is not found
	find_package(Torch)
	if (Torch_FOUND)
		target_link_libraries(${module_name} PRIVATE ${TORCH_LIBRARIES})
		#This broadcast the option to everithing
		target_compile_definitions(config INTERFACE "__PLUMED_HAS_LIBTORCH=1")
		if (${CMAKE_CXX_STANDARD} LESS 14)
			set_property(TARGET ${module_name} PROPERTY CXX_STANDARD 14)
		endif()
		
	else()
		if(option_relaxed_dependencies)
			#with relaxed
			message(WARNING "Torch not found, module_${module_name} will be deactivated")
			set(module_${module_name} OFF CACHE BOOL "activate module ${dir}" FORCE)
		else()
		#this does not break the cmake configuration, but stops the build
			message(SEND_ERROR "Torch not found, libtorch cannot be linked.\n"
				"Please follow the instruction of the previous warning or "
				"deactivate module_${module_name}"
			)
		endif()
		
	endif()
endif()