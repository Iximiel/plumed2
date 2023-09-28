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
		target_link_libraries(pytorch PRIVATE ${TORCH_LIBRARIES})
	else()
		#this does not break the cmake configuration, but stops the build
		message(SEND_ERROR "Torch not found, libtorch cannot be linked.\n"
				"Please follow the instruction of the previous warning or "
				"deactivate the libtorch module"
		)
		#set(module_pytorch OFF CACHE BOOL "activate module ${dir}" FORCE)
	endif()
endif()