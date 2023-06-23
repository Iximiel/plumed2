#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "maze")
#Note that the macros here require this directory added as a subdir of plumed/src
DECLAREPLUMEDMODULE(${module_name}
OFF
SOURCES
Core.cpp
Loss.cpp
Member.cpp
Memetic.cpp
Optimizer_Bias.cpp
Optimizer.cpp
Random_Acceleration_MD.cpp
Random_MT.cpp
Random_Walk.cpp
Simulated_Annealing.cpp
Steered_MD.cpp
Tools.cpp
NEEDS
	core tools colvar bias
DEPENDS
	colvar #tools core
	bias #tools core
)
