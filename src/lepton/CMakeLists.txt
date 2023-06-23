#the variable module_name is set up as a sugar to reduce "copy-paste" errors
set (module_name "lepton")
#Note that the macros here require this directory added as a subdir of plumed/src
DECLAREPLUMEDMODULE(${module_name}
"always"
SOURCES
CompiledExpression.cpp
ExpressionProgram.cpp
ExpressionTreeNode.cpp
Operation.cpp
ParsedExpression.cpp
Parser.cpp
EXTRA_HEADERS
CustomFunction.h
Exception.h
Lepton.h
MSVC_erfc.h
windowsIncludes.h
NEEDS
	asmjit
DEPENDS
	asmjit
)
