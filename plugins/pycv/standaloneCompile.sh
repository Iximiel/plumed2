#! /usr/bin/env bash

source compileConfiguration.sh


export PLUMED_MKLIB_CFLAGS="${python_cf_embedded} ${pybind11_cflags}"
export PLUMED_MKLIB_LDFLAGS="${PYCV_EXTRA_LDFLAGS} ${python_ld_embedded} ${conda_fixup}"

echo PLUMED_MKLIB_CFLAGS=$PLUMED_MKLIB_CFLAGS
echo PLUMED_MKLIB_LDFLAGS=$PLUMED_MKLIB_LDFLAGS

${plumed_program_name} mklib PythonCVInterface.cpp ActionWithPython.cpp PythonFunction.cpp PlumedPythonEmbeddedModule.cpp
