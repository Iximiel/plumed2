# Note that in conda libpython3xx is not found in the path returned by ldflags. IMHO it is a bug.
# The workaround is to -L appropriately. Will be fixed here.

conda_fixup=${CONDA_PREFIX+-L$CONDA_PREFIX/lib}
if [ ! -z "$conda_fixup" ]; then 
  echo "CONDA_PREFIX is set. Assuming conda and enabling a workaround for missing -L in python3-config --ldflags --embed"
fi

if ! python3-config --embed >/dev/null 2>/dev/null; then
  #TODO: verify that this does not give problems with conda
  echo "PyCV needs python to be built to be embedable"
  echo "(compiling python with --enable-shared should be enough)"
  exit 1
fi

export  PLUMED_MKLIB_CFLAGS="$(python3-config --cflags --embed) $(python -m pybind11 --includes)"

export PLUMED_MKLIB_LDFLAGS="$(python3-config --ldflags --embed) $conda_fixup"

echo PLUMED_MKLIB_CFLAGS=$PLUMED_MKLIB_CFLAGS
echo PLUMED_MKLIB_LDFLAGS=$PLUMED_MKLIB_LDFLAGS

plumed mklib PythonCVInterface.cpp ActionWithPython.cpp PlumedPythonEmbeddedModule.cpp
