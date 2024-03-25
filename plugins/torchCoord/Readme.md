## Set up
You can try to install libtorch in different ways:
You can try to install the latest version from [here](https://pytorch.org/get-started/locally/), or an older version, [passing by python](https://pytorch.org/get-started/previous-versions/), but I do not recommend that because there can be incompabilities with the CXXABI.

At the moment the latest version of libtorch 2.2.2 is linked to GLIBC_2.29, so if your compiler links to a different version you should compile your own version of libtorch:

If you have modules remeber to activate the one that you are using with plumed (same compiler, same glibc, same cxxabi, etc)
```bash
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch
git checkout eventual version
git submodule update --init --recursive
pip install -U pip
pip install -U pyyaml typing_extensions
python3 tools/build_libtorch.py
```
if the last command fails, you may try tor run `python3 tools/build_libtorch.py --rerun-cmake` after installing eventual python modules needed or new modules, to refresh the configuration.

I recomment to use 2.something because of the compile time when compiling the plumed accelerated files

Then you should change `/path/to/torch/` in the following module file to set up your environment
```tcl
#%Module1.0##############################################

set basedir "/path/to/torch/"

set include "include"
set lib "lib"

prepend-path  CPATH              ${basedir}/${include}
#the <torch/something> includes are here
prepend-path  CPATH              ${basedir}/${include}/torch/csrc/api/include
prepend-path  INCLUDE            ${basedir}/${include}
prepend-path  INCLUDE            ${basedir}/${include}/torch/csrc/api/include
prepend-path  LIBRARY_PATH       ${basedir}/${lib}
prepend-path  LD_LIBRARY_PATH    ${basedir}/${lib}
prepend-path  CMAKE_MODULE_PATH  ${basedir}/share/cmake

```
or you can source:
```bash
torchbasedir="/path/to/torch/"

export CPATH=${torchbasedir}/include:${CPATH}
#the <torch/something> includes are here
export CPATH=${torchbasedir}/include/torch/csrc/api/include:${CPATH}
export INCLUDE=${torchbasedir}/include:${INCLUDE}
export INCLUDE=${torchbasedir}/include/torch/csrc/api/include:${INCLUDE}
export LIBRARY_PATH=${torchbasedir}/lib:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${torchbasedir}/lib:${LD_LIBRARY_PATH}
export CMAKE_MODULE_PATH=${torchbasedir}/share/cmake:${CMAKE_MODULE_PATH}

```
the previous files are an adaptation from the torch cmake files.
If you are using an installed by python version the torch files are in the subdirectory "/lib64/python3.9/site-packages/torch" within the pyhon environment.
