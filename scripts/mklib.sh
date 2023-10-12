#! /usr/bin/env bash

if [ "$1" = --description ] ; then
  echo "compile a .cpp file into a shared library"
  exit 0
fi

if [ "$1" = --options ] ; then
  echo "--description --options"
  exit 0
fi

if [ $# != 1 ] || [[ "$1" != *.cpp ]]; then
  echo "ERROR"
  echo "type 'plumed mklib file.cpp'"
  exit 1
fi

file="$1"

if [ ! -f "$file" ]; then
  echo "ERROR: I cannot find file $file"
  exit 1
fi
#adding a simple tmpfile, to preprocess "in place" the input file,
#this assumes the user has write permission in the current directory
#which should be true since we are going to compile and output something here
tmpfile=$(mktemp ${file%.cpp}.XXXXXX.cpp)
cp "${file}" "${tmpfile}"

if grep -q '^#include "\(bias\|colvar\|function\|sasa\|vatom\)\/ActionRegister.h"' "${tmpfile}"; then
   >&2 echo 'WARNING: using a legacy ActionRegister.h include path, please use <<#include "core/ActionRegister.h">>'
   sed -i.bak 's%^#include ".*/ActionRegister.h"%#include "core/ActionRegister.h"%g' "${tmpfile}"
fi

if grep -q '^#include "\(cltools\)\/CLToolRegister.h"' "${tmpfile}"; then
   >&2  echo 'WARNING: using a legacy  CLToolRegister.h include path, please use <<#include "core/CLToolRegister.h">>'
   sed -i.bak 's%^#include ".*/CLToolRegister.h"%#include "core/CLToolRegister.h"%g' "${tmpfile}"
fi

srcDir=$(mktemp -d src.XXXXXXX)
buildDir=$(mktemp -d build.XXXXXXX)
# check if tmp dir was created
if [[ ! "$buildDir" || ! -d "$buildDir" || ! "$srcDir" || ! -d "$srcDir" ]]; then
  echo "Could not create temp dir"
  exit 1
fi

cat << EOF > "${srcDir}/CMakeLists.txt"
cmake_minimum_required(VERSION 3.20)
project(${file%%.cpp})
find_package(Plumed2 REQUIRED)
add_plumed_plugin(${file%%.cpp} SOURCES ../$file)
install (TARGETS ${file%%.cpp}
    LIBRARY DESTINATION ${PWD}
)
EOF

function cleanup {
  rm -rf "$srcDir"
  rm -rf "$buildDir"
}
trap cleanup EXIT

#If CMAKE_PREFIX_PATH has been set up by sourceme this will not be used
cmakePATH=$(dirname "$0")
cmakePATH=$(realpath "$cmakePATH")
#This is expected to be installed in lib/plumed/scripts
#the cmake files are in lib/cmake/plumed
cmakePATH=${cmakePATH}/../../cmake/plumed

export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:${cmakePATH}

cmake -B "$buildDir" -S "$srcDir" -DCMAKE_BUILD_TYPE:STRING=Release
cmake --build "$buildDir"
cmake --install "$buildDir"
