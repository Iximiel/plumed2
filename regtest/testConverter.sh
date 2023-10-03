#!/bin/bash
if [[ -f config ]]; then
  source config

  exports=$(grep "^export" config 2>/dev/null)
  #type->TYPE
  ##simplemd
  ##driver->Done
  ##sum_hills
  ##make->Done
  ##plumed
  ##python

  #these need to be implemented
  #PLUMED_ALLOW_SKIP_ON_TRAVIS
  #plumed_language
  #PLUMED_NUM_THREADS
  #plumed_regtest_before->Done
  #plumed_regtest_after->Done
  #plumed_custom_skip->Done
  #export variables->Done
  for i in plumed_regtest_before plumed_regtest_after plumed_custom_skip; do
    if declare -f $i >/dev/null; then
      #displaying the function
      {
        echo "#! /usr/bin/env bash"
        declare -f $i | head -n-1 | tail -n+3 | sed -e 's/;$//' -e 's/^    //' -e 's/return/exit/'
      } | tee $i
      chmod +x "$i"
    fi
  done
  name=${PWD##*/}
  echo ""
  {
    echo "PLUMED_TEST(${name} TYPE ${type}"
    if [[ $arg ]]; then
      echo "ARGS \"${arg}\""
    fi
    if [[ $mpiprocs ]]; then
      echo "MPIPROCS ${mpiprocs}"
    fi
    if [[ $plumed_modules ]]; then
      echo "MODULES $plumed_modules"
    fi
    if [[ $plumed_needs ]]; then
      echo "NEEDS $plumed_needs"
    fi
    if [[ $exports ]]; then
      echo "EXPORTVARIABLES"
      readarray -t exportsArr <<<"$exports"
      for exp in "${exportsArr[@]}"; do
        exp=${exp#export }
        echo "${exp%=*} \"${exp#*=}\""
      done
    fi
    if [[ $extra_files ]]; then
      echo "EXTRAFILES $extra_files"
    fi
    echo ")"
    echo "#the following variable is the original config file"
    echo "set(originalConfig [["
    cat config
    echo "]] )"
  } | tee CMakeLists.txt
fi
