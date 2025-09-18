#!/bin/bash

for f in ./rt-*/plumed.dat; do
  echo "$f"
  {
    echo LOAD FILE=../../../../plumedOpenACC.so
    cat "$f"
  } >tmp
  mv tmp "$f"
done
