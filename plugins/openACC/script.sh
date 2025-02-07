#!/bin/bash

echo d: DISTANCE ...
for i in `seq 1 54` ; do # shellcheck source-path=
   echo ATOMS$i=$((2*$i-1)),$((2*$i))
done
echo ...
echo PRINT ARG=d FILE=colvar