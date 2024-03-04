#!/bin/bash
runType=original
for dir in rt20-switch-*; do
#if [[ ${dir} != *"rational"* ]]; then continue; fi
  (
    cd "${dir}"|| exit
   {
    for i in $(seq 200); do
      make 1>&2 
      grep "forward loop" ./tmp/out | awk '{print $6, $7, $8}'
    done
   } >> "${dir}_${runType}"
  ) 
done