#!/bin/bash
runType=original
for file in *_new; do
n=${file#rt20-switch-}
{
# echo ${n}_n ${n}_tot ${n}_average
# cat $file
echo ${n}
cut -d " " -f 2 ${file}
} > ${file}_tmp
done

paste *new_tmp  > times_new
for file in *_original; do
n=${file#rt20-switch-}
{
# echo ${n}_n ${n}_tot ${n}_average
# cat $file
echo ${n}
cut -d " " -f 2 ${file}
} > ${file}_tmp
done
paste *original_tmp > times_original
rm *new_tmp *original_tmp