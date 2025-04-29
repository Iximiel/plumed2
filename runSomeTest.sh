#!/bin/bash
# checked with shellcheck
# formatted with shfmtv3.36.0

action="run"

dirs=(
    regtest/pamm/rt-pamm-aperiodic
    regtest/gridtools/rt-spherical-integral-2
    regtest/volumes/rt-q3-insphere
)

if [[ $action = "run" ]]; then
    for dir in "${dirs[@]}"; do
        echo -n "> Running $dir: "
        if (
            cd "$dir" || exit
            make valgrind
            grep -vzq FAILURE report.txt
        ) >>/dev/null; then
            echo -e "\e[32mSUCCESS\e[0m"
        else
            echo -e "\e[31mFAILURE\e[0m"
        fi
        cat "$dir/report.txt"
    done
fi
