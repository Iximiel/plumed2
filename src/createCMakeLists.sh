#!/bin/bash
#small utilty that recurses the src folder and creates a standard CMakeListst.txt
#for modules where is not present
#thinked to be launched in repodir/src

createCMakeLists() {
	dir=$1
	if test -f "$dir/module.type"; then
		if test -f $dir/CMakeLists.txt; then
			echo "$dir has the CMakeLists.txt"
			# if grep -q "automatically generated CMakeLists.txt, if it does not work" $dir/CMakeLists.txt; then
			#this update non modified CMakeLists.txt, decomment if needed
			# rm -v $dir/CMakeLists.txt
			# fi
		else
			echo "$dir"
		fi

		if test ! -f $dir/CMakeLists.txt; then
			(
				cd $dir || exit
				{
					echo "message(WARNING \"${dir} has an automatically generated CMakeLists.txt by createCMakeLists.sh, if it does not work modify it and remove this warning\")"
					echo "#the variable module_name is set up as a sugar to reduce \"copy-paste\" errors"
					echo "set (module_name \"${dir}\")"
					echo "#Note that the macro DECLAREPLUMEDMODULE require this directory added as a subdir of plumed/src or plumedCMakeMacros included"

					if [[ $(wc -l <Makefile) -gt 4 ]]; then
						#this makes you work on the CMakeLists.txt to keep up with the non-standard Makefile
						echo "message (FATAL_ERROR \"\${module_name} has a non standard Makefile (more than 4 lines) you need to modify the CMakeLists.txt!\")"
					fi

					echo "DECLAREPLUMEDMODULE(\${module_name}"
					#default activation state of the module
					case "$(cat "module.type")" in
						always) echo "\"always\"" ;;
						default-on) echo "ON" ;;
						default-off) echo "OFF" ;;
						#defaults to off
						*) echo "OFF" ;;
					esac
					echo "SOURCES"
					ls -1 *.cpp
					if grep -q USE Makefile; then
						echo "NEEDS"
						t=$(awk '/USE=/{print }' <Makefile)
						echo -e "\t${t#USE*=}"
						echo "DEPENDS"
						t=$(awk '/USE=/{print }' <Makefile)
						echo -e "\t${t#USE*=}"
					fi
					echo ")"
				} >CMakeLists.txt
				if ! grep -q "!/CMakeLists.txt" .gitignore && [[ -f .gitignore ]]; then
					echo "!/CMakeLists.txt" >>.gitignore
				fi
			)
		fi
	fi
}

for dir in */; do
	createCMakeLists "${dir///}"
done
