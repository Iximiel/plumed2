set(SCRIPTS
config.sh
mklib.sh
newcv.sh
partial_tempering.sh
patch.sh
selector.sh
vim2html.sh
)
Message(AUTHOR_WARNING "the scripts are configured with the \"Install\" configuration")
foreach(script ${SCRIPTS})
    install(FILES ${script}
        DESTINATION lib/plumed/scripts
    )
    get_filename_component(adaptedScript ${script} NAME_WE) 
    set(adaptedScript plumed-${adaptedScript})
    # The logic implemented here should be the same as in config/Config.inc.in
    # Please keep them consistent
    #here is set up the  "ifeq ($(PLUMED_INSTALL),Install)"
    # If plumed is installed, single paths can be overridden using env vars
    file(CONFIGURE OUTPUT ${adaptedScript}
        CONTENT
"#! /usr/bin/env bash
echo 'NOTE: shell only version, useful when plumed is cross compiled' >&2
export PLUMED_IS_INSTALLED=yes
export PLUMED_ROOT=\"\${PLUMED_ROOT-@CMAKE_INSTALL_PREFIX@/lib/@program_name@}\"
export PLUMED_INCLUDEDIR=\"\${PLUMED_INCLUDEDIR-@CMAKE_INSTALL_PREFIX@/include}\"
export PLUMED_HTMLDIR=\"\${PLUMED_HTMLDIR=@CMAKE_INSTALL_PREFIX@/share/doc/plumed}\"
export PLUMED_PROGRAM_NAME=\"\${PLUMED_PROGRAM_NAME-@program_name@}\"
export PLUMED_VERSION=@PLUMED_VERSION_LONG@
source \"\${PLUMED_ROOT}/scripts/@script@\""
@ONLY
     )
    
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${adaptedScript}
        DESTINATION lib/${program_name}
        PERMISSIONS
         OWNER_WRITE
         OWNER_READ GROUP_READ WORLD_READ
         OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE
    )
endforeach(script ${SCRIPTS})
