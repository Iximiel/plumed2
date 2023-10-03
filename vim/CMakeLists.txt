set(PLUMEDVIM_BINARY_DIR ${CMAKE_BINARY_DIR}/${DIRS_EXTRA_INSTALL}/vim)

add_custom_command(OUTPUT
        ${PLUMEDVIM_BINARY_DIR}/syntax/plumed.vim 
        ${PLUMEDVIM_BINARY_DIR}/syntax/plumedf.vim
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/vimsyntax.sh $<TARGET_FILE:Plumed2::plumed_bin> ${PLUMEDVIM_BINARY_DIR}
        DEPENDS Plumed2::plumed_bin
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating vim syntax files")

add_custom_target(vimSyntax ALL 
        DEPENDS ${PLUMEDVIM_BINARY_DIR}/syntax/plumed.vim)
install(FILES "${PLUMEDVIM_BINARY_DIR}/syntax/plumed.vim"
              "${PLUMEDVIM_BINARY_DIR}/syntax/plumedf.vim"
        DESTINATION "${DIRS_EXTRA_INSTALL}/vim/syntax"
)
install(FILES scripts.vim
        DESTINATION ${DIRS_EXTRA_INSTALL}/vim)
install(DIRECTORY ${PLUMEDVIM_BINARY_DIR}/help
        DESTINATION ${DIRS_EXTRA_INSTALL}/vim)
#build-only
file(COPY
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts.vim
     DESTINATION ${PLUMEDVIM_BINARY_DIR})

