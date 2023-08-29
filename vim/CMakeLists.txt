
add_custom_command(OUTPUT syntax/plumed.vim syntax/plumedf.vim
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/vimsyntax.sh $<TARGET_FILE:Plumed2::plumed_bin>
        DEPENDS Plumed2::plumed_bin
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating vim syntax files")

add_custom_target(vimSyntax ALL 
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/syntax/plumed.vim)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/syntax/plumed.vim"
              "${CMAKE_CURRENT_BINARY_DIR}/syntax/plumedf.vim"
        DESTINATION "${DIRS_EXTRA_INSTALL}/vim/syntax"
)
install(FILES scripts.vim
        DESTINATION ${DIRS_EXTRA_INSTALL}/vim)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/help
        DESTINATION ${DIRS_EXTRA_INSTALL}/vim)