
macro(add_plumed_plugin plugin_name)
  #use: DECLAREPLUMEDMODULE(<plugin_name>
  #SOURCES listOfSources
  #)
  #Please write the source files explicitly
  #This creates a target library called <plugin_name> that is linked to
  #Plumed2::Config and has will produce a <plugin_name>.so or <plugin_name>.dylib
  #You can add extra libraries by using the standard target_link_libraries()

  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SOURCES)
  cmake_parse_arguments(PLUMEDPLUGIN "${options}" "${oneValueArgs}"
                      "${multiValueArgs}" "${ARGN}" )

  add_library(${plugin_name} SHARED ${PLUMEDPLUGIN_SOURCES})
  target_link_libraries(${plugin_name} PUBLIC
      Plumed2::Config
  )
  set_target_properties(${plugin_name} PROPERTIES PREFIX "")
endmacro(add_plumed_plugin)