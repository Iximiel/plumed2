include_guard(GLOBAL)
macro(DECLAREPLUMEDMODULE module_name default_status)
    #use: DECLAREPLUMEDMODULE(<module_name> <default_status>
    #SOURCES listOfSources
    #[EXTRA_HEADERS files]
    #[NEEDS module names]
    #[DEPENDS module names]
    #)
    #Please write the source files explicitly
    # ``<default_status>`` must be either ON OFF or "always"
    
    #the first set and option are not overriden on subsequent runs of cmake
    if(${default_status} STREQUAL "always")
        set(module_${module_name} ON CACHE INTERNAL "always active module ${module_name}")
        else()
        option(module_${module_name} "activate module ${module_name}" ${default_status})
    endif(${default_status} STREQUAL "always")
    set(module_default_${module_name} ${default_status} CACHE INTERNAL "default status of the module ${module_name}")
    
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SOURCES EXTRA_HEADERS NEEDS DEPENDS)
    cmake_parse_arguments(DECLAREPLUMEDMODULE "${options}" "${oneValueArgs}"
                      "${multiValueArgs}" "${ARGN}" )
    if (VERBOSE)
        message("for module ${module_name}")
        message("SOURCES ${DECLAREPLUMEDMODULE_SOURCES}")
        if(DECLAREPLUMEDMODULE_NEEDS)
            message("NEEDS ${DECLAREPLUMEDMODULE_NEEDS}")
        endif()
        if(DECLAREPLUMEDMODULE_DEPENDS)
            message("DEPENDS ${DECLAREPLUMEDMODULE_DEPENDS}")
        endif()
        if(DECLAREPLUMEDMODULE_EXTRA_HEADERS)
            message("EXTRA_HEADERS ${DECLAREPLUMEDMODULE_EXTRA_HEADERS}")
        endif()
    endif()

    set(moduleNeeds_${module_name} ${DECLAREPLUMEDMODULE_NEEDS} PARENT_SCOPE)
    
    if(${module_${module_name}} )
        add_library(${module_name} OBJECT ${DECLAREPLUMEDMODULE_SOURCES})
        target_include_directories(${module_name} PRIVATE ${PLUMED_SRC})
        list(APPEND modulesForKernel ${module_name})
        set(modulesForKernel ${modulesForKernel} PARENT_SCOPE)
        #add default headers
        foreach(file ${DECLAREPLUMEDMODULE_SOURCES})
            get_filename_component(filename ${file} NAME_WE)    
            if (EXISTS "${CMAKE_CURRENT_LIST_DIR}/${filename}.h")
                set_property(TARGET ${module_name} APPEND
                    PROPERTY PUBLIC_HEADER "${filename}.h")
            endif ()
        endforeach()
        if (DECLAREPLUMEDMODULE_DEPENDS)
            foreach(lib ${DECLAREPLUMEDMODULE_DEPENDS})
                #message("${module_name} is linked with ${lib}")
                target_link_libraries(${module_name} PUBLIC ${lib})
            endforeach(lib ${DECLAREPLUMEDMODULE_DEPENDS})
        endif(DECLAREPLUMEDMODULE_DEPENDS)
        if(DECLAREPLUMEDMODULE_EXTRA_HEADERS)
            set_property(TARGET ${module_name} APPEND
                PROPERTY PUBLIC_HEADER ${DECLAREPLUMEDMODULE_EXTRA_HEADERS})
        endif()
        install (TARGETS ${module_name}
            PUBLIC_HEADER
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/plumed/${module_name}
        )
    endif(${module_${module_name}})
endmacro(DECLAREPLUMEDMODULE)

function(CONFIGSETTINGS module_name settingFlag)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs LINK_LIBRARIES COMPILE_DEFINITIONS PLUMED_DYNAMIC_LIBS)
    cmake_parse_arguments(CONFIGSETTINGS "${options}" "${oneValueArgs}"
                      "${multiValueArgs}" "${ARGN}" )
    if (${settingFlag})
        target_link_libraries (${module_name}
            INTERFACE "${CONFIGSETTINGS_LINK_LIBRARIES}")
        if(CONFIGSETTINGS_COMPILE_DEFINITIONS)
            foreach(def "${CONFIGSETTINGS_COMPILE_DEFINITIONS}")
                target_compile_definitions(${module_name}
                    INTERFACE "${def}=1")
            endforeach(def "${CONFIGSETTINGS_COMPILE_DEFINITIONS}")
        endif()
        if(CONFIGSETTINGS_PLUMED_DYNAMIC_LIBS)
            list(APPEND PLUMED_DYNAMIC_LIBS ${CONFIGSETTINGS_PLUMED_DYNAMIC_LIBS})
            set(PLUMED_DYNAMIC_LIBS ${PLUMED_DYNAMIC_LIBS} PARENT_SCOPE)
        endif()
    else()
        if(CONFIGSETTINGS_COMPILE_DEFINITIONS)
            foreach(def "${CONFIGSETTINGS_COMPILE_DEFINITIONS}")
                message(STATUS "cannot enable \"${def}\"")
            endforeach(def "${CONFIGSETTINGS_COMPILE_DEFINITIONS}")
        endif()
    endif(${settingFlag})    
endfunction(CONFIGSETTINGS)

function(print_target_property target_name property)
    get_target_property(my${property} ${target_name} ${property})
    if (my${property})
        message("-- ${target_name} <${property}>: ${my${property}}")
        else()
        message("-- ${target_name} <${property}>:")
    endif()
    unset(my${property})    
endfunction(print_target_property)
