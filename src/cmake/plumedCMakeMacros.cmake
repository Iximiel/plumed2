include_guard(GLOBAL)
macro(ADDMODULETOKERNEL module_name)
    #use: ADDMODULETOKERNEL(module_name SOURCES listOfSources
    #[EXTRA_HEADERS files]
    #[NEEDS module names]
    #[DEPENDS module names]
    #)
    #Please write the source files explicitly
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SOURCES EXTRA_HEADERS NEEDS DEPENDS)
    cmake_parse_arguments(ADDMODULETOKERNEL "${options}" "${oneValueArgs}"
                      "${multiValueArgs}" "${ARGN}" )
    if (VERBOSE)
        message("for module ${module_name}")
        message("SOURCES ${ADDMODULETOKERNEL_SOURCES}")
        if(ADDMODULETOKERNEL_NEEDS)
            message("NEEDS ${ADDMODULETOKERNEL_NEEDS}")
        endif()
        if(ADDMODULETOKERNEL_DEPENDS)
            message("DEPENDS ${ADDMODULETOKERNEL_DEPENDS}")
        endif()
        if(ADDMODULETOKERNEL_EXTRA_HEADERS)
            message("EXTRA_HEADERS ${ADDMODULETOKERNEL_EXTRA_HEADERS}")
        endif()
    endif()

    set(moduleNeeds_${module_name} ${ADDMODULETOKERNEL_NEEDS} PARENT_SCOPE)
    
    if(${module_${module_name}} )
        add_library(${module_name} OBJECT ${ADDMODULETOKERNEL_SOURCES})
        target_include_directories(${module_name} PRIVATE ${PLUMED_SRC})
        list(APPEND modulesForKernel ${module_name})
        set(modulesForKernel ${modulesForKernel} PARENT_SCOPE)
        #add default headers
        foreach(file ${ADDMODULETOKERNEL_SOURCES})
            get_filename_component(filename ${file} NAME_WE)    
            if (EXISTS "${CMAKE_CURRENT_LIST_DIR}/${filename}.h")
                set_property(TARGET ${module_name} APPEND
                    PROPERTY PUBLIC_HEADER "${filename}.h")
            endif ()
        endforeach()
        if (ADDMODULETOKERNEL_DEPENDS)
            foreach(lib ${ADDMODULETOKERNEL_DEPENDS})
                #message("${module_name} is linked with ${lib}")
                target_link_libraries(${module_name} PUBLIC ${lib})
            endforeach(lib ${ADDMODULETOKERNEL_DEPENDS})
        endif(ADDMODULETOKERNEL_DEPENDS)
        if(ADDMODULETOKERNEL_EXTRA_HEADERS)
            set_property(TARGET ${module_name} APPEND
                PROPERTY PUBLIC_HEADER ${ADDMODULETOKERNEL_EXTRA_HEADERS})
        endif()
        install (TARGETS ${module_name}
            PUBLIC_HEADER
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/plumed/${module_name}
        )
    endif(${module_${module_name}})
endmacro(ADDMODULETOKERNEL)

function(CONFIGSETTINGS module_name settingFlag)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs LINK_LIBRARIES COMPILE_DEFINITIONS)
    cmake_parse_arguments(CONFIGSETTINGS "${options}" "${oneValueArgs}"
                      "${multiValueArgs}" "${ARGN}" )
    if (settingFlag)
        target_link_libraries     (${module_name}
            INTERFACE "${CONFIGSETTINGS_LINK_LIBRARIES}")
        if(CONFIGSETTINGS_COMPILE_DEFINITIONS)
            foreach(def "${CONFIGSETTINGS_COMPILE_DEFINITIONS}")
                target_compile_definitions(${module_name}
                    INTERFACE "${def}=1")
            endforeach(def "${CONFIGSETTINGS_COMPILE_DEFINITIONS}")
        endif()
    else()
        if(CONFIGSETTINGS_COMPILE_DEFINITIONS)
            foreach(def "${CONFIGSETTINGS_COMPILE_DEFINITIONS}")
                message(STATUS "cannot enable \"${def}\"")
            endforeach(def "${CONFIGSETTINGS_COMPILE_DEFINITIONS}")
        endif()
    endif(settingFlag)    
endfunction(CONFIGSETTINGS)



function(print_target_property target_name property)
    get_target_property(_${property} ${target_name} ${property})
    message("${target_name} <${property}>: ${_${property}}")
    unset(_${property})    
endfunction(print_target_property)