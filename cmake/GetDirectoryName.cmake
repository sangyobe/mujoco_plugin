cmake_minimum_required(VERSION 3.8)

function(GetCurrentDirectoryName dir_name)
    get_filename_component(list_name ${CMAKE_CURRENT_LIST_DIR} NAME)
    string(REPLACE " " "_" list_name ${list_name})
    set("${dir_name}" "${list_name}" PARENT_SCOPE)
endfunction()