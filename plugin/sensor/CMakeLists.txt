# file(GLOB SENSOR_DIRS "*")
set(SENSOR_DIRS "")
list(APPEND SENSOR_DIRS "./touch_grid")
list(APPEND SENSOR_DIRS "./elevation_map")

foreach(sensor_dir ${SENSOR_DIRS})
    message(STATUS "build ${sensor_dir}...")
    add_subdirectory(${sensor_dir})
endforeach()
