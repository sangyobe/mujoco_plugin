# --------------------------------------------------------
# ENVIRONMENT SECTION
# --------------------------------------------------------
cmake_minimum_required(VERSION 3.16...3.25)

# Cmake Policies
# set(CMAKE_POLICY_DEFAULT_CMP0002 NEW)

# Project Definition
project(
	mujoco_plugins
	VERSION 2.0.0
	DESCRIPTION ""
	HOMEPAGE_URL ""
)

# Add CMake Module Path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Include CMake Modules
include(DebugUtil)
include(GetDirectoryName)
include(ProcessorCount)

# Parallize Build Process
ProcessorCount(procNum)
if("${procNum} - 2" GREATER_EQUAL "0")
    message("Processor Number : " ${procNum})
    MATH(EXPR procNum "${procNum}-2")
    set(ENV{CMAKE_BUILD_PARALLEL_LEVEL} ${procNum})
    message("CMAKE_BUILD_PARALLEL_LEVEL : " $ENV{CMAKE_BUILD_PARALLEL_LEVEL})
endif()

# C++ Specification
enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # For LLVM tooling
set(CMAKE_CXX_COMPILER_FORCED ON)

# Layout build dir like install dir
include(GNUInstallDirs)
option(BUILD_SHARED_LIBS "Build shared libraries." ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})
set(CMAKE_INSTALL_RPATH ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

# ART Framework
file(TO_CMAKE_PATH "$ENV{ARTF_INSTALL_DIR}" ARTF_INSTALL_DIR)
if ("${ARTF_INSTALL_DIR}" STREQUAL "")
    message(FATAL_ERROR "ARTF_INSTALL_DIR is not set.")
else()
    # PRINT_VAR(ARTF_INSTALL_DIR)
    message(STATUS "ARTF_INSTALL_DIR=${ARTF_INSTALL_DIR}")
endif()
list(APPEND CMAKE_PREFIX_PATH "${ARTF_INSTALL_DIR}")

# Project Options
option(BUILD_SENSOR_PLUGIN "Build sensor plugins for ${PROJECT_NAME}" ON)

# Project Dependencies
find_package(artf CONFIG REQUIRED)

# --------------------------------------------------------
# BUILD SECTION
# --------------------------------------------------------
add_subdirectory(plugin/sensor/touch_grid)

# --------------------------------------------------------
# DEPLOY SECTION
# --------------------------------------------------------

