cmake_minimum_required(VERSION 3.14)
include(FetchContent)

project(fbstab)
enable_testing()

# Export include directory.
set(ENV{FBSTAB_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# TODO(dliaomcp@umich.edu)
# Figure out how to add error and warning flags to the build without
# breaking Eigen or googletest...
set(FBSTAB_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# Finds a system Eigen,
# if you haven't run Eigen's cmake + install scripts then
# you need to supply it by setting the Eigen3_DIR variable.
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Make googletest available if FBstab is the main project.
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME))
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.10.0
)
FetchContent_MakeAvailable(googletest)
endif()

# The cmake build system isn't used for debugging,
# so default to an optimized build.
add_compile_options(-O3 -g0 -std=c++11) 
add_subdirectory(fbstab)
