cmake_minimum_required(VERSION 3.5)
project(filter_bay)

# Default ROS2 configurations for compilation
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# filter_bay is header only -> INTERFACE lib
add_library(filter_bay INTERFACE)
target_include_directories(filter_bay INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_sources(filter_bay INTERFACE
  INTERFACE $<BUILD_INTERFACE:
  ${CMAKE_CURRENT_SOURCE_DIR}/include/filter_bay/kalman_filter/kalman_filter.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/filter_bay/model/depth_observation_model.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/filter_bay/model/gaussian_observation_model.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/filter_bay/model/linear_transition_model.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/filter_bay/model/movement_state.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/filter_bay/particle_filter/log_particle_filter.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/filter_bay/particle_filter/particle_filter.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/filter_bay/tracker/impedance_tracker.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/filter_bay/utility/log_arithmetics.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/filter_bay/utility/normal_sampler.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/filter_bay/utility/uniform_random.hpp>)
target_link_libraries(filter_bay INTERFACE 
  Eigen3::Eigen)

# install target
install(DIRECTORY include/filter_bay DESTINATION include)
install(TARGETS filter_bay
  EXPORT filter_bay_targets)
# versioning
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  filter_bayConfigVersion.cmake
  VERSION 0.1
  COMPATIBILITY AnyNewerVersion)
# for find_package support
install(EXPORT filter_bay_targets
  FILE filter_bay_targets.cmake
  NAMESPACE filter_bay::
  DESTINATION lib/cmake/filter_bay)
configure_file(filter_bayConfig.cmake.in filter_bayConfig.cmake @ONLY)
install(FILES 
  "${CMAKE_CURRENT_BINARY_DIR}/filter_bayConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/filter_bayConfigVersion.cmake"
  DESTINATION lib/cmake/filter_bay)
# export for source tree
export(TARGETS filter_bay
  NAMESPACE filter_bay::
  FILE filter_bay_targets.cmake)
export(PACKAGE filter_bay)

# testing
include(CTest)
if(${BUILD_TESTING})
  # unit testing with google test
  find_package(GTest REQUIRED)
  # add all sources from the test/ directory as test
  file(GLOB TEST_SRCS test/*.cpp)
  foreach(sourcefile ${TEST_SRCS})
    get_filename_component(testname ${sourcefile} NAME_WE)
    add_executable(${testname} ${sourcefile})
    target_link_libraries (${testname}
      ${GTEST_LIBRARIES}
      filter_bay
      pthread)
    gtest_discover_tests(${testname})
  endforeach(sourcefile ${TEST_SRCS})
endif()
