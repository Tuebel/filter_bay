cmake_minimum_required(VERSION 3.5)
project(filter_bay)

# Default ROS2 configurations for compilation
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Create filter_bay library with interface definitions
add_library(filter_bay INTERFACE)
target_include_directories(filter_bay INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(filter_bay INTERFACE Eigen3::Eigen)

# installation
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)
install(FILES filter_bay-config.cmake DESTINATION lib/filter_bay/)
install(TARGETS filter_bay DESTINATION lib EXPORT filter_bay_targets)
install(EXPORT filter_bay_targets DESTINATION lib/filter_bay)
export(TARGETS filter_bay FILE filter_bay_exports.cmake)

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
