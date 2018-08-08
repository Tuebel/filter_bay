cmake_minimum_required(VERSION 3.5)
project(filter_bay)

# Default ROS2 configurations for compilation
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# enable testing with ctest
include(CTest)
# external packages
find_package(Eigen3 3.3 REQUIRED)

include_directories(include)

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
      pthread
    Eigen3::Eigen)
    gtest_discover_tests(${testname})
  endforeach(sourcefile ${TEST_SRCS})
endif()

## Install include dir
install(
  DIRECTORY include/filter_bay
  DESTINATION include)
## Install executable and lib
install(TARGETS
  ARCHIVE DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib/${PROJECT_NAME}
  RUNTIME DESTINATION lib/${PROJECT_NAME})
