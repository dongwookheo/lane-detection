cmake_minimum_required(VERSION 3.10)
project(lane_detection VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(modules)

add_executable(main examples/main.cpp)
# add_executable(main examples/simple_threshold.cpp)
# add_executable(main examples/test_hough.cpp)
# add_executable(main examples/nahye_hough.cpp)
# add_executable(main examples/dongwook_test.cpp)
# add_executable(main examples/image_test.cpp)
# add_executable(main examples/nahye_line_detection.cpp)
# add_executable(main examples/test.cpp)

target_link_libraries(main PRIVATE
        LaneDetection
)
