cmake_minimum_required(VERSION 3.10)
project(LANE_DETECTION
        VERSION 1.0.0
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(${PROJECT_NAME}_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(${PROJECT_NAME}_EXAMPLES ${${PROJECT_NAME}_PATH}/examples)

add_subdirectory(modules)

add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_EXAMPLES}/main.cpp)
# add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_PATH}/test/changine.cpp)
# add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_PATH}/test/test.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE
        LaneDetection
)
