cmake_minimum_required(VERSION 3.10)
project(LaneDetection VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(${PROJECT_NAME}_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(${PROJECT_NAME}_SOURCE_PATH "${${PROJECT_NAME}_PATH}/src")

file(GLOB ${PROJECT_NAME}_SOURCE_FILES
    "${${PROJECT_NAME}_SOURCE_PATH}/*.cpp"
)

add_library(LaneDetection
        ${${PROJECT_NAME}_SOURCE_FILES}
)

find_package(OpenCV REQUIRED HINTS ${CMAKE_SOURCE_DIR}/thirdparty/OpenCV/install/lib/cmake/opencv4)

if(OpenCV_FOUND)
    message(STATUS "OpenCV Found! - ${OpenCV_DIR}")
endif()

target_include_directories(LaneDetection PUBLIC
        include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(LaneDetection PRIVATE
        ${OpenCV_LIBS}
)