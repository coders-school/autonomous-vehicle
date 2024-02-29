cmake_minimum_required(VERSION 3.0)
project(YourProjectName)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
    opencv
    GIT_REPOSITORY "https://github.com/opencv/opencv.git"
    GIT_TAG 4.9.0
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(opencv)

set(MY_SOURCES 
    src/main.cpp
)

include(OpenCVConfig.cmake)

add_executable(${PROJECT_NAME} ${MY_SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE ${OPENCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OPENCV_LIBRARIES})

add_subdirectory(test)