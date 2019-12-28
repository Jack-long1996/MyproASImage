cmake_minimum_required(VERSION 3.15)
SET(execName MyproASImage)
PROJECT(${execName})
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(OpenCV_DIR "C:\\OpenCV\\opencv\\MinGWBuild\\install")
find_package(OpenCV REQUIRED)
find_package(Qt5 COMPONENTS Widgets REQUIRED)

include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

FILE(GLOB_RECURSE SRC_FILES src/*.cpp)
FILE(GLOB_RECURSE HDR_FILES include/*.hpp)

add_executable(${execName} ${HDR_FILES} ${SRC_FILES})
TARGET_LINK_LIBRARIES(${execName} ${OpenCV_LIBRARIES})