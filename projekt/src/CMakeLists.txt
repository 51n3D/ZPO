cmake_minimum_required(VERSION 3.0.0)
project(ImageFilters)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INSTALL_PREFIX install)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt5Widgets REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory(imagefilters)
add_subdirectory(app)

