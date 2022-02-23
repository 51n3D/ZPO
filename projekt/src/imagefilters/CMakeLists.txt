
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE sources src/*.cpp src/*.h)

add_library(ImageFilters STATIC ${sources})
target_include_directories(ImageFilters PUBLIC src)
target_link_libraries(ImageFilters PRIVATE ${OpenCV_LIBS})
