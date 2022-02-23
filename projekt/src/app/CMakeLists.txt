
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE sources gui/*.cpp gui/*.h window.ui cvfilters/*.h)

add_executable(CamFilters main.cpp ${sources})
target_link_libraries(CamFilters PRIVATE ${OpenCV_LIBS})
target_link_libraries(CamFilters PRIVATE ImageFilters)
target_link_libraries(CamFilters PRIVATE Qt5::Widgets)
