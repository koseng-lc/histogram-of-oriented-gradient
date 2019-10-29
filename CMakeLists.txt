cmake_minimum_required(VERSION 2.8.3)
project(HOG)
add_compile_options(-std=c++11)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc objdetect calib3d ml)
add_executable(${PROJECT_NAME} hog.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBRARIES}
)
