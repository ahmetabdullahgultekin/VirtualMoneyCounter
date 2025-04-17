cmake_minimum_required(VERSION 3.30)
project(VirtualMoneyCounter)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_C_STANDARD 23)

# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Include the OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(VirtualMoneyCounter src/Main.cpp src/vc.c src/vc.h)

# Link the OpenCV libraries to your executable
target_link_libraries(VirtualMoneyCounter ${OpenCV_LIBS})
