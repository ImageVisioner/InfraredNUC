cmake_minimum_required(VERSION 3.10)
project(infrared_destripe)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/main.cpp
    src/admm_algorithm.cpp
    src/lra_svd_denoising.cpp
    src/image_processing.cpp
)

# Create executable
add_executable(infrared_destripe ${SOURCES})

# Link OpenCV libraries
target_link_libraries(infrared_destripe ${OpenCV_LIBS})

# Compiler flags
if(MSVC)
    target_compile_options(infrared_destripe PRIVATE /W4)
else()
    target_compile_options(infrared_destripe PRIVATE -Wall -Wextra -O2)
endif()
