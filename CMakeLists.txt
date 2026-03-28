cmake_minimum_required(VERSION 3.16)

# Set build type BEFORE project()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

project(vector_db)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release optimizations and SIMD flags  
add_compile_options(-O3 -mavx2)

find_package(OpenMP REQUIRED)

set(VECTOR_DB_SOURCES
    src/main.cpp
    src/index/flat/flat_index.cpp
    src/index/hnsw/hnsw_index.cpp
    src/index/ivf/ivf_index.cpp
)

add_executable(vector_db ${VECTOR_DB_SOURCES})

target_include_directories(vector_db PRIVATE src)
target_link_libraries(vector_db PUBLIC OpenMP::OpenMP_CXX)