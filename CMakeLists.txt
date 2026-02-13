cmake_minimum_required(VERSION 3.16)
project(vector_db)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(vector_db
    src/main.cpp
    src/index/flat/flat_index.cpp
)

target_include_directories(vector_db PRIVATE src)