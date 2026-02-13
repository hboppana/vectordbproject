cmake_minimum_required(VERSION 3.16)
project(vector_db)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(VECTOR_DB_SOURCES
    src/main.cpp
    src/index/flat/flat_index.cpp
    src/index/hnsw/hnsw_index.cpp
)

add_executable(vector_db ${VECTOR_DB_SOURCES})

target_include_directories(vector_db PRIVATE src)