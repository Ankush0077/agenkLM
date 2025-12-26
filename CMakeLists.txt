# CMakeLists.txt (FINAL, DEFINITIVE)
cmake_minimum_required(VERSION 3.10)
project(fmm_model CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
find_library(LMDB_LIBRARY lmdb)
find_package(Threads REQUIRED)
add_executable(fmm src/main.cpp src/inference.cpp)
target_link_libraries(fmm PRIVATE ${LMDB_LIBRARY} Threads::Threads OpenMP::OpenMP_CXX)
target_include_directories(fmm PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/hnswlib
    ${CMAKE_CURRENT_SOURCE_DIR}/libs # For nlohmann
)