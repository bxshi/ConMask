cmake_minimum_required(VERSION 3.0)
project(target_sampling)

set(CMAKE_CXX_STANDARD 11)

EXEC_PROGRAM(uname OUTPUT_VARIABLE OS)
MESSAGE(STATUS "OS Type: " ${OS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -shared -fPIC -O2")

if (${OS} STREQUAL "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -undefined dynamic_lookup")
endif ()

EXEC_PROGRAM("python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())' 2>/dev/null | grep -v 'I'" OUTPUT_VARIABLE TF_FILES ERROR_VARIABLE TF_ERRORS)
message(STATUS "TensorFlow include dir: " ${TF_FILES})

set(SOURCE_FILES pairwise_sampler.cpp)

include_directories(${TF_FILES})

add_library(sampling SHARED ${SOURCE_FILES})