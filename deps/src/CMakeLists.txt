cmake_minimum_required(VERSION 2.8.7)

project(Caffe C CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

find_package(Protobuf REQUIRED)

set(Caffe_LINKER_LIBS "")

file(GLOB ProtoFiles "${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/caffe/proto/*.proto")

PROTOBUF_GENERATE_CPP(ProtoSources ProtoHeaders ${ProtoFiles})

include_directories(${CMAKE_BINARY_DIR}) # for protobuf files

add_custom_command(
	OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/caffe/include/caffe/proto/caffe.pb.h
	DEPENDS ${CMAKE_BINARY_DIR}/caffe.pb.h
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_BINARY_DIR}/caffe.pb.h ${CMAKE_CURRENT_SOURCE_DIR}/caffe/include/caffe/proto/caffe.pb.h)

find_package(Boost 1.46 REQUIRED COMPONENTS system thread)
include_directories(SYSTEM ${Boost_INCLUDE_DIR})
list(APPEND Caffe_LINKER_LIBS ${Boost_LIBRARIES})

find_package(Threads REQUIRED)
list(APPEND Caffe_LINKER_LIBS ${CMAKE_THREAD_LIBS_INIT})

find_package(Glog REQUIRED)
include_directories(SYSTEM ${GLOG_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS ${GLOG_LIBRARIES})

find_package(GFlags REQUIRED)
include_directories(SYSTEM ${GFLAGS_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS ${GFLAGS_LIBRARIES})

find_package(CUDA QUIET)
if(CUDA_FOUND)
	include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
else(CUDA_FOUND)
	add_definitions(-DCPU_ONLY)
endif(CUDA_FOUND)

## Other stuff

option(USE_64_BIT_BLAS "Use 64 bit version of BLAS with suffix 64_" OFF)

if(USE_64_BIT_BLAS)
	add_definitions(-DBLASSUFFIX=1 -DCAFFE_HEADLESS)
else(USE_64_BIT_BLAS)
	add_definitions(-DBLASSUFFIX=0 -DCAFFE_HEADLESS)
endif(USE_64_BIT_BLAS)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/caffe/include/)
include_directories(${PROJECT_BINARY_DIR}/caffe/include/)

# switch of symbol undefined warning for BLAS on Mac OS

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -undefined dynamic_lookup")
endif()

# for gtest
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/)

## The files

file(GLOB Source GLOB "${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/caffe/*.cpp")
file(GLOB SourceUtil GLOB "${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/caffe/util/*.cpp")
file(GLOB SourceGPUUtil GLOB "${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/caffe/util/*.cu")
file(GLOB SourceLayers GLOB "${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/caffe/layers/*.cpp")
file(GLOB SourceGPULayers GLOB "${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/caffe/layers/*.cu")

list(REMOVE_ITEM SourceLayers "${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/caffe/layers/data_layer.cpp")
list(REMOVE_ITEM SourceGPULayers "${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/caffe/layers/data_layer.cu")

list(REMOVE_ITEM SourceLayers "${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/caffe/layers/hdf5_output_layer.cpp")
list(REMOVE_ITEM SourceGPULayers "${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/caffe/layers/hdf5_output_layer.cu")

list(REMOVE_ITEM SourceLayers "${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/caffe/layers/hdf5_data_layer.cpp")
list(REMOVE_ITEM SourceGPULayers "${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/caffe/layers/hdf5_data_layer.cu")

list(REMOVE_ITEM SourceLayers "${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/caffe/layers/image_data_layer.cpp")
list(REMOVE_ITEM SourceGPULayers "${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/caffe/layers/image_data_layer.cu")

list(REMOVE_ITEM SourceLayers "${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/caffe/layers/window_data_layer.cpp")
list(REMOVE_ITEM SourceGPULayers "${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/caffe/layers/window_data_layer.cu")

list(REMOVE_ITEM SourceLayers "${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/caffe/layers/mat_data_layer.cpp")

list(REMOVE_ITEM SourceUtil "${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/caffe/util/db.cpp")
list(REMOVE_ITEM SourceGPUUtil "${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/caffe/util/db.cu")

list(REMOVE_ITEM SourceUtil "${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/caffe/util/hdf5.cpp")
list(REMOVE_ITEM SourceUtil "${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/caffe/util/db_lmdb.cpp")
list(REMOVE_ITEM SourceUtil "${CMAKE_CURRENT_SOURCE_DIR}/caffe/src/caffe/util/db_leveldb.cpp")

# force generation of proto header
add_library(proto STATIC ${CMAKE_CURRENT_SOURCE_DIR}/caffe/include/caffe/proto/caffe.pb.h ${ProtoSources})
set(Caffe_LINKER_LIBS proto ${Caffe_LINKER_LIBS})

if(CUDA_FOUND)
	cuda_add_library(jlcaffe SHARED ${ProtoSources} ${ProtoHeaders} ${Source} ${SourceLayers} ${SourceGPULayers} ${SourceUtil} ${SourceGPUUtil} jlcaffe/jlcaffe.cpp ${CMAKE_CURRENT_SOURCE_DIR}/caffe/include/caffe/proto/caffe.pb.h)

	cuda_add_cublas_to_target(jlcaffe)

	target_link_libraries(jlcaffe ${PROTOBUF_LIBRARY} ${Caffe_LINKER_LIBS} ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})
else(CUDA_FOUND)
	add_library(jlcaffe SHARED ${ProtoSources} ${ProtoHeaders} ${Source} ${SourceLayers} ${SourceUtil} jlcaffe/jlcaffe.cpp ${CMAKE_CURRENT_SOURCE_DIR}/caffe/include/caffe/proto/caffe.pb.h)

	target_link_libraries(jlcaffe ${PROTOBUF_LIBRARY} ${Caffe_LINKER_LIBS})
endif(CUDA_FOUND)

