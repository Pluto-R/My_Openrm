find_package(OpenCV REQUIRED)

add_library(
    openrm_cudatools
        SHARED
)
target_sources(
    openrm_cudatools
        PRIVATE
        ${CMAKE_SOURCE_DIR}/cuda/src/resize/affine.cc
        ${CMAKE_SOURCE_DIR}/cuda/src/resize/kernal.cc
        ${CMAKE_SOURCE_DIR}/cuda/src/resize/resize.cc
)

target_include_directories(
    openrm_cudatools
        PRIVATE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/cuda/include>
        $<INSTALL_INTERFACE:include/openrm>
)

target_link_libraries(openrm_cudatools ${OpenCV_LIBS})