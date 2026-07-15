cmake_minimum_required(VERSION 3.16)

project(ncnn_omni VERSION 0.1.0 LANGUAGES CXX)

option(NCNN_OMNI_BUILD_TESTS "Build ncnn-omni tests" ON)
option(NCNN_OMNI_BUILD_TOOLS "Build ncnn-omni validation tools" ON)
set(NCNN_OMNI_NCNN_SOURCE "" CACHE PATH "Path to an ncnn source checkout")

if(TARGET ncnn)
    message(STATUS "Using the ncnn target supplied by the parent project")
elseif(NCNN_OMNI_NCNN_SOURCE)
    if(NOT EXISTS "${NCNN_OMNI_NCNN_SOURCE}/CMakeLists.txt")
        message(FATAL_ERROR "NCNN_OMNI_NCNN_SOURCE is not an ncnn source directory")
    endif()
    set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory("${NCNN_OMNI_NCNN_SOURCE}" "${CMAKE_BINARY_DIR}/ncnn")
else()
    find_package(ncnn CONFIG REQUIRED)
endif()

add_library(ncnn_omni
    src/models/qwen3_asr.cpp
    src/processors/qwen2_tokenizer.cpp
    src/processors/wav.cpp
    src/processors/whisper_log_mel.cpp
    src/runtime/ncnn/ncnn_module.cpp)

target_compile_features(ncnn_omni PUBLIC cxx_std_17)
target_include_directories(ncnn_omni
    PUBLIC include
    PRIVATE src)
target_link_libraries(ncnn_omni PUBLIC ncnn)

if(MSVC)
    target_compile_options(ncnn_omni PRIVATE /W4)
else()
    target_compile_options(ncnn_omni PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(ncnn-omni-asr examples/asr/qwen3_asr_cli.cpp)
target_link_libraries(ncnn-omni-asr PRIVATE ncnn_omni)

if(NCNN_OMNI_BUILD_TOOLS)
    add_executable(ncnn-omni-qwen3-asr-frontend-dump
        tools/parity/qwen3_asr_frontend_dump.cpp)
    target_include_directories(ncnn-omni-qwen3-asr-frontend-dump PRIVATE src)
    target_link_libraries(ncnn-omni-qwen3-asr-frontend-dump PRIVATE ncnn_omni)
endif()

if(NCNN_OMNI_BUILD_TESTS)
    enable_testing()
    add_executable(ncnn-omni-unit-tests
        tests/unit/test_main.cpp
        tests/unit/test_qwen2_tokenizer.cpp
        tests/unit/test_whisper_log_mel.cpp)
    target_include_directories(ncnn-omni-unit-tests PRIVATE src)
    target_link_libraries(ncnn-omni-unit-tests PRIVATE ncnn_omni)
    add_test(NAME ncnn-omni-unit-tests COMMAND ncnn-omni-unit-tests)
endif()
