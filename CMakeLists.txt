cmake_minimum_required(VERSION 3.20)

project(vkperf VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(VKPERF_SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders)
file(GLOB VKPERF_SHADER_SOURCES CONFIGURE_DEPENDS
    ${VKPERF_SHADER_SOURCE_DIR}/*.comp
)

if(NOT VKPERF_SHADER_SOURCES)
    message(FATAL_ERROR "No shader sources found in ${VKPERF_SHADER_SOURCE_DIR}")
endif()

find_program(GLSLANG_VALIDATOR_EXECUTABLE
    NAMES glslangValidator glslangValidator.exe
    HINTS
        "$ENV{VULKAN_SDK}/Bin"
        "$ENV{VULKAN_SDK}/Bin32"
    REQUIRED
)

set(VKPERF_COMPILED_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated_shaders)
foreach(VKPERF_SHADER_SOURCE IN LISTS VKPERF_SHADER_SOURCES)
    get_filename_component(VKPERF_SHADER_NAME_WE ${VKPERF_SHADER_SOURCE} NAME_WE)
    set(VKPERF_SHADER_SPV ${VKPERF_COMPILED_SHADER_DIR}/${VKPERF_SHADER_NAME_WE}.spv)

    add_custom_command(
        OUTPUT ${VKPERF_SHADER_SPV}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${VKPERF_COMPILED_SHADER_DIR}
        COMMAND ${GLSLANG_VALIDATOR_EXECUTABLE} -V -o ${VKPERF_SHADER_SPV} ${VKPERF_SHADER_SOURCE}
        DEPENDS ${VKPERF_SHADER_SOURCE}
        VERBATIM
    )

    list(APPEND VKPERF_SHADER_SPV_FILES ${VKPERF_SHADER_SPV})
endforeach()

set(VKPERF_GENERATED_RESOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated_resources)
set(VKPERF_GENERATED_SHADER_HEADER ${VKPERF_GENERATED_RESOURCE_DIR}/compiled_spv.h)
add_custom_command(
    OUTPUT ${VKPERF_GENERATED_SHADER_HEADER}
    COMMAND ${CMAKE_COMMAND}
        -DOUTPUT=${VKPERF_GENERATED_SHADER_HEADER}
        "-DRESOURCE_FILES=${VKPERF_SHADER_SPV_FILES}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spv_resources.cmake
    DEPENDS
        ${VKPERF_SHADER_SPV_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spv_resources.cmake
    VERBATIM
)
add_custom_target(vkperf_embed_shaders DEPENDS ${VKPERF_GENERATED_SHADER_HEADER})
set_source_files_properties(${VKPERF_GENERATED_SHADER_HEADER} PROPERTIES GENERATED TRUE)
set_source_files_properties(src/resources.c PROPERTIES OBJECT_DEPENDS ${VKPERF_GENERATED_SHADER_HEADER})

add_executable(vkperf
    src/buffer_filler.c
    src/helper.c
    src/latency_helper.c
    src/logger.c
    src/main.c
    src/resources.c
    src/runner.c
    src/vulkan_command_buffer.c
    src/vulkan_compute_pipeline.c
    src/vulkan_helper.c
    src/vulkan_memory.c
    src/vulkan_runner.c
    src/vulkan_shader.c
    src/vulkan_staging.c
    src/vulkan_texture.c
    src/tests/test_vk_bandwidth.c
    src/tests/test_vk_cuprobe.cpp
    src/tests/test_vk_info.c
    src/tests/test_vk_latency.c
    src/tests/test_vk_list.c
    src/tests/test_vk_rate.c
    src/tests/test_vk_rate_texture_fill.c
    src/tests/test_vk_storage_type.c
    src/tests/test_vk_uplink.c
    ${VKPERF_GENERATED_SHADER_HEADER}
)

add_dependencies(vkperf vkperf_embed_shaders)

target_include_directories(vkperf PRIVATE
    ${VKPERF_GENERATED_RESOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(vkperf PRIVATE _CLI)

if(WIN32)
    target_sources(vkperf PRIVATE src/windows_helper.c)
    target_link_libraries(vkperf PRIVATE setupapi)
endif()

if(ANDROID)
    target_link_libraries(vkperf PRIVATE vulkan)
else()
    find_package(Vulkan REQUIRED)
    target_link_libraries(vkperf PRIVATE Vulkan::Vulkan)
endif()

if(NOT WIN32)
    find_package(Threads REQUIRED)
    target_link_libraries(vkperf PRIVATE Threads::Threads)
endif()

if(MSVC)
    target_compile_options(vkperf PRIVATE /W3 /wd4996)
else()
    target_compile_options(vkperf PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
