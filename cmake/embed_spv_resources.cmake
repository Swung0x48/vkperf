if(NOT DEFINED OUTPUT)
    message(FATAL_ERROR "OUTPUT is required")
endif()

if(NOT DEFINED RESOURCE_FILES)
    message(FATAL_ERROR "RESOURCE_FILES is required")
endif()

get_filename_component(output_dir "${OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${output_dir}")

set(tmp_output "${OUTPUT}.tmp")
file(WRITE "${tmp_output}" "// !!! THIS FILE IS MACHINE GENERATED, ANY EDITS WILL BE OVERWRITTEN !!!\n\n")
file(APPEND "${tmp_output}" "#ifndef VKPERF_COMPILED_SPV_H\n")
file(APPEND "${tmp_output}" "#define VKPERF_COMPILED_SPV_H\n\n")
file(APPEND "${tmp_output}" "#ifdef EMBED_RESOURCES\n\n")
file(APPEND "${tmp_output}" "#ifndef _AB_EMBEDDED_RESOURCE_STRUCT_DEFINED\n")
file(APPEND "${tmp_output}" "#define _AB_EMBEDDED_RESOURCE_STRUCT_DEFINED\n")
file(APPEND "${tmp_output}" "struct _ab_embedded_resource_t {\n")
file(APPEND "${tmp_output}" "    const char *file_name;\n")
file(APPEND "${tmp_output}" "    size_t file_size;\n")
file(APPEND "${tmp_output}" "    const unsigned char *file_data;\n")
file(APPEND "${tmp_output}" "};\n")
file(APPEND "${tmp_output}" "#endif\n\n")

set(resource_count 0)
set(resource_entries "")

foreach(resource_file IN LISTS RESOURCE_FILES)
    if(NOT EXISTS "${resource_file}")
        message(FATAL_ERROR "Resource file does not exist: ${resource_file}")
    endif()

    get_filename_component(resource_name "${resource_file}" NAME)
    file(READ "${resource_file}" resource_hex HEX)
    string(LENGTH "${resource_hex}" resource_hex_length)
    math(EXPR resource_size "${resource_hex_length} / 2")

    set(array_name "compiled_spv_content_${resource_count}")
    file(APPEND "${tmp_output}" "static const unsigned char ${array_name}[] = {\n")

    set(byte_offset 0)
    set(line "    ")
    while(byte_offset LESS resource_hex_length)
        string(SUBSTRING "${resource_hex}" "${byte_offset}" 2 byte_hex)
        string(APPEND line "0x${byte_hex}")

        math(EXPR next_byte_offset "${byte_offset} + 2")
        if(next_byte_offset LESS resource_hex_length)
            string(APPEND line ",")
        endif()

        math(EXPR byte_index "${byte_offset} / 2")
        math(EXPR line_index "${byte_index} % 16")
        if(line_index EQUAL 15)
            file(APPEND "${tmp_output}" "${line}\n")
            set(line "    ")
        else()
            string(APPEND line " ")
        endif()

        set(byte_offset "${next_byte_offset}")
    endwhile()

    string(STRIP "${line}" stripped_line)
    if(NOT stripped_line STREQUAL "")
        file(APPEND "${tmp_output}" "${line}\n")
    endif()

    file(APPEND "${tmp_output}" "};\n\n")
    string(APPEND resource_entries "{\n")
    string(APPEND resource_entries "    \"${resource_name}\",\n")
    string(APPEND resource_entries "    ${resource_size},\n")
    string(APPEND resource_entries "    ${array_name}\n")
    string(APPEND resource_entries "},\n")

    math(EXPR resource_count "${resource_count} + 1")
endforeach()

file(APPEND "${tmp_output}" "static size_t compiled_spv_resource_count = ${resource_count};\n\n")
file(APPEND "${tmp_output}" "static struct _ab_embedded_resource_t compiled_spv_resources[] = {\n")
file(APPEND "${tmp_output}" "${resource_entries}")
file(APPEND "${tmp_output}" "};\n\n")
file(APPEND "${tmp_output}" "#endif\n")
file(APPEND "${tmp_output}" "#endif\n")

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${tmp_output}" "${OUTPUT}"
    COMMAND_ERROR_IS_FATAL ANY
)
file(REMOVE "${tmp_output}")
