FetchContent_Declare(
        nanobench
        GIT_REPOSITORY https://github.com/martinus/nanobench.git
        GIT_TAG v4.3.11
        GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(nanobench)

function(addProg)
    set(options NANOBENCH)
    set(oneValueArgs NAME)
    cmake_parse_arguments(PARSE_ARGV 0 arg "${options}" "${oneValueArgs}" "")

    add_executable(${arg_NAME} "${arg_NAME}.cpp")
    if(arg_NANOBENCH)
        target_sources(${arg_NAME} PRIVATE nanobench.cpp)
        target_link_libraries(${arg_NAME} PRIVATE nanobench)
    endif()

    target_link_libraries(${arg_NAME} PUBLIC rimd)

    set_target_properties(${arg_NAME} PROPERTIES CXX_EXTENSIONS OFF)

    if (MSVC)
        target_compile_definitions(${arg_NAME} PRIVATE UNICODE _UNICODE _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS NOMINMAX)

        # These options make msvc behave normal
        # Add all those nice errors
        target_compile_options(${arg_NAME} PRIVATE /W4)

        target_compile_options("${arg_NAME}" PRIVATE /w34062) # Enum switch detection
        target_compile_options("${arg_NAME}" PRIVATE /w14548) # Expression before comma has no effect
        target_compile_options(${arg_NAME} PRIVATE /w14555) # Expression has no effect

        # Enable behavior that should be default, but microsoft have to support everthing under the sun
        target_compile_options(${arg_NAME} PRIVATE /Zc:__cplusplus)
        target_compile_options(${arg_NAME} PRIVATE /Zc:preprocessor)
        target_compile_options(${arg_NAME} PRIVATE /Zc:lambda)
        target_compile_options(${arg_NAME} PRIVATE /Zc:inline)
        target_compile_options(${arg_NAME} PRIVATE /Zc:throwingNew)

        # Disable annoying warnings
        add_compile_options(/wd4244)
        add_compile_options(/wd4267)

    else()
        target_compile_options(${arg_NAME} PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(${arg_NAME} PRIVATE -Wno-overflow)
        # I don't know what this error is, so I'm disabling it for now.
        target_compile_options(${arg_NAME} PRIVATE -Wno-ignored-attributes)
    endif()
endfunction()

addProg(NAME "leetcode-921" NANOBENCH)
addProg(NAME "leetcode-2696")
addProg(NAME "leetcode-2938" NANOBENCH)