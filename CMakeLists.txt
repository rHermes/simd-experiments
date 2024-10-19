cmake_minimum_required(VERSION 3.25)

cmake_policy(SET CMP0069 NEW)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(
        rimd
        VERSION 0.1
        DESCRIPTION "A simd library with experiemnts"
        LANGUAGES CXX)


include(CheckCXXCompilerFlag)
include(CheckIPOSupported)

check_ipo_supported(RESULT result)
if(result)
    if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        # We want link time optimization and whole program optimization
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endif()

if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_CXX_STANDARD 23)
    set(CMAKE_CXX_EXTENSIONS OFF)

    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    CHECK_CXX_COMPILER_FLAG("-march=x86-64-v3" COMPILER_SUPPORTS_MARCH)
    if (COMPILER_SUPPORTS_MARCH)
        add_compile_options("-march=x86-64-v3")
    endif ()

    CHECK_CXX_COMPILER_FLAG("-mtune=native" COMPILER_SUPPORTS_MTUNE)
    if (COMPILER_SUPPORTS_MTUNE)
        add_compile_options("-mtune=native")
    endif ()

    # add_compile_definitions("-D_GLIBCXX_DEBUG")

    #CHECK_CXX_COMPILER_FLAG("-ftime-trace" COMPILER_SUPPORTS_TIME_TRACE)
    #if(COMPILER_SUPPORTS_TIME_TRACE)
    #    add_compile_options("-ftime-trace")
    #endif()

    # add_compile_options("-fprofile-instr-generate" "-fcoverage-mapping")
    # add_link_options("-fprofile-instr-generate" "-fcoverage-mapping")

    include(CTest)
endif ()

include(FetchContent)



add_subdirectory(src)
add_subdirectory(examples)
