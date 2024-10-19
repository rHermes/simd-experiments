set(CORE_HEADER_LIST
        "${rimd_SOURCE_DIR}/include/rimd/all.hpp"
        "${rimd_SOURCE_DIR}/include/rimd/core.hpp"
        "${rimd_SOURCE_DIR}/include/rimd/utils.hpp"
        "${rimd_SOURCE_DIR}/include/rimd/details/sse.hpp"

)

add_library(rimd INTERFACE)
target_sources(rimd INTERFACE ${CORE_HEADER_LIST})
target_include_directories(rimd INTERFACE "../include")