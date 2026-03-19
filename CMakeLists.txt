cmake_minimum_required(VERSION 3.28)
project(psw C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Default to Release for single-config generators (Ninja/Unix Makefiles).
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(PSW_ENABLE_NATIVE "Enable -march=native for optimized builds" ON)
option(PSW_ENABLE_LTO "Enable link-time optimization for optimized builds" ON)

add_executable(psw
    main.c
    psw_gg.c
    psw_gg2.c
    psw_gg3.c
    psw_gg3_sse.c
)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(psw PRIVATE
            -Wall
            -Wextra
            -Wc++-compat
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:Release>:-DNDEBUG>
            $<$<CONFIG:RelWithDebInfo>:-O3>
            $<$<CONFIG:RelWithDebInfo>:-g>
            $<$<CONFIG:RelWithDebInfo>:-DNDEBUG>
            $<$<AND:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>,$<BOOL:${PSW_ENABLE_NATIVE}>>:-march=native>
            $<$<AND:$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},x86>,$<NOT:$<BOOL:${PSW_ENABLE_NATIVE}>>>:-msse2>
    )

endif()

if(PSW_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_ok OUTPUT ipo_err)
    if(ipo_ok)
        set_property(TARGET psw PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
        set_property(TARGET psw PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
    else()
        message(STATUS "IPO/LTO not enabled: ${ipo_err}")
    endif()
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "PSW_ENABLE_NATIVE=${PSW_ENABLE_NATIVE}, PSW_ENABLE_LTO=${PSW_ENABLE_LTO}")
