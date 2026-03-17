cmake_minimum_required(VERSION 3.28)
project(psw C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(psw
    main.c
    psw_gg.c
)
