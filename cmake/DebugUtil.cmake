cmake_minimum_required(VERSION 3.8)

macro(PRINT_VAR var)
    message(STATUS "${var}=${${var}}")
endmacro()