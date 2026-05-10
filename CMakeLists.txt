cmake_minimum_required(VERSION 3.15)
set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)


project(turbogator_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 REQUIRED)
find_package(Python COMPONENTS Interpreter Development)

file(GLOB TURBOGATOR_BACKEND_SOURCES "${CMAKE_SOURCE_DIR}/csrc/backends/*.cpp")

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import os, torch; print(os.path.join(torch.__path__[0], 'lib'))"
    OUTPUT_VARIABLE TORCH_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(Torch REQUIRED)

find_library(
    TORCH_PYTHON_LIBRARY
    NAMES torch_python
    PATHS "${TORCH_LIB_DIR}"
    NO_DEFAULT_PATH
)

pybind11_add_module(turbogator_ext
    "${CMAKE_SOURCE_DIR}/csrc/bindings.cpp"
    ${TURBOGATOR_BACKEND_SOURCES}
)

target_compile_options(turbogator_ext PRIVATE
    -Wall
    -Wextra
    -march=native
    -g0
)

target_include_directories(turbogator_ext PRIVATE 
    "${CMAKE_SOURCE_DIR}/csrc"
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(turbogator_ext PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(turbogator_ext PRIVATE ${TORCH_PYTHON_LIBRARY})
target_link_options(turbogator_ext PRIVATE "-Wl,--no-as-needed")

# target_link_options(turbogator_ext PRIVATE "-fuse-ld=lld")

set_target_properties(turbogator_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/turbogator"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/turbogator"
)

install(TARGETS turbogator_ext
    LIBRARY DESTINATION "turbogator"
    RUNTIME DESTINATION "turbogator"
)