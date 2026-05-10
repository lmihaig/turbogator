cmake_minimum_required(VERSION 3.15)
project(turbogator_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 REQUIRED)
find_package(Python COMPONENTS Interpreter Development)

file(GLOB CONFIGURE_DEPENDS TURBOGATOR_BACKEND_SOURCES "${CMAKE_SOURCE_DIR}/csrc/backends/*.cpp")
set(BINDINGS_SRC "${CMAKE_SOURCE_DIR}/csrc/bindings.cpp")

pybind11_add_module(turbogator_ext ${BINDINGS_SRC} ${TURBOGATOR_BACKEND_SOURCES})

target_include_directories(turbogator_ext PRIVATE ${CMAKE_SOURCE_DIR}/csrc)

if (UNIX)
	set_target_properties(turbogator_ext PROPERTIES PREFIX "")
endif()

set_target_properties(turbogator_ext PROPERTIES
	LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/turbogator"
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/turbogator"
)

install(TARGETS turbogator_ext
	LIBRARY DESTINATION "turbogator"
	RUNTIME DESTINATION "turbogator"
)
