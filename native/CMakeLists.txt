set(NATIVE_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(NATIVE_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(METAL_SRC ${NATIVE_SRC_DIR}/shaders.metal)
set(METAL_AIR ${NATIVE_BUILD_DIR}/quantum_native_runtime.air)
set(METAL_LIB ${NATIVE_BUILD_DIR}/quantum_native_runtime.metallib)

add_custom_command(
    OUTPUT ${METAL_AIR}
    COMMAND xcrun -sdk macosx metal -std=metal3.0 -O3 -c ${METAL_SRC} -o ${METAL_AIR}
    DEPENDS ${METAL_SRC}
    VERBATIM
)

add_custom_command(
    OUTPUT ${METAL_LIB}
    COMMAND xcrun -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
    DEPENDS ${METAL_AIR}
    VERBATIM
)

add_custom_target(quantum_native_metallib ALL DEPENDS ${METAL_LIB})

pybind11_add_module(
    quantum_native_runtime
    MODULE
    ${NATIVE_SRC_DIR}/py_module.cpp
    ${NATIVE_SRC_DIR}/runtime.mm
)

add_dependencies(quantum_native_runtime quantum_native_metallib)

target_compile_definitions(
    quantum_native_runtime
    PRIVATE
    QUANTUM_METALLIB_FILENAME="quantum_native_runtime.metallib"
)

target_link_libraries(
    quantum_native_runtime
    PRIVATE
    "-framework Metal"
    "-framework Foundation"
)

install(TARGETS quantum_native_runtime DESTINATION .)
install(FILES ${METAL_LIB} DESTINATION .)
