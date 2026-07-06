# Tensorflow Lite library

set(TFLITE_VERSION tensorflow-build-v2.17.0--bazel--x86-64)
set(TFLITE_ROOT ${PROJECT_SOURCE_DIR}/third_party/${TFLITE_VERSION})

add_library(tflite SHARED IMPORTED)
set_target_properties(tflite PROPERTIES
    IMPORTED_LOCATION "${TFLITE_ROOT}/lib/libtensorflowlite.so"
)

target_include_directories(tflite
    INTERFACE
        ${TFLITE_ROOT}/include
)
