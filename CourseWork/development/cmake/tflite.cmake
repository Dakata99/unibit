# Tensorflow Lite library

# TODO: properly build TFLite source and get only the needed stuff
# set(TFLITE_VERSION v2.20.0)
set(TFLITE_ROOT ${PROJECT_SOURCE_DIR}/third_party/${TFLITE_VERSION})

add_library(tflite SHARED IMPORTED)
set_target_properties(tflite PROPERTIES
    IMPORTED_LOCATION "${TFLITE_ROOT}/libtensorflowlite.so"
)

target_include_directories(tflite
    INTERFACE
        ${TFLITE_ROOT}
        ${TFLITE_ROOT}/flatbuffers/include
)
