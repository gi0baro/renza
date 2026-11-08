# Run third_party/fetch-llama.sh, then configure with: cmake -S . -B build -G Ninja
# The default build stays C-only; ctest builds the selected CPU tests via a fixture.
# Build the engine explicitly with: cmake --build build --target renza-engine
# and the server on it with: cmake --build build --target renza
cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CUDAToolkit_ROOT /opt/cuda)
project(renza LANGUAGES C CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

enable_testing()
set(STRATA_ENABLE_CUDA ON)
set(STRATA_Q6K_EXPERTS ON)
set(STRATA_MMQ_KQUANTS OFF)
set(STRATA_GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp")
if(NOT EXISTS "${STRATA_GGML_DIR}/ggml/CMakeLists.txt")
    message(FATAL_ERROR "Run third_party/fetch-llama.sh before configuring (no network FetchContent).")
endif()
option(RENZA_BUILD_STRATA_TESTS "Register Strata's CPU-only tests (built on demand by ctest)" ON)
set(STRATA_BUILD_TESTS ${RENZA_BUILD_STRATA_TESTS})
set(STRATA_BUILD_CONVERSATION_TESTS ${RENZA_BUILD_STRATA_TESTS})

# Child directories snapshot the original C flags here, before ds4's overrides below.
add_subdirectory(engine/strata engine/strata EXCLUDE_FROM_ALL)

# Strata's CPU tests renza runs; their executables are EXCLUDE_FROM_ALL: a setup test builds them first.
set(strata_cpu_tests
    conv_cache_test draft_kv_plan_test
    suffix_drafter_test controller_test draft_policy_test
    coupled_draft_test spec_prob_test conversation_cache_test
    conversation_memory_test conversation_file_test session_save_reclaim_test
    conversation_split_failure_test nv_defaults_test window_evict_test
    adapt_decay_test ple_residency_test
    auto_cache_test pcie_share_rule_test adapt_async_default_test)
if(RENZA_BUILD_STRATA_TESTS)
    set_tests_properties(${strata_cpu_tests} DIRECTORY engine/strata
        PROPERTIES FIXTURES_REQUIRED strata_cpu_tests TIMEOUT 120)
    add_custom_target(renza-strata-tests DEPENDS ${strata_cpu_tests})
    add_test(NAME strata-cpu-tests-build
        COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}" --target renza-strata-tests)
    set_tests_properties(strata-cpu-tests-build PROPERTIES FIXTURES_SETUP strata_cpu_tests TIMEOUT 600)
endif()

# ds4's Makefile flags on Linux; -O3 and -g are set here, so the build-type
# defaults are cleared to keep them from adding -DNDEBUG or another -O level.
set(CMAKE_C_FLAGS_RELEASE "")
set(CMAKE_C_FLAGS_DEBUG "")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "")
set(CMAKE_C_FLAGS_MINSIZEREL "")
set(RENZA_C_FLAGS
    -std=c99 -D_GNU_SOURCE -O3 -ffast-math -fno-finite-math-only -g
    -march=native -Wall -Wextra)

add_library(renza_text STATIC text/gguf.c text/tokenizer.c)
target_include_directories(renza_text PUBLIC text/include ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(renza_text PRIVATE ${RENZA_C_FLAGS})

add_executable(renza-tokenize text/renza-tokenize.c)
target_compile_options(renza-tokenize PRIVATE ${RENZA_C_FLAGS})
target_link_libraries(renza-tokenize PRIVATE renza_text)

# Image decode and preprocessing (C, default build); the encoder (CUDA) builds by name:
# cmake --build build --target renza-vision
add_library(renza_vision_image STATIC vision/image.c)
target_include_directories(renza_vision_image PUBLIC vision)
target_compile_options(renza_vision_image PRIVATE ${RENZA_C_FLAGS})
find_package(Threads REQUIRED)
target_link_libraries(renza_vision_image PUBLIC m Threads::Threads)

find_package(CUDAToolkit REQUIRED)
add_library(renza_vision STATIC EXCLUDE_FROM_ALL vision/encoder.c vision/encoder_cuda.cu)
target_include_directories(renza_vision PUBLIC vision/include)
# ds4's nvcc flags for its CUDA objects.
target_compile_options(renza_vision PRIVATE
    "$<$<COMPILE_LANGUAGE:C>:${RENZA_C_FLAGS}>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--use_fast_math;-Xcompiler=-Wall,-Wextra>")
target_link_libraries(renza_vision PUBLIC renza_vision_image renza_text CUDA::cudart CUDA::cublas)

add_executable(renza-vision EXCLUDE_FROM_ALL vision/renza-vision.c)
target_compile_options(renza-vision PRIVATE ${RENZA_C_FLAGS})
target_link_libraries(renza-vision PRIVATE renza_vision)

add_executable(renza-vision-image-tests vision/test/image_test.c)
target_compile_options(renza-vision-image-tests PRIVATE ${RENZA_C_FLAGS})
target_link_libraries(renza-vision-image-tests PRIVATE renza_vision_image ${CMAKE_DL_LIBS})
add_test(NAME renza-vision-image-tests
    COMMAND renza-vision-image-tests "${CMAKE_CURRENT_SOURCE_DIR}/vision/test/rotated.jpg"
        "${CMAKE_CURRENT_SOURCE_DIR}/vision/test/prog_sos_se_oob.jpg")

set(RENZA_DS4_ROOT "$ENV{HOME}/Downloads/code/ds4" CACHE PATH "Read-only ds4 tree (preprocessing oracle)")
list(JOIN RENZA_C_FLAGS " " renza_c_flags_string)
add_test(NAME renza-vision-ds4-parity
    COMMAND ${CMAKE_COMMAND}
        "-DDS4_DIR=${RENZA_DS4_ROOT}"
        "-DTEST_BINARY=$<TARGET_FILE:renza-vision-image-tests>"
        "-DC_COMPILER=${CMAKE_C_COMPILER}"
        "-DC_FLAGS=${renza_c_flags_string}"
        "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}"
        "-DIMAGES=${CMAKE_CURRENT_SOURCE_DIR}/tests/baseline/images/shapes.png=70:10:7|${RENZA_DS4_ROOT}/tests/vision-fixtures/qwen38/orbit.png|${RENZA_DS4_ROOT}/tests/vision-fixtures/qwen38/maple.png|${CMAKE_CURRENT_SOURCE_DIR}/vision/test/rotated.jpg|${RENZA_DS4_ROOT}/tests/vision-fixtures/glm53/screenshot.png"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/vision/check_ds4_parity.cmake")
set_tests_properties(renza-vision-ds4-parity PROPERTIES SKIP_REGULAR_EXPRESSION "SKIP:")

add_library(renza_engine STATIC engine/stub/stub.c engine/renza_common.c)
target_include_directories(renza_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/engine)
target_compile_options(renza_engine PRIVATE ${RENZA_C_FLAGS})
target_link_libraries(renza_engine PUBLIC renza_text)

add_library(renza_server STATIC
    server/ds4_ssd.c
    server/ds4_gpu_args.c
    server/ds4_help.c
    server/ds4_kvstore.c
    server/rax.c
    engine/stub/ds4_shim.c)
# Over renza.h only: each executable links one engine, the stub or Strata's.
target_include_directories(renza_server PUBLIC server ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(renza_server PRIVATE ${RENZA_C_FLAGS})
target_link_libraries(renza_server PUBLIC m pthread)

add_executable(renza-server-tests server/ds4_server.c)
target_compile_definitions(renza-server-tests PRIVATE DS4_SERVER_TEST)
target_compile_options(renza-server-tests PRIVATE ${RENZA_C_FLAGS} -Wno-unused-function)
target_link_libraries(renza-server-tests PRIVATE renza_server renza_engine)

# The shim's generation over a fake decode defined by the test (no stub engine).
add_executable(renza-shim-tests engine/stub/shim_test.c engine/stub/ds4_shim.c engine/renza_common.c)
target_include_directories(renza-shim-tests PRIVATE server ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/engine)
target_compile_options(renza-shim-tests PRIVATE ${RENZA_C_FLAGS})
target_link_libraries(renza-shim-tests PRIVATE renza_text)

add_executable(renza-stub-tests engine/stub/stub_test.c)
target_compile_options(renza-stub-tests PRIVATE ${RENZA_C_FLAGS})
target_link_libraries(renza-stub-tests PRIVATE renza_engine)

add_executable(renza-text-tests engine/stub/text_test.c)
target_compile_options(renza-text-tests PRIVATE ${RENZA_C_FLAGS})
target_link_libraries(renza-text-tests PRIVATE renza_engine)

add_test(NAME renza-server-tests COMMAND renza-server-tests)
add_test(NAME renza-stub-tests COMMAND renza-stub-tests)
add_test(NAME renza-text-tests COMMAND renza-text-tests)
add_test(NAME renza-shim-tests COMMAND renza-shim-tests)

# renza.h over Strata's engine (CUDA): not in the default build; the targets build by name.
add_library(renza_engine_strata STATIC EXCLUDE_FROM_ALL engine/strata_adapter.cpp engine/renza_common.c)
target_include_directories(renza_engine_strata PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/engine)
target_compile_features(renza_engine_strata PRIVATE cxx_std_20)
target_compile_options(renza_engine_strata PRIVATE
    "$<$<COMPILE_LANGUAGE:C>:${RENZA_C_FLAGS}>" $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
target_link_libraries(renza_engine_strata PUBLIC strata_serve renza_text renza_vision)

add_executable(renza-engine EXCLUDE_FROM_ALL engine/serve/serve_main.c)
target_compile_options(renza-engine PRIVATE ${RENZA_C_FLAGS})
target_link_libraries(renza-engine PRIVATE renza_engine_strata pthread)

add_executable(renza EXCLUDE_FROM_ALL server/ds4_server.c)
target_compile_options(renza PRIVATE ${RENZA_C_FLAGS})
target_link_libraries(renza PRIVATE renza_server renza_engine_strata)

add_executable(renza-strata-open-test EXCLUDE_FROM_ALL engine/strata_open_test.c)
target_compile_options(renza-strata-open-test PRIVATE ${RENZA_C_FLAGS})
target_link_libraries(renza-strata-open-test PRIVATE renza_engine_strata)
add_test(NAME renza-strata-open-test-build
    COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}" --target renza-strata-open-test)
set_tests_properties(renza-strata-open-test-build PROPERTIES FIXTURES_SETUP renza_strata_open TIMEOUT 1800)
add_test(NAME renza-strata-open-test COMMAND renza-strata-open-test)
set_tests_properties(renza-strata-open-test PROPERTIES FIXTURES_REQUIRED renza_strata_open)

set(RENZA_TOKENIZER_GGUF "/data/ML/models-renza/swift-IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
    CACHE FILEPATH "GGUF used by the tokenizer oracle check")
set(RENZA_STRATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/upstream/strata-0.1.45" CACHE PATH "Read-only Strata oracle tree")
set(RENZA_STRATA_TOKENIZER "${CMAKE_CURRENT_SOURCE_DIR}/data/qwen/pack/tokenizer"
    CACHE PATH "Strata tokenizer tables")
set(RENZA_STRATA_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/tests/.venv/bin/python"
    CACHE FILEPATH "Python interpreter for the Strata oracle")
add_test(NAME renza-tokenizer-check
    COMMAND ${CMAKE_COMMAND}
        "-DTOKENIZER_BINARY=$<TARGET_FILE:renza-tokenize>"
        "-DGGUF=${RENZA_TOKENIZER_GGUF}"
        "-DSTRATA_ROOT=${RENZA_STRATA_ROOT}"
        "-DSTRATA_PYTHON=${RENZA_STRATA_PYTHON}"
        "-DTOKENIZER_DIR=${RENZA_STRATA_TOKENIZER}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/text/check_tokenizer.cmake")
set_tests_properties(renza-tokenizer-check PROPERTIES SKIP_REGULAR_EXPRESSION "SKIP:")

# The adapter over a fake of Strata's engine API (no CUDA); it reads the tokenizer from the native GGUF.
add_executable(renza-adapter-tests engine/strata_adapter_test.cpp engine/strata_adapter.cpp engine/renza_common.c)
target_include_directories(renza-adapter-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/engine
    ${CMAKE_CURRENT_SOURCE_DIR}/engine/strata/include ${CMAKE_CURRENT_SOURCE_DIR}/vision/include)
target_compile_features(renza-adapter-tests PRIVATE cxx_std_20)
target_compile_options(renza-adapter-tests PRIVATE
    "$<$<COMPILE_LANGUAGE:C>:${RENZA_C_FLAGS}>" $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
target_link_libraries(renza-adapter-tests PRIVATE renza_text)
target_link_options(renza-adapter-tests PRIVATE
    -Wl,--wrap=renza_engine_text_open -Wl,--wrap=renza_engine_text_close)
add_test(NAME renza-adapter-tests COMMAND renza-adapter-tests "${RENZA_TOKENIZER_GGUF}")
set_tests_properties(renza-adapter-tests PROPERTIES SKIP_RETURN_CODE 77)

# ds4's KV store over the shim and the stub engine's saved sessions; it reads the tokenizer from the native GGUF.
add_executable(renza-kvstore-tests engine/stub/kvstore_test.c)
target_compile_options(renza-kvstore-tests PRIVATE ${RENZA_C_FLAGS})
target_link_libraries(renza-kvstore-tests PRIVATE renza_server renza_engine)
add_test(NAME renza-kvstore-tests COMMAND renza-kvstore-tests "${RENZA_TOKENIZER_GGUF}")
set_tests_properties(renza-kvstore-tests PROPERTIES SKIP_RETURN_CODE 77)
