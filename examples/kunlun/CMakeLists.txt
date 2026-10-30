add_executable(kunlun-gemv-bench gemv-bench.cpp)
target_include_directories(kunlun-gemv-bench PRIVATE ../../ggml/include ../../ggml/src)
target_link_libraries(kunlun-gemv-bench PRIVATE ggml ggml-base)
add_executable(kunlun-llm-smoke llm-smoke.cpp)
target_link_libraries(kunlun-llm-smoke PRIVATE llama ggml ggml-base ${CMAKE_DL_LIBS})
add_executable(kunlun-backend-contract backend-contract.cpp)
target_link_libraries(kunlun-backend-contract PRIVATE ggml ggml-base)

add_executable(kunlun-quality-eval quality-eval.cpp)
target_link_libraries(kunlun-quality-eval PRIVATE llama ggml ggml-base)

add_executable(kunlun-swiglu-test swiglu-test.cpp)
target_link_libraries(kunlun-swiglu-test PRIVATE ggml ggml-base)

add_executable(kunlun-attention-test attention-test.cpp)
target_link_libraries(kunlun-attention-test PRIVATE llama ggml)

add_executable(kunlun-flash-decoding-bench flash-decoding-bench.cpp)
target_link_libraries(kunlun-flash-decoding-bench PRIVATE llama ggml ggml-base ${CMAKE_DL_LIBS})

add_executable(kunlun-kv-max-bench kv-max-bench.cpp)
target_link_libraries(kunlun-kv-max-bench PRIVATE llama ggml ggml-base ${CMAKE_DL_LIBS})

add_executable(kunlun-activation-share-test activation-share-test.cpp)
target_link_libraries(kunlun-activation-share-test PRIVATE ggml ggml-base)

# Standalone numerical diagnostic, never used in normal inference.
if(NOT TARGET KunlunK200::kunlunk200)
    find_package(KunlunK200 0.7.12 CONFIG REQUIRED)
endif()
add_executable(kunlun-swiglu-numerics swiglu-numerics.cpp)
target_link_libraries(kunlun-swiglu-numerics PRIVATE ggml ggml-base KunlunK200::kunlunk200)
