# Wrapper so ggml builds as a subproject (its standalone mode wants a ggml.pc.in that
# the llama.cpp tree doesn't ship).
cmake_minimum_required(VERSION 3.14)
project(hybrid_cpu_backend C CXX)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp/ggml ggml)

# clang-cl's /arch:AVX2 enables AVX2/FMA/F16C/BMI2 but not AVX-VNNI, which the alderlake
# variant's code uses. Add it to that variant only, so the older variants stay safe on older CPUs.
if (TARGET ggml-cpu-alderlake)
    target_compile_options(ggml-cpu-alderlake PRIVATE -mavxvnni)
endif()
