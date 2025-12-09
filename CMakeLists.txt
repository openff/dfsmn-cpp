cmake_minimum_required(VERSION 3.16)
project(dfsmn-denoiser C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Compile flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# Static linking option (for portable binaries)
option(STATIC_BUILD "Build with static linking" OFF)
if(STATIC_BUILD)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
endif()

# ============================================================================
# Third-party dependencies
# ============================================================================

# ONNX Runtime
set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/third_party/onnxruntime)
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_DIR}/include)
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_DIR}/lib/libonnxruntime.a)

# KissFFT (required by kaldi-native-fbank)
set(KISSFFT_DIR ${CMAKE_SOURCE_DIR}/third_party/kissfft)
add_library(kissfft 
  ${KISSFFT_DIR}/kiss_fft.c
  ${KISSFFT_DIR}/kiss_fftr.c
)
target_include_directories(kissfft PUBLIC ${KISSFFT_DIR})
target_compile_definitions(kissfft PUBLIC kiss_fft_scalar=float)

# Kaldi Native Fbank
set(KNF_DIR ${CMAKE_SOURCE_DIR}/third_party/kaldi-native-fbank/kaldi-native-fbank/csrc)
add_library(kaldi-native-fbank STATIC
  ${KNF_DIR}/feature-fbank.cc
  ${KNF_DIR}/feature-functions.cc
  ${KNF_DIR}/feature-mfcc.cc
  ${KNF_DIR}/feature-raw-audio-samples.cc
  ${KNF_DIR}/feature-window.cc
  ${KNF_DIR}/kaldi-math.cc
  ${KNF_DIR}/log.cc
  ${KNF_DIR}/mel-computations.cc
  ${KNF_DIR}/online-feature.cc
  ${KNF_DIR}/rfft.cc
  ${KNF_DIR}/stft.cc
  ${KNF_DIR}/istft.cc
  ${KNF_DIR}/whisper-feature.cc
)
target_include_directories(kaldi-native-fbank PUBLIC
  ${CMAKE_SOURCE_DIR}/third_party/kaldi-native-fbank
)
target_link_libraries(kaldi-native-fbank PUBLIC kissfft)

# ============================================================================
# PortAudio (for real-time audio)
# ============================================================================
set(PORTAUDIO_DIR ${CMAKE_SOURCE_DIR}/third_party/stream-modle/portaudio)

# Build PortAudio as static library with minimal dependencies
set(PA_BUILD_SHARED_LIBS OFF CACHE BOOL "Build dynamic library" FORCE)
set(PA_BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
set(PA_BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
set(PA_USE_JACK OFF CACHE BOOL "Disable JACK" FORCE)
set(PA_USE_OSS OFF CACHE BOOL "Disable OSS" FORCE)

# Add PortAudio subdirectory
add_subdirectory(${PORTAUDIO_DIR} ${CMAKE_BINARY_DIR}/portaudio EXCLUDE_FROM_ALL)

# ============================================================================
# Main library
# ============================================================================

add_library(dfsmn-lib STATIC
  ${CMAKE_SOURCE_DIR}/src/dfsmn-denoiser.cc
  ${CMAKE_SOURCE_DIR}/src/wave-reader.cc
  ${CMAKE_SOURCE_DIR}/src/wave-writer.cc
  ${CMAKE_SOURCE_DIR}/src/resample.cc
)

target_include_directories(dfsmn-lib PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/third_party/kaldi-native-fbank
)

target_link_libraries(dfsmn-lib PUBLIC
  kaldi-native-fbank
  ${ONNXRUNTIME_LIB}
  pthread
  dl
)

# ============================================================================
# Stream denoiser library (for real-time processing)
# ============================================================================

add_library(stream-lib STATIC
  ${CMAKE_SOURCE_DIR}/src/stream-denoiser.cc
)

target_include_directories(stream-lib PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/third_party/kaldi-native-fbank
)

target_link_libraries(stream-lib PUBLIC
  kaldi-native-fbank
  ${ONNXRUNTIME_LIB}
  pthread
  dl
)

# ============================================================================
# Executables
# ============================================================================

# File-based denoiser (original)
add_executable(dfsmn-denoiser
  ${CMAKE_SOURCE_DIR}/src/file-main.cc
)
target_link_libraries(dfsmn-denoiser PRIVATE dfsmn-lib)

# Real-time denoiser (new)
add_executable(dfsmn-realtime
  ${CMAKE_SOURCE_DIR}/src/realtime-main.cc
)
target_include_directories(dfsmn-realtime PRIVATE
  ${PORTAUDIO_DIR}/include
)
target_link_libraries(dfsmn-realtime PRIVATE
  stream-lib
  portaudio
)

# ============================================================================
# Install
# ============================================================================

install(TARGETS dfsmn-denoiser dfsmn-realtime DESTINATION bin)
