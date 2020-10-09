cmake_minimum_required(VERSION 3.5.1)

add_library(
  decoder-library
  INTERFACE
  )

target_sources(
  decoder-library
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/LexiconDecoder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/LexiconFreeDecoder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/LexiconSeq2SeqDecoder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/LexiconFreeSeq2SeqDecoder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/BatchedLexiconFreeSeq2SeqDecoder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Trie.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp
  )

target_link_libraries(
  decoder-library
  INTERFACE
  lm-library
  )

if (W2L_LIBRARIES_USE_CUDA)
  include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
  )

  cuda_add_library(
    w2l-decoder-library-cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/cuda/BatchedLexiconFreeSeq2SeqDecoder.cu
  )
  
  target_link_libraries(
    decoder-library
    INTERFACE
    ${CUDA_LIBRARIES}
    w2l-decoder-library-cuda
    flashlight::flashlight
  )

  target_include_directories(
    decoder-library
    INTERFACE
    ${CUDA_INCLUDE_DIRS}
    ${cereal_INCLUDE_DIRS}
  )
  
else()
  include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
  )

  add_library(
    w2l-decoder-library-cpu
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/BatchedLexiconFreeSeq2SeqDecoder.cpp
  )

  target_link_libraries(
    decoder-library
    INTERFACE
    w2l-decoder-library-cpu
    flashlight::flashlight
  )

  target_include_directories(
    decoder-library
    INTERFACE
    ${cereal_INCLUDE_DIRS}
  )  
endif ()
