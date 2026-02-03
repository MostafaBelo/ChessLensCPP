cmake_minimum_required(VERSION 3.15)
project(ChessLens VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(USE_HTTP "Enable HTTP FEN updates via libcurl" ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(CURL)

# ONNX Runtime
# Adjust the path to your ONNX Runtime installation
set(ONNXRUNTIME_INCLUDE_DIR "/opt/onnxruntime-linux-x64-1.23.2/include" CACHE PATH "ONNX Runtime include directory")
set(ONNXRUNTIME_LIB_DIR "/opt/onnxruntime-linux-x64-1.23.2/lib" CACHE PATH "ONNX Runtime library directory")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

# Link directories
link_directories(
    ${ONNXRUNTIME_LIB_DIR}
)

# ============================================================================
# Utils Library
# ============================================================================
add_library(chess_utils STATIC
    Utils/ChessUtils.cpp
    Utils/Utils.cpp
)

target_include_directories(chess_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(chess_utils
    ${OpenCV_LIBS}
)

# ============================================================================
# Chess Game State Library (your existing C++ implementation)
# ============================================================================
add_library(chess_game_state STATIC
    ChessGameState.cpp
)

target_include_directories(chess_game_state PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# ============================================================================
# Chess HMM Library (your existing C++ implementation)
# ============================================================================
add_library(chess_hmm STATIC
    ChessHMM.cpp
)

target_include_directories(chess_hmm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(chess_hmm
    chess_game_state
    chess_utils
)

# ============================================================================
# Context Aware Models Library
# ============================================================================
add_library(context_models STATIC
    ContextAwareModels/HMM.cpp
)

target_include_directories(context_models PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(context_models
    chess_utils
    chess_hmm
)

# ============================================================================
# Computer Vision Modules
# ============================================================================

# Image Provider
add_library(image_provider STATIC
    ImageProvider.cpp
    LibCameraCapture.cpp
)
target_link_libraries(image_provider ${OpenCV_LIBS})

# Board Detection
add_library(board_detection STATIC
    BoardDetection.cpp
    BoardSaddle.cpp
)
target_include_directories(board_detection PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(board_detection ${OpenCV_LIBS})

# Wakeup Module
add_library(wakeup_module STATIC
    WakeupModule.cpp
)
target_link_libraries(wakeup_module ${OpenCV_LIBS})

# Occlusion Detector
add_library(occlusion_detector STATIC
    OcclusionDetector.cpp
)
target_link_libraries(occlusion_detector 
    ${OpenCV_LIBS}
    onnxruntime
)

# Piece Detector
add_library(piece_detector STATIC
    PieceCropper.cpp
    PieceDetectionCNN.cpp
    PieceDetection.cpp
)
target_include_directories(piece_detector PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(piece_detector
    ${OpenCV_LIBS}
    onnxruntime
)

# ============================================================================
# ChessLens Main Library
# ============================================================================
add_library(chesslens STATIC
    ChessLens.cpp
)

target_include_directories(chesslens PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(chesslens
    chess_utils
    context_models
    image_provider
    board_detection
    wakeup_module
    occlusion_detector
    piece_detector
    ${OpenCV_LIBS}
    onnxruntime
)

# ============================================================================
# Main Executable
# ============================================================================
add_executable(chesslens_main
    main.cpp
)

target_link_libraries(chesslens_main
    chesslens
)

# Add HTTP support if enabled
if(USE_HTTP AND CURL_FOUND)
    target_compile_definitions(chesslens_main PRIVATE USE_HTTP)
    target_link_libraries(chesslens_main ${CURL_LIBRARIES})
    target_include_directories(chesslens_main PRIVATE ${CURL_INCLUDE_DIRS})
endif()

# ============================================================================
# Installation
# ============================================================================
install(TARGETS chesslens_main
    RUNTIME DESTINATION bin
)

install(TARGETS chesslens chess_utils context_models
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION include/chesslens
    FILES_MATCHING PATTERN "*.h"
)

# ============================================================================
# Print configuration
# ============================================================================
message(STATUS "")
message(STATUS "ChessLens Configuration:")
message(STATUS "  C++ Standard:        ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenCV Version:      ${OpenCV_VERSION}")
message(STATUS "  OpenCV Libs:         ${OpenCV_LIBS}")
message(STATUS "  ONNX Runtime Include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "  ONNX Runtime Lib:    ${ONNXRUNTIME_LIB_DIR}")
message(STATUS "  HTTP Support:        ${USE_HTTP}")
if(USE_HTTP AND CURL_FOUND)
message(STATUS "  CURL Version:        ${CURL_VERSION_STRING}")
endif()
message(STATUS "")