cmake_minimum_required(VERSION 3.16)
project(tool_aimbot_sim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(glfw3 REQUIRED)

set(MUJOCO_FRAMEWORK_DIR "/Library/Frameworks/mujoco.framework" CACHE PATH "MuJoCo macOS framework")
set(AIMBOT_SOFTWARE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../27_aimbot_software" CACHE PATH "Sibling 27_aimbot_software checkout")

set(AIMBOT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/aimbot_io")
if(EXISTS "${AIMBOT_SOFTWARE_DIR}/io/command.hpp")
    message(STATUS "Using aimbot command header from ${AIMBOT_SOFTWARE_DIR}")
    set(AIMBOT_INCLUDE_DIR "${AIMBOT_SOFTWARE_DIR}")
else()
    message(STATUS "Using fallback aimbot command header")
endif()

function(configure_mujoco_target target)
    target_include_directories(${target} PRIVATE
        "${MUJOCO_FRAMEWORK_DIR}/Headers"
        "${AIMBOT_INCLUDE_DIR}"
    )
    target_link_libraries(${target}
        "-framework mujoco"
        "-framework OpenGL"
        "-framework Cocoa"
        "-framework IOKit"
        glfw
    )
    set_target_properties(${target} PROPERTIES
        LINK_FLAGS "-F/Library/Frameworks"
        BUILD_RPATH "/Library/Frameworks"
    )
endfunction()

add_executable(sim main.cpp src/aim_predictor.cpp src/aim_predictor_intercept.cpp src/aim_predictor_ppo.cpp)
configure_mujoco_target(sim)

# ONNX Runtime (Homebrew, optional — PPO approach loads model at runtime).
find_library(ONNXRUNTIME_LIB onnxruntime HINTS /opt/homebrew/lib /usr/local/lib)
find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h
          HINTS /opt/homebrew/include/onnxruntime /usr/local/include/onnxruntime)
if(ONNXRUNTIME_LIB AND ONNXRUNTIME_INCLUDE)
    message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_LIB}")
    target_include_directories(sim PRIVATE "${ONNXRUNTIME_INCLUDE}")
    target_link_libraries(sim "${ONNXRUNTIME_LIB}")
    set_target_properties(sim PROPERTIES
        BUILD_RPATH "/Library/Frameworks;/opt/homebrew/lib"
        INSTALL_RPATH "/Library/Frameworks;/opt/homebrew/lib"
    )
else()
    message(WARNING "ONNX Runtime not found — PPO approach will be unavailable")
endif()

add_executable(headless_check test_headless.cpp)
configure_mujoco_target(headless_check)
