cmake_minimum_required(VERSION 3.0)
project(orb_slam_fusion)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message("Build type: " ${CMAKE_BUILD_TYPE})

string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
# set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
# For breakpad analysis
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native -g -O2")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -g -O2")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  add_definitions(-DCOMPILEDWITHC11)
  message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
  add_definitions(-DCOMPILEDWITHC0X)
  message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 4 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4 not found.")
   endif()

message("OPENCV VERSION:")
message(${OpenCV_VERSION})

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(realsense2 QUIET)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/include/CameraModels
${PROJECT_SOURCE_DIR}/3rdparty/Sophus
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/../breakpad/src
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib/${BUILD_TYPE})

add_library(${PROJECT_NAME} SHARED
src/system.cc
src/tracking.cc
src/localmapping.cc
src/loopclosing.cc
src/viewer.cc
src/utils/converter.cc
src/utils/map_drawer.cc
src/utils/frame_drawer.cc
src/utils/geometric_tools.cc
src/map/atlas.cc
src/map/map.cc
src/map/frame.cc
src/map/keyframe.cc
src/map/keyframe_database.cc
src/map/mappoint.cc
src/imu/imu_types.cc
src/cam/camera_models/pinhole_model.cc
src/cam/camera_models/kannalabrandt8_model.cc
src/cam/orb_feature/orb_extractor.cc
src/cam/orb_feature/orb_matcher.cc
src/cam/two_view_reconstruction.cc
src/config/config.cc
src/config/settings.cc
src/solver/mlpnp_solver.cc
src/solver/sim3_solver.cc
src/solver/g2o_solver/g2o_types.cc
src/solver/g2o_solver/optimizable_types.cc
src/solver/g2o_solver/optimizer.cc
include/system.h
include/tracking.h
include/localmapping.h
include/loopclosing.h
include/viewer.h
include/utils/converter.h
include/utils/map_drawer.h
include/utils/frame_drawer.h
include/utils/geometric_tools.h
include/utils/serialization_utils.h
include/map/atlas.h
include/map/map.h
include/map/frame.h
include/map/keyframe.h
include/map/keyframe_database.h
include/map/mappoint.h
include/imu/imu_types.h
include/cam/camera_models/geometric_camera.h
include/cam/camera_models/pinhole_model.h
include/cam/camera_models/kannalabrandt8_model.h
include/cam/orb_feature/orb_extractor.h
include/cam/orb_feature/orb_matcher.h
include/cam/two_view_reconstruction.h
include/config/config.h
include/config/settings.h
include/solver/mlpnp_solver.h
include/solver/sim3_solver.h
include/solver/g2o_solver/g2o_types.h
include/solver/g2o_solver/optimizable_types.h
include/solver/g2o_solver/optimizer.h
)

add_subdirectory(3rdparty/g2o)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/lib/${BUILD_TYPE}/libDBoW2.so
${PROJECT_SOURCE_DIR}/lib/${BUILD_TYPE}/libg2o.so
-lboost_serialization
-lcrypto
-lpthread
# brakpad for stack analysis
${PROJECT_SOURCE_DIR}/lib/libbreakpad_client.a
)

# If RealSense SDK is found the library is added
if(realsense2_FOUND)
  include_directories(${PROJECT_NAME}
  ${realsense_INCLUDE_DIR}
  )
  target_link_libraries(${PROJECT_NAME}
  ${realsense2_LIBRARY}
  )
endif()

# Build executions

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Tests
add_executable(slam_euroc_si
tests/slam_euroc_si.cc)
target_link_libraries(slam_euroc_si ${PROJECT_NAME})

# Tools
add_executable(orbvoc_binary_convertor 
tools/orbvoc_binary_convertor.cc)
target_link_libraries(orbvoc_binary_convertor ${PROJECT_NAME})
