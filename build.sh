#!/bin/bash

build_mode=$1
build_mode=${build_mode,,}
arch_type=$2

mkdir -p lib/debug/
mkdir -p lib/release/

function build_orb_slam_fusion() {
  echo "Building orb_slam_fusion in ${build_mode} mode..."

  echo "Configuring and building 3rdparty/DBoW2..."
  cd 3rdparty/DBoW2
  mkdir build_${build_mode}
  cd build_${build_mode}
  cmake .. -DCMAKE_BUILD_TYPE=${build_mode}
  make -j${nproc}
  echo "Done!"

  echo "Configuring and building 3rdparty/g2o..."
  cd ../../g2o
  mkdir build_${build_mode}
  cd build_${build_mode}
  cmake .. -DCMAKE_BUILD_TYPE=${build_mode}
  make -j${nproc}
  echo "Done!"

  echo "Copying 3rdparty libs to lib directory..."
  cd ../../../
  cp 3rdparty/DBoW2/lib/${build_mode}/libDBoW2.so lib/${build_mode}
  cp 3rdparty/g2o/lib/${build_mode}/libg2o.so lib/${build_mode}
  echo "Done!"

  echo "Configuring and building orb_slam_fusion..."
  mkdir build_${build_mode}
  cd build_${build_mode}
  if [ ${arch_type} == "x86" ]; then
    cmake .. -DCMAKE_BUILD_TYPE=${build_mode}
  elif [ ${arch_type} == "aarch64" ]; then
  #TODO: Add aarch64 build command.
    cmake .. -DCMAKE_BUILD_TYPE=${build_mode}
  fi
  make -j${nproc}
  echo "Done!"
}

function print_usage() {
  echo "usage: ./build.sh [build_mode] [arch_type]"
  echo "build_mode: specify build mode, you can use \"release\" or \"debug\""
  echo "arch_type: specify architecture type, you can use \"x86\" or \"aarch64\""
}

tar -zxvf vocabulary/ORBvoc.txt.tar.gz

print_usage

if [ "${build_mode}" == "release" ] || [ "${build_mode}" == "debug" ]; then
  if [ "${arch_type}" == "x86" ] || [ "${arch_type}" == "aarch64" ]; then
    echo "#######################################################################"
    echo "Building start..."
    build_orb_slam_fusion
    echo "Building finished!"
  else
    echo "Wrong arch_type, see usage above..."
  fi
else
  echo "Wrong build_mode, see usage above..."
fi