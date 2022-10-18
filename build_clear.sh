echo "Removing build from orb_slam_fusion..."
rm -rf build_release build_debug lib

echo "Removing build from DBoW2..."
cd 3rdparty/DBoW2
rm -rf build_release build_debug lib

echo "Removing build from g2o..."
cd ../g2o
rm -rf build_release build_debug lib
