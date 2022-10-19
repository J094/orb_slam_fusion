echo "Removing build from orb_slam_fusion..."
rm -r build_release build_debug lib bin

echo "Removing build from DBoW2..."
cd 3rdparty/DBoW2
rm -r build_release build_debug lib

echo "Removing build from g2o..."
cd ../g2o
rm -r build_release build_debug lib
