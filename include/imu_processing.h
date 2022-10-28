#ifndef IMU_PROCESSING_H
#define IMU_PROCESSING_H

#include <list>

#include "imu/imu_types.h"

using namespace std;

namespace ORB_SLAM_FUSION {

class Settings;
class CamProcessing;

class ImuProcessing {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ImuProcessing(Settings* settings);
  ~ImuProcessing();

  void ImuParamLoader(Settings* settings);

  void GrabImuData(IMU::Point* imu_data);

  bool CheckImuQueue();

  void UpdatePreintegratedFrame();

  void ProcessImuQueue();

  void PredictState();

  void Run();


 public:
  // CamProcessing
  CamProcessing* cam_processing_;


  double imu_freq_;
  double imu_per_;
  bool insert_kfs_lost_;

  // Imu calibration parameters
  IMU::Calib* imu_calib_;

  // Latest bias estimation
  IMU::Bias latest_f_bias_;
  IMU::Bias latest_kf_bias_;

  // Latest pose estimation
  Sophus::SE3d latest_f_pose_;
  Sophus::SE3d latest_kf_pose_;

  // Imu preintegration from cam
  IMU::Preintegrated* imu_preint_f_;
  IMU::Preintegrated* imu_preint_kf_;

  // Imu data
  std::list<IMU::Point*> imu_queue_;
  std::mutex mutex_imu_queue_;

  IMU::Point* last_imu_;
  IMU::Point* curr_imu_;

  // Flag for refreshing preintegrated
  bool frame_in_;
  double timestamp_f_;
  std::mutex mutex_f_;

  // Flag for rolling back
  bool pose_updated_;

  // Flag for loop
  bool finished_;

  // Flag for status
  bool bad_imu_;
};

}  // namespace ORB_SLAM_FUSION

#endif