#include "imu_processing.h"

#include "cam_processing.h"
#include "config/settings.h"

namespace ORB_SLAM_FUSION {

ImuProcessing::ImuProcessing(Settings* settings) { ImuParamLoader(settings); }

void ImuProcessing::ImuParamLoader(Settings* settings) {
  // IMU parameters
  Sophus::SE3f Tbc = settings->Tbc();
  insert_kfs_lost_ = settings->insertKFsWhenLost();
  imu_freq_ = settings->imuFrequency();
  // TODO: Check, if this change is good?
  imu_per_ = 1.0 / (float)imu_freq_;
  // imu_per_ = 0.001;
  float Ng = settings->noiseGyro();
  float Na = settings->noiseAcc();
  float Ngw = settings->gyroWalk();
  float Naw = settings->accWalk();

  const float sf = sqrt(imu_freq_);
  imu_calib_ = new IMU::Calib(Tbc, Ng * sf, Na * sf, Ngw / sf, Naw / sf);
}

void ImuProcessing::GrabImuData(IMU::Point* imu_data) {
  unique_lock<mutex> lock(mutex_imu_queue_);
  imu_queue_.push_back(imu_data);
}

bool ImuProcessing::CheckImuQueue() {
  unique_lock<mutex> lock(mutex_imu_queue_);
  return (!imu_queue_.empty());
}

void ImuProcessing::UpdatePreintegratedFrame() {
  unique_lock<mutex> lock(cam_processing_->mutex_preint_f_);
  if (cam_processing_->imu_preint_f_) delete cam_processing_->imu_preint_f_;
  if (cam_processing_->imu_preint_kf_) delete cam_processing_->imu_preint_kf_;
  cam_processing_->imu_preint_f_ = new IMU::Preintegrated(imu_preint_f_);
  cam_processing_->imu_preint_kf_ = new IMU::Preintegrated(imu_preint_kf_);

  if (imu_preint_f_) delete imu_preint_f_;
  imu_preint_f_ = new IMU::Preintegrated(latest_f_bias_, *imu_calib_);
}

void ImuProcessing::ProcessImuQueue() {
  {
    curr_imu_ = imu_queue_.front();
    imu_queue_.pop_front();
  }
  Eigen::Vector3f acc_ab, ang_vel_ab;
  Eigen::Vector3f acc_a, ang_vel_a;
  Eigen::Vector3f acc_b, ang_vel_b;
  {
    unique_lock<mutex> lock(mutex_f_);
    if (frame_in_) {
      float t_ab = curr_imu_->t - last_imu_->t;
      float t_a = timestamp_f_ - last_imu_->t;
      float t_b = curr_imu_->t - timestamp_f_;
      acc_ab = (last_imu_->a + curr_imu_->a) * 0.5f;
      acc_a = (last_imu_->a + curr_imu_->a -
               (curr_imu_->a - last_imu_->a) * (t_b / t_ab)) *
              0.5f;
      acc_b = (last_imu_->a + curr_imu_->a +
               (curr_imu_->a - last_imu_->a) * (t_a / t_ab)) *
              0.5f;
      ang_vel_ab = (last_imu_->w + curr_imu_->w) * 0.5f;
      ang_vel_a = (last_imu_->w + curr_imu_->w -
                   (curr_imu_->w - last_imu_->w) * (t_b / t_ab)) *
                  0.5f;
      ang_vel_b = (last_imu_->w + curr_imu_->w +
                   (curr_imu_->w - last_imu_->w) * (t_a / t_ab)) *
                  0.5f;
      imu_preint_f_->IntegrateNewMeasurement(acc_a, ang_vel_a, t_a);
      imu_preint_kf_->IntegrateNewMeasurement(acc_a, ang_vel_a, t_a);
      UpdatePreintegratedFrame();
      imu_preint_f_->IntegrateNewMeasurement(acc_b, ang_vel_b, t_b);
      imu_preint_kf_->IntegrateNewMeasurement(acc_b, ang_vel_b, t_b);
    } else {
      float t_ab = curr_imu_->t - last_imu_->t;
      acc_ab = (last_imu_->a + curr_imu_->a) * 0.5f;
      ang_vel_ab = (last_imu_->w + curr_imu_->w) * 0.5f;
    }
  }
}

//TODO: Predict state according to imu preintegrated
void ImuProcessing::PredictState() {}

void ImuProcessing::Run() {
  finished_ = false;

  while (true) {
    if (CheckImuQueue() && !bad_imu_) {
      if (!last_imu_) {
        unique_lock<mutex> lock(mutex_imu_queue_);
        last_imu_ = imu_queue_.front();
        imu_queue_.pop_front();
        continue;
      }
      ProcessImuQueue();
    }
    last_imu_ = curr_imu_;
    usleep(1000);
  }
}

}  // namespace ORB_SLAM_FUSION