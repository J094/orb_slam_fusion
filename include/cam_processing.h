#ifndef CAM_PROCESSING_H
#define CAM_PROCESSING_H

#include <opencv2/opencv.hpp>
#include <string>

#include "3rdparty/Sophus/sophus/se3.hpp"
#include "imu/imu_types.h"
#include "map/frame.h"

namespace ORB_SLAM_FUSION {

class Atlas;
class Map;
class LocalMapping;
class LoopClosing;
class System;
class KeyFrame;
class KeyFrameDatabase;
class ORBVocabulary;
class Settings;
class Viewer;
class FrameDrawer;
class MapDrawer;
class ImuProcessing;

class CamProcessing {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CamProcessing(System* sys, ORBVocabulary* voc, FrameDrawer* frame_drawer,
                MapDrawer* map_drawer, Atlas* atlas,
                KeyFrameDatabase* kf_database, const int sensor,
                Settings* settings);

  ~CamProcessing();

  void CamParamLoader(Settings* settings);

  // Preprocess the input and call Track(). Extract features and performs stereo
  // matching
  Sophus::SE3f GrabImageStereo(const cv::Mat& img_rect_left,
                               const cv::Mat& img_rect_right,
                               const double& timestamp, std::string filename);

  void SetLocalMapper(LocalMapping* pLocalMapper);
  void SetLoopClosing(LoopClosing* pLoopClosing);
  void SetViewer(Viewer* viewer);
  void SetStepByStep(bool bSet);
  bool GetStepByStep();

  // Load new settings
  // The focal lenght should be similar or scale prediction will fail when
  // projecting points
  void ChangeCalibration(const std::string& strSettingPath);

  // Use this function if you have deactivated local mapping and you only want
  // to localize the camera.
  void InformOnlyTracking(const bool& flag);

  void UpdateFrameIMU(const float s, const IMU::Bias& b,
                      KeyFrame* pCurrentKeyFrame);
  KeyFrame* GetLastKeyFrame() { return mpLastKeyFrame; }

  void CreateMapInAtlas();
  // std::mutex mMutexTracks;

  int GetMatchesInliers();

  float GetImageScale();

  void Reset(bool bLocMap = false);
  void ResetActiveMap(bool bLocMap = false);

  vector<MapPoint*> GetLocalMapMPS();

 public:
  // Tracking states
  enum eTrackingState {
    SYSTEM_NOT_READY = -1,
    NO_IMAGES_YET = 0,
    NOT_INITIALIZED = 1,
    OK = 2,
    RECENTLY_LOST = 3,
    LOST = 4,
    OK_KLT = 5
  };

  eTrackingState state_;
  eTrackingState last_processed_state_;

  // Input sensor
  int sensor_;

  // Current Frame
  Frame curr_frame_;
  Frame last_frame_;

  cv::Mat img_gray_;

  // Initialization Variables (Monocular)
  std::vector<int> mvIniLastMatches;
  std::vector<int> mvIniMatches;
  std::vector<cv::Point2f> mvbPrevMatched;
  std::vector<cv::Point3f> mvIniP3D;
  Frame mInitialFrame;

  // Lists used to recover the full camera trajectory at the end of the
  // execution. Basically we store the reference keyframe for each frame and its
  // relative transformation
  list<Sophus::SE3f> mlRelativeFramePoses;
  list<KeyFrame*> mlpReferences;
  list<double> mlFrameTimes;
  list<bool> mlbLost;

  // frames with estimated pose
  int mTrackedFr;
  bool mbStep;

  // True if local mapping is deactivated and we are performing only
  // localization
  bool mbOnlyTracking;

  void Reset(bool bLocMap = false);
  void ResetActiveMap(bool bLocMap = false);

  float mMeanTrack;
  bool mbInitWith3KFs;
  double t0;     // time-stamp of first read frame
  double t0vis;  // time-stamp of first inserted keyframe
  double t0IMU;  // time-stamp of IMU initialization
  bool mFastInit = false;

  vector<MapPoint*> GetLocalMapMPS();

  bool mbWriteStats;

 protected:
  // Main tracking function. It is independent of the input sensor.
  void Track();

  // Map initialization for stereo and RGB-D
  void StereoInitialization();

  // Map initialization for monocular
  void MonocularInitialization();
  // void CreateNewMapPoints();
  void CreateInitialMapMonocular();

  void CheckReplacedInLastFrame();
  bool TrackReferenceKeyFrame();
  void UpdateLastFrame();
  bool TrackWithMotionModel();
  bool PredictStateIMU();

  bool Relocalization();

  void UpdateLocalMap();
  void UpdateLocalPoints();
  void UpdateLocalKeyFrames();

  bool TrackLocalMap();
  void SearchLocalPoints();

  bool NeedNewKeyFrame();
  void CreateNewKeyFrame();

  // Perform preintegration from last frame
  void PreintegrateIMU();

  // Reset IMU biases and compute frame velocity
  void ResetFrameIMU();

  bool mbMapUpdated;

  // Imu preintegration from last frame
  IMU::Preintegrated* imu_preint_kf_;

  // Queue of IMU measurements between frames
  std::list<IMU::Point> mlQueueImuData;

  // Vector of IMU measurements from previous to current frame (to be filled by
  // PreintegrateIMU)
  std::vector<IMU::Point> mvImuFromLastFrame;
  std::mutex mutex_imu_;

  // Imu calibration parameters
  IMU::Calib* imu_calib_;

  // Last Bias Estimation (at keyframe creation)
  IMU::Bias mLastBias;

  // In case of performing only localization, this flag is true when there are
  // no matches to points in the map. Still tracking will continue if there are
  // enough matches with temporal points. In that case we are doing visual
  // odometry. The system will try to do relocalization to recover "zero-drift"
  // localization to the map.
  bool mbVO;

  // Other Thread Pointers
  LocalMapping* mpLocalMapper;
  LoopClosing* mpLoopClosing;

  // ORB
  ORBextractor *orb_extractor_left_, *orb_extractor_right_;
  ORBextractor* mpIniORBextractor;

  // BoW
  ORBVocabulary* orb_voc_;
  KeyFrameDatabase* mpKeyFrameDB;

  // Initalization (only for monocular)
  bool mbReadyToInitializate;
  bool mbSetInit;

  // Local Map
  KeyFrame* mpReferenceKF;
  std::vector<KeyFrame*> mvpLocalKeyFrames;
  std::vector<MapPoint*> mvpLocalMapPoints;

  // System
  System* mpSystem;

  // Drawers
  Viewer* mpViewer;
  FrameDrawer* frame_drawer_;
  MapDrawer* mpMapDrawer;
  bool bStepByStep;

  // Atlas
  Atlas* atlas_;

  // Calibration matrix
  cv::Mat cv_K_;
  Eigen::Matrix3f eig_K_;
  cv::Mat dist_coef_;
  float bf_;
  float img_scale_;

  float mImuFreq;
  double mImuPer;
  bool mInsertKFsLost;

  // New KeyFrame rules (according to fps)
  int min_frames_;
  int max_frames_;

  int mnFirstImuFrameId;
  int mnFramesToResetIMU;

  // Threshold close/far points
  // Points seen as close by the stereo/kRgbd sensor are considered reliable
  // and inserted from just one frame. Far points requiere a match in two
  // keyframes.
  float th_depth_;

  // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are
  // scaled.
  float depth_map_factor_;

  // Current matches in frame
  int mnMatchesInliers;

  // Last Frame, KeyFrame and Relocalisation Info
  KeyFrame* mpLastKeyFrame;
  unsigned int mnLastKeyFrameId;
  unsigned int mnLastRelocFrameId;
  double mTimeStampLost;
  double time_recently_lost;

  unsigned int mnFirstFrameId;
  unsigned int mnInitialFrameId;
  unsigned int mnLastInitFrameId;

  bool mbCreatedMap;

  // Motion Model
  bool mbVelocity{false};
  Sophus::SE3f mVelocity;

  // Color order (true RGB, false BGR, ignored if grayscale)
  bool is_Rgb_;

  list<MapPoint*> mlpTemporalPoints;

  // int nMapChangeIndex;

  int num_dataset_;

  ofstream f_track_stats;

  ofstream f_track_times;
  double mTime_PreIntIMU;
  double mTime_PosePred;
  double mTime_LocalMapTrack;
  double mTime_NewKF_Dec;

  GeometricCamera *cam_, *cam2_;

  int initID, lastID;

  Sophus::SE3f so_Tlr_;

  void newParameterLoader(Settings* settings);

 public:
  cv::Mat img_right_;

 public:
  ImuProcessing* imu_processing_;

  IMU::Preintegrated* imu_preint_f_;
  IMU::Preintegrated* imu_preint_kf_;
  std::mutex mutex_preint_f_;
};
}  // namespace ORB_SLAM_FUSION

#endif