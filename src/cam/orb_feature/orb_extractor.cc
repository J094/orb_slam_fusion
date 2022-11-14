/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez
 * Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós,
 * University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * ORB-SLAM3. If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "cam/orb_feature/orb_extractor.h"

#include <algorithm>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

using namespace cv;
using namespace std;

namespace ORB_SLAM_FUSION {

const int kPatchSize = 31;
const int kHalfPatchSize = 15;
const int kEdgeThreshold = 19;

static float IC_Angle(const Mat& image, Point2f pt, const vector<int>& u_max) {
  int m_01 = 0, m_10 = 0;

  const uchar* center = &image.at<uchar>(cvRound(pt.y), cvRound(pt.x));

  // Treat the center line differently, v=0
  for (int u = -kHalfPatchSize; u <= kHalfPatchSize; ++u)
    m_10 += u * center[u];

  // Go line by line in the circuI853lar patch
  int step = (int)image.step1();
  for (int v = 1; v <= kHalfPatchSize; ++v) {
    // Proceed over the two lines
    int v_sum = 0;
    int d = u_max[v];
    for (int u = -d; u <= d; ++u) {
      int val_plus = center[u + v * step], val_minus = center[u - v * step];
      v_sum += (val_plus - val_minus);
      m_10 += u * (val_plus + val_minus);
    }
    m_01 += v * v_sum;
  }

  return fastAtan2((float)m_01, (float)m_10);
}

const float factorPI = (float)(CV_PI / 180.f);
static void computeOrbDescriptor(const KeyPoint& kpt, const Mat& img,
                                 const Point* patterns_, uchar* desc) {
  float angle = (float)kpt.angle * factorPI;
  float a = (float)cos(angle), b = (float)sin(angle);

  const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
  const int step = (int)img.step;

#define GET_VALUE(idx)                                             \
  center[cvRound(patterns_[idx].x * b + patterns_[idx].y * a) * step + \
         cvRound(patterns_[idx].x * a - patterns_[idx].y * b)]

  for (int i = 0; i < 32; ++i, patterns_ += 16) {
    int t0, t1, val;
    t0 = GET_VALUE(0);
    t1 = GET_VALUE(1);
    val = t0 < t1;
    t0 = GET_VALUE(2);
    t1 = GET_VALUE(3);
    val |= (t0 < t1) << 1;
    t0 = GET_VALUE(4);
    t1 = GET_VALUE(5);
    val |= (t0 < t1) << 2;
    t0 = GET_VALUE(6);
    t1 = GET_VALUE(7);
    val |= (t0 < t1) << 3;
    t0 = GET_VALUE(8);
    t1 = GET_VALUE(9);
    val |= (t0 < t1) << 4;
    t0 = GET_VALUE(10);
    t1 = GET_VALUE(11);
    val |= (t0 < t1) << 5;
    t0 = GET_VALUE(12);
    t1 = GET_VALUE(13);
    val |= (t0 < t1) << 6;
    t0 = GET_VALUE(14);
    t1 = GET_VALUE(15);
    val |= (t0 < t1) << 7;

    desc[i] = (uchar)val;
  }

#undef GET_VALUE
}

const int kBitPattern31[256 * 4] = {
    8,   -3,  9,   5 /*mean (0), correlation (0)*/,
    4,   2,   7,   -12 /*mean (1.12461e-05), correlation (0.0437584)*/,
    -11, 9,   -8,  2 /*mean (3.37382e-05), correlation (0.0617409)*/,
    7,   -12, 12,  -13 /*mean (5.62303e-05), correlation (0.0636977)*/,
    2,   -13, 2,   12 /*mean (0.000134953), correlation (0.085099)*/,
    1,   -7,  1,   6 /*mean (0.000528565), correlation (0.0857175)*/,
    -2,  -10, -2,  -4 /*mean (0.0188821), correlation (0.0985774)*/,
    -13, -13, -11, -8 /*mean (0.0363135), correlation (0.0899616)*/,
    -13, -3,  -12, -9 /*mean (0.121806), correlation (0.099849)*/,
    10,  4,   11,  9 /*mean (0.122065), correlation (0.093285)*/,
    -13, -8,  -8,  -9 /*mean (0.162787), correlation (0.0942748)*/,
    -11, 7,   -9,  12 /*mean (0.21561), correlation (0.0974438)*/,
    7,   7,   12,  6 /*mean (0.160583), correlation (0.130064)*/,
    -4,  -5,  -3,  0 /*mean (0.228171), correlation (0.132998)*/,
    -13, 2,   -12, -3 /*mean (0.00997526), correlation (0.145926)*/,
    -9,  0,   -7,  5 /*mean (0.198234), correlation (0.143636)*/,
    12,  -6,  12,  -1 /*mean (0.0676226), correlation (0.16689)*/,
    -3,  6,   -2,  12 /*mean (0.166847), correlation (0.171682)*/,
    -6,  -13, -4,  -8 /*mean (0.101215), correlation (0.179716)*/,
    11,  -13, 12,  -8 /*mean (0.200641), correlation (0.192279)*/,
    4,   7,   5,   1 /*mean (0.205106), correlation (0.186848)*/,
    5,   -3,  10,  -3 /*mean (0.234908), correlation (0.192319)*/,
    3,   -7,  6,   12 /*mean (0.0709964), correlation (0.210872)*/,
    -8,  -7,  -6,  -2 /*mean (0.0939834), correlation (0.212589)*/,
    -2,  11,  -1,  -10 /*mean (0.127778), correlation (0.20866)*/,
    -13, 12,  -8,  10 /*mean (0.14783), correlation (0.206356)*/,
    -7,  3,   -5,  -3 /*mean (0.182141), correlation (0.198942)*/,
    -4,  2,   -3,  7 /*mean (0.188237), correlation (0.21384)*/,
    -10, -12, -6,  11 /*mean (0.14865), correlation (0.23571)*/,
    5,   -12, 6,   -7 /*mean (0.222312), correlation (0.23324)*/,
    5,   -6,  7,   -1 /*mean (0.229082), correlation (0.23389)*/,
    1,   0,   4,   -5 /*mean (0.241577), correlation (0.215286)*/,
    9,   11,  11,  -13 /*mean (0.00338507), correlation (0.251373)*/,
    4,   7,   4,   12 /*mean (0.131005), correlation (0.257622)*/,
    2,   -1,  4,   4 /*mean (0.152755), correlation (0.255205)*/,
    -4,  -12, -2,  7 /*mean (0.182771), correlation (0.244867)*/,
    -8,  -5,  -7,  -10 /*mean (0.186898), correlation (0.23901)*/,
    4,   11,  9,   12 /*mean (0.226226), correlation (0.258255)*/,
    0,   -8,  1,   -13 /*mean (0.0897886), correlation (0.274827)*/,
    -13, -2,  -8,  2 /*mean (0.148774), correlation (0.28065)*/,
    -3,  -2,  -2,  3 /*mean (0.153048), correlation (0.283063)*/,
    -6,  9,   -4,  -9 /*mean (0.169523), correlation (0.278248)*/,
    8,   12,  10,  7 /*mean (0.225337), correlation (0.282851)*/,
    0,   9,   1,   3 /*mean (0.226687), correlation (0.278734)*/,
    7,   -5,  11,  -10 /*mean (0.00693882), correlation (0.305161)*/,
    -13, -6,  -11, 0 /*mean (0.0227283), correlation (0.300181)*/,
    10,  7,   12,  1 /*mean (0.125517), correlation (0.31089)*/,
    -6,  -3,  -6,  12 /*mean (0.131748), correlation (0.312779)*/,
    10,  -9,  12,  -4 /*mean (0.144827), correlation (0.292797)*/,
    -13, 8,   -8,  -12 /*mean (0.149202), correlation (0.308918)*/,
    -13, 0,   -8,  -4 /*mean (0.160909), correlation (0.310013)*/,
    3,   3,   7,   8 /*mean (0.177755), correlation (0.309394)*/,
    5,   7,   10,  -7 /*mean (0.212337), correlation (0.310315)*/,
    -1,  7,   1,   -12 /*mean (0.214429), correlation (0.311933)*/,
    3,   -10, 5,   6 /*mean (0.235807), correlation (0.313104)*/,
    2,   -4,  3,   -10 /*mean (0.00494827), correlation (0.344948)*/,
    -13, 0,   -13, 5 /*mean (0.0549145), correlation (0.344675)*/,
    -13, -7,  -12, 12 /*mean (0.103385), correlation (0.342715)*/,
    -13, 3,   -11, 8 /*mean (0.134222), correlation (0.322922)*/,
    -7,  12,  -4,  7 /*mean (0.153284), correlation (0.337061)*/,
    6,   -10, 12,  8 /*mean (0.154881), correlation (0.329257)*/,
    -9,  -1,  -7,  -6 /*mean (0.200967), correlation (0.33312)*/,
    -2,  -5,  0,   12 /*mean (0.201518), correlation (0.340635)*/,
    -12, 5,   -7,  5 /*mean (0.207805), correlation (0.335631)*/,
    3,   -10, 8,   -13 /*mean (0.224438), correlation (0.34504)*/,
    -7,  -7,  -4,  5 /*mean (0.239361), correlation (0.338053)*/,
    -3,  -2,  -1,  -7 /*mean (0.240744), correlation (0.344322)*/,
    2,   9,   5,   -11 /*mean (0.242949), correlation (0.34145)*/,
    -11, -13, -5,  -13 /*mean (0.244028), correlation (0.336861)*/,
    -1,  6,   0,   -1 /*mean (0.247571), correlation (0.343684)*/,
    5,   -3,  5,   2 /*mean (0.000697256), correlation (0.357265)*/,
    -4,  -13, -4,  12 /*mean (0.00213675), correlation (0.373827)*/,
    -9,  -6,  -9,  6 /*mean (0.0126856), correlation (0.373938)*/,
    -12, -10, -8,  -4 /*mean (0.0152497), correlation (0.364237)*/,
    10,  2,   12,  -3 /*mean (0.0299933), correlation (0.345292)*/,
    7,   12,  12,  12 /*mean (0.0307242), correlation (0.366299)*/,
    -7,  -13, -6,  5 /*mean (0.0534975), correlation (0.368357)*/,
    -4,  9,   -3,  4 /*mean (0.099865), correlation (0.372276)*/,
    7,   -1,  12,  2 /*mean (0.117083), correlation (0.364529)*/,
    -7,  6,   -5,  1 /*mean (0.126125), correlation (0.369606)*/,
    -13, 11,  -12, 5 /*mean (0.130364), correlation (0.358502)*/,
    -3,  7,   -2,  -6 /*mean (0.131691), correlation (0.375531)*/,
    7,   -8,  12,  -7 /*mean (0.160166), correlation (0.379508)*/,
    -13, -7,  -11, -12 /*mean (0.167848), correlation (0.353343)*/,
    1,   -3,  12,  12 /*mean (0.183378), correlation (0.371916)*/,
    2,   -6,  3,   0 /*mean (0.228711), correlation (0.371761)*/,
    -4,  3,   -2,  -13 /*mean (0.247211), correlation (0.364063)*/,
    -1,  -13, 1,   9 /*mean (0.249325), correlation (0.378139)*/,
    7,   1,   8,   -6 /*mean (0.000652272), correlation (0.411682)*/,
    1,   -1,  3,   12 /*mean (0.00248538), correlation (0.392988)*/,
    9,   1,   12,  6 /*mean (0.0206815), correlation (0.386106)*/,
    -1,  -9,  -1,  3 /*mean (0.0364485), correlation (0.410752)*/,
    -13, -13, -10, 5 /*mean (0.0376068), correlation (0.398374)*/,
    7,   7,   10,  12 /*mean (0.0424202), correlation (0.405663)*/,
    12,  -5,  12,  9 /*mean (0.0942645), correlation (0.410422)*/,
    6,   3,   7,   11 /*mean (0.1074), correlation (0.413224)*/,
    5,   -13, 6,   10 /*mean (0.109256), correlation (0.408646)*/,
    2,   -12, 2,   3 /*mean (0.131691), correlation (0.416076)*/,
    3,   8,   4,   -6 /*mean (0.165081), correlation (0.417569)*/,
    2,   6,   12,  -13 /*mean (0.171874), correlation (0.408471)*/,
    9,   -12, 10,  3 /*mean (0.175146), correlation (0.41296)*/,
    -8,  4,   -7,  9 /*mean (0.183682), correlation (0.402956)*/,
    -11, 12,  -4,  -6 /*mean (0.184672), correlation (0.416125)*/,
    1,   12,  2,   -8 /*mean (0.191487), correlation (0.386696)*/,
    6,   -9,  7,   -4 /*mean (0.192668), correlation (0.394771)*/,
    2,   3,   3,   -2 /*mean (0.200157), correlation (0.408303)*/,
    6,   3,   11,  0 /*mean (0.204588), correlation (0.411762)*/,
    3,   -3,  8,   -8 /*mean (0.205904), correlation (0.416294)*/,
    7,   8,   9,   3 /*mean (0.213237), correlation (0.409306)*/,
    -11, -5,  -6,  -4 /*mean (0.243444), correlation (0.395069)*/,
    -10, 11,  -5,  10 /*mean (0.247672), correlation (0.413392)*/,
    -5,  -8,  -3,  12 /*mean (0.24774), correlation (0.411416)*/,
    -10, 5,   -9,  0 /*mean (0.00213675), correlation (0.454003)*/,
    8,   -1,  12,  -6 /*mean (0.0293635), correlation (0.455368)*/,
    4,   -6,  6,   -11 /*mean (0.0404971), correlation (0.457393)*/,
    -10, 12,  -8,  7 /*mean (0.0481107), correlation (0.448364)*/,
    4,   -2,  6,   7 /*mean (0.050641), correlation (0.455019)*/,
    -2,  0,   -2,  12 /*mean (0.0525978), correlation (0.44338)*/,
    -5,  -8,  -5,  2 /*mean (0.0629667), correlation (0.457096)*/,
    7,   -6,  10,  12 /*mean (0.0653846), correlation (0.445623)*/,
    -9,  -13, -8,  -8 /*mean (0.0858749), correlation (0.449789)*/,
    -5,  -13, -5,  -2 /*mean (0.122402), correlation (0.450201)*/,
    8,   -8,  9,   -13 /*mean (0.125416), correlation (0.453224)*/,
    -9,  -11, -9,  0 /*mean (0.130128), correlation (0.458724)*/,
    1,   -8,  1,   -2 /*mean (0.132467), correlation (0.440133)*/,
    7,   -4,  9,   1 /*mean (0.132692), correlation (0.454)*/,
    -2,  1,   -1,  -4 /*mean (0.135695), correlation (0.455739)*/,
    11,  -6,  12,  -11 /*mean (0.142904), correlation (0.446114)*/,
    -12, -9,  -6,  4 /*mean (0.146165), correlation (0.451473)*/,
    3,   7,   7,   12 /*mean (0.147627), correlation (0.456643)*/,
    5,   5,   10,  8 /*mean (0.152901), correlation (0.455036)*/,
    0,   -4,  2,   8 /*mean (0.167083), correlation (0.459315)*/,
    -9,  12,  -5,  -13 /*mean (0.173234), correlation (0.454706)*/,
    0,   7,   2,   12 /*mean (0.18312), correlation (0.433855)*/,
    -1,  2,   1,   7 /*mean (0.185504), correlation (0.443838)*/,
    5,   11,  7,   -9 /*mean (0.185706), correlation (0.451123)*/,
    3,   5,   6,   -8 /*mean (0.188968), correlation (0.455808)*/,
    -13, -4,  -8,  9 /*mean (0.191667), correlation (0.459128)*/,
    -5,  9,   -3,  -3 /*mean (0.193196), correlation (0.458364)*/,
    -4,  -7,  -3,  -12 /*mean (0.196536), correlation (0.455782)*/,
    6,   5,   8,   0 /*mean (0.1972), correlation (0.450481)*/,
    -7,  6,   -6,  12 /*mean (0.199438), correlation (0.458156)*/,
    -13, 6,   -5,  -2 /*mean (0.211224), correlation (0.449548)*/,
    1,   -10, 3,   10 /*mean (0.211718), correlation (0.440606)*/,
    4,   1,   8,   -4 /*mean (0.213034), correlation (0.443177)*/,
    -2,  -2,  2,   -13 /*mean (0.234334), correlation (0.455304)*/,
    2,   -12, 12,  12 /*mean (0.235684), correlation (0.443436)*/,
    -2,  -13, 0,   -6 /*mean (0.237674), correlation (0.452525)*/,
    4,   1,   9,   3 /*mean (0.23962), correlation (0.444824)*/,
    -6,  -10, -3,  -5 /*mean (0.248459), correlation (0.439621)*/,
    -3,  -13, -1,  1 /*mean (0.249505), correlation (0.456666)*/,
    7,   5,   12,  -11 /*mean (0.00119208), correlation (0.495466)*/,
    4,   -2,  5,   -7 /*mean (0.00372245), correlation (0.484214)*/,
    -13, 9,   -9,  -5 /*mean (0.00741116), correlation (0.499854)*/,
    7,   1,   8,   6 /*mean (0.0208952), correlation (0.499773)*/,
    7,   -8,  7,   6 /*mean (0.0220085), correlation (0.501609)*/,
    -7,  -4,  -7,  1 /*mean (0.0233806), correlation (0.496568)*/,
    -8,  11,  -7,  -8 /*mean (0.0236505), correlation (0.489719)*/,
    -13, 6,   -12, -8 /*mean (0.0268781), correlation (0.503487)*/,
    2,   4,   3,   9 /*mean (0.0323324), correlation (0.501938)*/,
    10,  -5,  12,  3 /*mean (0.0399235), correlation (0.494029)*/,
    -6,  -5,  -6,  7 /*mean (0.0420153), correlation (0.486579)*/,
    8,   -3,  9,   -8 /*mean (0.0548021), correlation (0.484237)*/,
    2,   -12, 2,   8 /*mean (0.0616622), correlation (0.496642)*/,
    -11, -2,  -10, 3 /*mean (0.0627755), correlation (0.498563)*/,
    -12, -13, -7,  -9 /*mean (0.0829622), correlation (0.495491)*/,
    -11, 0,   -10, -5 /*mean (0.0843342), correlation (0.487146)*/,
    5,   -3,  11,  8 /*mean (0.0929937), correlation (0.502315)*/,
    -2,  -13, -1,  12 /*mean (0.113327), correlation (0.48941)*/,
    -1,  -8,  0,   9 /*mean (0.132119), correlation (0.467268)*/,
    -13, -11, -12, -5 /*mean (0.136269), correlation (0.498771)*/,
    -10, -2,  -10, 11 /*mean (0.142173), correlation (0.498714)*/,
    -3,  9,   -2,  -13 /*mean (0.144141), correlation (0.491973)*/,
    2,   -3,  3,   2 /*mean (0.14892), correlation (0.500782)*/,
    -9,  -13, -4,  0 /*mean (0.150371), correlation (0.498211)*/,
    -4,  6,   -3,  -10 /*mean (0.152159), correlation (0.495547)*/,
    -4,  12,  -2,  -7 /*mean (0.156152), correlation (0.496925)*/,
    -6,  -11, -4,  9 /*mean (0.15749), correlation (0.499222)*/,
    6,   -3,  6,   11 /*mean (0.159211), correlation (0.503821)*/,
    -13, 11,  -5,  5 /*mean (0.162427), correlation (0.501907)*/,
    11,  11,  12,  6 /*mean (0.16652), correlation (0.497632)*/,
    7,   -5,  12,  -2 /*mean (0.169141), correlation (0.484474)*/,
    -1,  12,  0,   7 /*mean (0.169456), correlation (0.495339)*/,
    -4,  -8,  -3,  -2 /*mean (0.171457), correlation (0.487251)*/,
    -7,  1,   -6,  7 /*mean (0.175), correlation (0.500024)*/,
    -13, -12, -8,  -13 /*mean (0.175866), correlation (0.497523)*/,
    -7,  -2,  -6,  -8 /*mean (0.178273), correlation (0.501854)*/,
    -8,  5,   -6,  -9 /*mean (0.181107), correlation (0.494888)*/,
    -5,  -1,  -4,  5 /*mean (0.190227), correlation (0.482557)*/,
    -13, 7,   -8,  10 /*mean (0.196739), correlation (0.496503)*/,
    1,   5,   5,   -13 /*mean (0.19973), correlation (0.499759)*/,
    1,   0,   10,  -13 /*mean (0.204465), correlation (0.49873)*/,
    9,   12,  10,  -1 /*mean (0.209334), correlation (0.49063)*/,
    5,   -8,  10,  -9 /*mean (0.211134), correlation (0.503011)*/,
    -1,  11,  1,   -13 /*mean (0.212), correlation (0.499414)*/,
    -9,  -3,  -6,  2 /*mean (0.212168), correlation (0.480739)*/,
    -1,  -10, 1,   12 /*mean (0.212731), correlation (0.502523)*/,
    -13, 1,   -8,  -10 /*mean (0.21327), correlation (0.489786)*/,
    8,   -11, 10,  -6 /*mean (0.214159), correlation (0.488246)*/,
    2,   -13, 3,   -6 /*mean (0.216993), correlation (0.50287)*/,
    7,   -13, 12,  -9 /*mean (0.223639), correlation (0.470502)*/,
    -10, -10, -5,  -7 /*mean (0.224089), correlation (0.500852)*/,
    -10, -8,  -8,  -13 /*mean (0.228666), correlation (0.502629)*/,
    4,   -6,  8,   5 /*mean (0.22906), correlation (0.498305)*/,
    3,   12,  8,   -13 /*mean (0.233378), correlation (0.503825)*/,
    -4,  2,   -3,  -3 /*mean (0.234323), correlation (0.476692)*/,
    5,   -13, 10,  -12 /*mean (0.236392), correlation (0.475462)*/,
    4,   -13, 5,   -1 /*mean (0.236842), correlation (0.504132)*/,
    -9,  9,   -4,  3 /*mean (0.236977), correlation (0.497739)*/,
    0,   3,   3,   -9 /*mean (0.24314), correlation (0.499398)*/,
    -12, 1,   -6,  1 /*mean (0.243297), correlation (0.489447)*/,
    3,   2,   4,   -8 /*mean (0.00155196), correlation (0.553496)*/,
    -10, -10, -10, 9 /*mean (0.00239541), correlation (0.54297)*/,
    8,   -13, 12,  12 /*mean (0.0034413), correlation (0.544361)*/,
    -8,  -12, -6,  -5 /*mean (0.003565), correlation (0.551225)*/,
    2,   2,   3,   7 /*mean (0.00835583), correlation (0.55285)*/,
    10,  6,   11,  -8 /*mean (0.00885065), correlation (0.540913)*/,
    6,   8,   8,   -12 /*mean (0.0101552), correlation (0.551085)*/,
    -7,  10,  -6,  5 /*mean (0.0102227), correlation (0.533635)*/,
    -3,  -9,  -3,  9 /*mean (0.0110211), correlation (0.543121)*/,
    -1,  -13, -1,  5 /*mean (0.0113473), correlation (0.550173)*/,
    -3,  -7,  -3,  4 /*mean (0.0140913), correlation (0.554774)*/,
    -8,  -2,  -8,  3 /*mean (0.017049), correlation (0.55461)*/,
    4,   2,   12,  12 /*mean (0.01778), correlation (0.546921)*/,
    2,   -5,  3,   11 /*mean (0.0224022), correlation (0.549667)*/,
    6,   -9,  11,  -13 /*mean (0.029161), correlation (0.546295)*/,
    3,   -1,  7,   12 /*mean (0.0303081), correlation (0.548599)*/,
    11,  -1,  12,  4 /*mean (0.0355151), correlation (0.523943)*/,
    -3,  0,   -3,  6 /*mean (0.0417904), correlation (0.543395)*/,
    4,   -11, 4,   12 /*mean (0.0487292), correlation (0.542818)*/,
    2,   -4,  2,   1 /*mean (0.0575124), correlation (0.554888)*/,
    -10, -6,  -8,  1 /*mean (0.0594242), correlation (0.544026)*/,
    -13, 7,   -11, 1 /*mean (0.0597391), correlation (0.550524)*/,
    -13, 12,  -11, -13 /*mean (0.0608974), correlation (0.55383)*/,
    6,   0,   11,  -13 /*mean (0.065126), correlation (0.552006)*/,
    0,   -1,  1,   4 /*mean (0.074224), correlation (0.546372)*/,
    -13, 3,   -9,  -2 /*mean (0.0808592), correlation (0.554875)*/,
    -9,  8,   -6,  -3 /*mean (0.0883378), correlation (0.551178)*/,
    -13, -6,  -8,  -2 /*mean (0.0901035), correlation (0.548446)*/,
    5,   -9,  8,   10 /*mean (0.0949843), correlation (0.554694)*/,
    2,   7,   3,   -9 /*mean (0.0994152), correlation (0.550979)*/,
    -1,  -6,  -1,  -1 /*mean (0.10045), correlation (0.552714)*/,
    9,   5,   11,  -2 /*mean (0.100686), correlation (0.552594)*/,
    11,  -3,  12,  -8 /*mean (0.101091), correlation (0.532394)*/,
    3,   0,   3,   5 /*mean (0.101147), correlation (0.525576)*/,
    -1,  4,   0,   10 /*mean (0.105263), correlation (0.531498)*/,
    3,   -6,  4,   5 /*mean (0.110785), correlation (0.540491)*/,
    -13, 0,   -10, 5 /*mean (0.112798), correlation (0.536582)*/,
    5,   8,   12,  11 /*mean (0.114181), correlation (0.555793)*/,
    8,   9,   9,   -6 /*mean (0.117431), correlation (0.553763)*/,
    7,   -4,  8,   -12 /*mean (0.118522), correlation (0.553452)*/,
    -10, 4,   -10, 9 /*mean (0.12094), correlation (0.554785)*/,
    7,   3,   12,  4 /*mean (0.122582), correlation (0.555825)*/,
    9,   -7,  10,  -2 /*mean (0.124978), correlation (0.549846)*/,
    7,   0,   12,  -2 /*mean (0.127002), correlation (0.537452)*/,
    -1,  -6,  0,   -11 /*mean (0.127148), correlation (0.547401)*/
};

OrbExtractor::OrbExtractor(int num_feats, float scale_factor, int num_levs,
                           int ini_th_fast, int min_th_fast)
    : num_feats_(num_feats),
      scale_factor_(scale_factor),
      num_levs_(num_levs),
      ini_th_fast_(ini_th_fast),
      min_th_fast_(min_th_fast) {
  scale_factors_.resize(num_levs_);
  lev_sigma_2_.resize(num_levs_);
  scale_factors_[0] = 1.0f;
  lev_sigma_2_[0] = 1.0f;
  for (int i = 1; i < num_levs_; i++) {
    scale_factors_[i] = scale_factors_[i - 1] * scale_factor_;
    lev_sigma_2_[i] = scale_factors_[i] * scale_factors_[i];
  }

  inv_scale_factors_.resize(num_levs_);
  inv_lev_sigma_2_.resize(num_levs_);
  for (int i = 0; i < num_levs_; i++) {
    inv_scale_factors_[i] = 1.0f / scale_factors_[i];
    inv_lev_sigma_2_[i] = 1.0f / lev_sigma_2_[i];
  }

  img_pyramid_.resize(num_levs_);

  num_feats_per_lev_.resize(num_levs_);
  float factor = 1.0f / scale_factor_;
  float nDesiredFeaturesPerScale =
      num_feats_ * (1 - factor) /
      (1 - (float)pow((double)factor, (double)num_levs_));

  int sumFeatures = 0;
  for (int lev = 0; lev < num_levs_ - 1; lev++) {
    num_feats_per_lev_[lev] = cvRound(nDesiredFeaturesPerScale);
    sumFeatures += num_feats_per_lev_[lev];
    nDesiredFeaturesPerScale *= factor;
  }
  num_feats_per_lev_[num_levs_ - 1] = std::max(num_feats_ - sumFeatures, 0);

  const int npoints = 512;
  const Point* pattern0 = (const Point*)kBitPattern31;
  std::copy(pattern0, pattern0 + npoints, std::back_inserter(patterns_));

  // This is for orientation
  //  pre-compute the end of a row in a circular patch
  umax_.resize(kHalfPatchSize + 1);

  int v, v0, vmax = cvFloor(kHalfPatchSize * sqrt(2.f) / 2 + 1);
  int vmin = cvCeil(kHalfPatchSize * sqrt(2.f) / 2);
  const double hp2 = kHalfPatchSize * kHalfPatchSize;
  for (v = 0; v <= vmax; ++v) umax_[v] = cvRound(sqrt(hp2 - v * v));

  // Make sure we are symmetric
  for (v = kHalfPatchSize, v0 = 0; v >= vmin; --v) {
    while (umax_[v0] == umax_[v0 + 1]) ++v0;
    umax_[v] = v0;
    ++v0;
  }
}

static void computeOrientation(const Mat& image, vector<KeyPoint>& kps,
                               const vector<int>& umax_) {
  for (vector<KeyPoint>::iterator keypoint = kps.begin(),
                                  keypointEnd = kps.end();
       keypoint != keypointEnd; ++keypoint) {
    keypoint->angle = IC_Angle(image, keypoint->pt, umax_);
  }
}

void ExtractorNode::DivideNode(ExtractorNode& n_1, ExtractorNode& n_2,
                               ExtractorNode& n_3, ExtractorNode& n_4) {
  const int halfX = ceil(static_cast<float>(UR_.x - UL_.x) / 2);
  const int halfY = ceil(static_cast<float>(BR_.y - UL_.y) / 2);

  // Define boundaries of childs
  n_1.UL_ = UL_;
  n_1.UR_ = cv::Point2i(UL_.x + halfX, UL_.y);
  n_1.BL_ = cv::Point2i(UL_.x, UL_.y + halfY);
  n_1.BR_ = cv::Point2i(UL_.x + halfX, UL_.y + halfY);
  n_1.kps_.reserve(kps_.size());

  n_2.UL_ = n_1.UR_;
  n_2.UR_ = UR_;
  n_2.BL_ = n_1.BR_;
  n_2.BR_ = cv::Point2i(UR_.x, UL_.y + halfY);
  n_2.kps_.reserve(kps_.size());

  n_3.UL_ = n_1.BL_;
  n_3.UR_ = n_1.BR_;
  n_3.BL_ = BL_;
  n_3.BR_ = cv::Point2i(n_1.BR_.x, BL_.y);
  n_3.kps_.reserve(kps_.size());

  n_4.UL_ = n_3.UR_;
  n_4.UR_ = n_2.BR_;
  n_4.BL_ = n_3.BR_;
  n_4.BR_ = BR_;
  n_4.kps_.reserve(kps_.size());

  // Associate points to childs
  for (size_t i = 0; i < kps_.size(); i++) {
    const cv::KeyPoint& kp = kps_[i];
    if (kp.pt.x < n_1.UR_.x) {
      if (kp.pt.y < n_1.BR_.y)
        n_1.kps_.push_back(kp);
      else
        n_3.kps_.push_back(kp);
    } else if (kp.pt.y < n_1.BR_.y)
      n_2.kps_.push_back(kp);
    else
      n_4.kps_.push_back(kp);
  }

  if (n_1.kps_.size() == 1) n_1.no_more_ = true;
  if (n_2.kps_.size() == 1) n_2.no_more_ = true;
  if (n_3.kps_.size() == 1) n_3.no_more_ = true;
  if (n_4.kps_.size() == 1) n_4.no_more_ = true;
}

//NOTE: stable_sort need const arguments
static bool compareNodes(const pair<int, ExtractorNode*>& e1,
                         const pair<int, ExtractorNode*>& e2) {
  if (e1.first < e2.first) {
    return true;
  } else if (e1.first > e2.first) {
    return false;
  } else {
    if (e1.second->UL_.x < e2.second->UL_.x) {
      return true;
    } else {
      return false;
    }
  }
}

vector<cv::KeyPoint> OrbExtractor::DistributeOctTree(
    const vector<cv::KeyPoint>& to_dist_kps, const int& min_x,
    const int& max_x, const int& min_y, const int& max_y, const int& num_feats,
    const int& lev) {
  // TODO: Check, if there is a bug here?
  //  Compute how many initial nodes
  const int num_ini_nds = round(static_cast<float>(max_x - min_x) / (max_y - min_y));

  const float h_x = static_cast<float>(max_x - min_x) / num_ini_nds;

  list<ExtractorNode> nds;

  vector<ExtractorNode*> ini_nds;
  ini_nds.resize(num_ini_nds);

  for (int i = 0; i < num_ini_nds; i++) {
    ExtractorNode ni;
    ni.UL_ = cv::Point2i(h_x * static_cast<float>(i), 0);
    ni.UR_ = cv::Point2i(h_x * static_cast<float>(i + 1), 0);
    ni.BL_ = cv::Point2i(ni.UL_.x, max_y - min_y);
    ni.BR_ = cv::Point2i(ni.UR_.x, max_y - min_y);
    ni.kps_.reserve(to_dist_kps.size());

    nds.push_back(ni);
    ini_nds[i] = &nds.back();
  }

  // Associate points to childs
  for (size_t i = 0; i < to_dist_kps.size(); i++) {
    const cv::KeyPoint& kp = to_dist_kps[i];
    ini_nds[kp.pt.x / h_x]->kps_.push_back(kp);
  }

  list<ExtractorNode>::iterator lit = nds.begin();

  while (lit != nds.end()) {
    if (lit->kps_.size() == 1) {
      lit->no_more_ = true;
      lit++;
    } else if (lit->kps_.empty())
      lit = nds.erase(lit);
    else
      lit++;
  }

  bool is_finished = false;

  int num_iter = 0;

  vector<pair<int, ExtractorNode*> > kps_size_and_nd;
  kps_size_and_nd.reserve(nds.size() * 4);

  while (!is_finished) {
    num_iter++;

    int size_prev = nds.size();

    lit = nds.begin();

    int num_to_expand = 0;

    kps_size_and_nd.clear();

    while (lit != nds.end()) {
      if (lit->no_more_) {
        // If node only contains one point do not subdivide and continue
        lit++;
        continue;
      } else {
        // If more than one point, subdivide
        ExtractorNode n_1, n_2, n_3, n_4;
        lit->DivideNode(n_1, n_2, n_3, n_4);

        // Add childs if they contain points
        if (n_1.kps_.size() > 0) {
          nds.push_front(n_1);
          if (n_1.kps_.size() > 1) {
            num_to_expand++;
            kps_size_and_nd.push_back(
                make_pair(n_1.kps_.size(), &nds.front()));
            nds.front().lit_ = nds.begin();
          }
        }
        if (n_2.kps_.size() > 0) {
          nds.push_front(n_2);
          if (n_2.kps_.size() > 1) {
            num_to_expand++;
            kps_size_and_nd.push_back(
                make_pair(n_2.kps_.size(), &nds.front()));
            nds.front().lit_ = nds.begin();
          }
        }
        if (n_3.kps_.size() > 0) {
          nds.push_front(n_3);
          if (n_3.kps_.size() > 1) {
            num_to_expand++;
            kps_size_and_nd.push_back(
                make_pair(n_3.kps_.size(), &nds.front()));
            nds.front().lit_ = nds.begin();
          }
        }
        if (n_4.kps_.size() > 0) {
          nds.push_front(n_4);
          if (n_4.kps_.size() > 1) {
            num_to_expand++;
            kps_size_and_nd.push_back(
                make_pair(n_4.kps_.size(), &nds.front()));
            nds.front().lit_ = nds.begin();
          }
        }

        lit = nds.erase(lit);
        continue;
      }
    }

    // Finish if there are more nodes than required features
    // or all nodes contain just one point
    if ((int)nds.size() >= num_feats || (int)nds.size() == size_prev) {
      is_finished = true;
    } else if (((int)nds.size() + num_to_expand * 3) > num_feats) {
      while (!is_finished) {
        size_prev = nds.size();

        vector<pair<int, ExtractorNode*> > vPrevSizeAndPointerToNode =
            kps_size_and_nd;
        kps_size_and_nd.clear();

        // TODO: Check, if this change is good?
        stable_sort(vPrevSizeAndPointerToNode.begin(),
                    vPrevSizeAndPointerToNode.end(), compareNodes);
        for (int j = vPrevSizeAndPointerToNode.size() - 1; j >= 0; j--) {
          ExtractorNode n_1, n_2, n_3, n_4;
          vPrevSizeAndPointerToNode[j].second->DivideNode(n_1, n_2, n_3, n_4);

          // Add childs if they contain points
          if (n_1.kps_.size() > 0) {
            nds.push_front(n_1);
            if (n_1.kps_.size() > 1) {
              kps_size_and_nd.push_back(
                  make_pair(n_1.kps_.size(), &nds.front()));
              nds.front().lit_ = nds.begin();
            }
          }
          if (n_2.kps_.size() > 0) {
            nds.push_front(n_2);
            if (n_2.kps_.size() > 1) {
              kps_size_and_nd.push_back(
                  make_pair(n_2.kps_.size(), &nds.front()));
              nds.front().lit_ = nds.begin();
            }
          }
          if (n_3.kps_.size() > 0) {
            nds.push_front(n_3);
            if (n_3.kps_.size() > 1) {
              kps_size_and_nd.push_back(
                  make_pair(n_3.kps_.size(), &nds.front()));
              nds.front().lit_ = nds.begin();
            }
          }
          if (n_4.kps_.size() > 0) {
            nds.push_front(n_4);
            if (n_4.kps_.size() > 1) {
              kps_size_and_nd.push_back(
                  make_pair(n_4.kps_.size(), &nds.front()));
              nds.front().lit_ = nds.begin();
            }
          }

          nds.erase(vPrevSizeAndPointerToNode[j].second->lit_);

          if ((int)nds.size() >= num_feats) break;
        }

        if ((int)nds.size() >= num_feats || (int)nds.size() == size_prev)
          is_finished = true;
      }
    }
  }

  // Retain the best point in each node
  vector<cv::KeyPoint> vResultKeys;
  vResultKeys.reserve(num_feats_);
  for (list<ExtractorNode>::iterator lit = nds.begin(); lit != nds.end();
       lit++) {
    vector<cv::KeyPoint>& vNodeKeys = lit->kps_;
    cv::KeyPoint* pKP = &vNodeKeys[0];
    float maxResponse = pKP->response;

    for (size_t k = 1; k < vNodeKeys.size(); k++) {
      if (vNodeKeys[k].response > maxResponse) {
        pKP = &vNodeKeys[k];
        maxResponse = vNodeKeys[k].response;
      }
    }

    vResultKeys.push_back(*pKP);
  }

  return vResultKeys;
}

void OrbExtractor::ComputeKeyPointsOctTree(
    vector<vector<KeyPoint> >& all_kps) {
  all_kps.resize(num_levs_);

  const float W = 35;

  for (int lev = 0; lev < num_levs_; ++lev) {
    const int min_border_x = kEdgeThreshold - 3;
    const int min_border_y = min_border_x;
    const int max_border_x = img_pyramid_[lev].cols - kEdgeThreshold + 3;
    const int max_border_y = img_pyramid_[lev].rows - kEdgeThreshold + 3;

    vector<cv::KeyPoint> to_dist_kps;
    to_dist_kps.reserve(num_feats_ * 10);

    const float w = (max_border_x - min_border_x);
    const float h = (max_border_y - min_border_y);

    const int num_cols = w / W;
    const int num_rows = h / W;
    const int w_cell = ceil(w / num_cols);
    const int h_cell = ceil(h / num_rows);

    for (int i = 0; i < num_rows; i++) {
      const float ini_y = min_border_y + i * h_cell;
      float max_y = ini_y + h_cell + 6;

      if (ini_y >= max_border_y - 3) continue;
      if (max_y > max_border_y) max_y = max_border_y;

      for (int j = 0; j < num_cols; j++) {
        const float ini_x = min_border_x + j * w_cell;
        float max_x = ini_x + w_cell + 6;
        // TODO: Check, if this change is good?
        if (ini_x >= max_border_x - 3) continue;
        if (max_x > max_border_x) max_x = max_border_x;

        vector<cv::KeyPoint> cell_kps;

        FAST(img_pyramid_[lev].rowRange(ini_y, max_y).colRange(ini_x, max_x),
             cell_kps, ini_th_fast_, true);

        /*if(bRight && j <= 13){
            FAST(img_pyramid_[lev].rowRange(ini_y,max_y).colRange(ini_x,max_x),
                 cell_kps,10,true);
        }
        else if(!bRight && j >= 16){
            FAST(img_pyramid_[lev].rowRange(ini_y,max_y).colRange(ini_x,max_x),
                 cell_kps,10,true);
        }
        else{
            FAST(img_pyramid_[lev].rowRange(ini_y,max_y).colRange(ini_x,max_x),
                 cell_kps,ini_th_fast_,true);
        }*/

        if (cell_kps.empty()) {
          FAST(img_pyramid_[lev].rowRange(ini_y, max_y).colRange(ini_x, max_x),
               cell_kps, min_th_fast_, true);
          /*if(bRight && j <= 13){
              FAST(img_pyramid_[lev].rowRange(ini_y,max_y).colRange(ini_x,max_x),
                   cell_kps,5,true);
          }
          else if(!bRight && j >= 16){
              FAST(img_pyramid_[lev].rowRange(ini_y,max_y).colRange(ini_x,max_x),
                   cell_kps,5,true);
          }
          else{
              FAST(img_pyramid_[lev].rowRange(ini_y,max_y).colRange(ini_x,max_x),
                   cell_kps,min_th_fast_,true);
          }*/
        }

        if (!cell_kps.empty()) {
          for (vector<cv::KeyPoint>::iterator vit = cell_kps.begin();
               vit != cell_kps.end(); vit++) {
            (*vit).pt.x += j * w_cell;
            (*vit).pt.y += i * h_cell;
            to_dist_kps.push_back(*vit);
          }
        }
      }
    }

    vector<KeyPoint>& kps = all_kps[lev];
    kps.reserve(num_feats_);

    kps =
        DistributeOctTree(to_dist_kps, min_border_x, max_border_x, min_border_y,
                          max_border_y, num_feats_per_lev_[lev], lev);

    const int scaled_patch_size = kPatchSize * scale_factors_[lev];

    // Add border to coordinates and scale information
    const int num_kps = kps.size();
    for (int i = 0; i < num_kps; i++) {
      kps[i].pt.x += min_border_x;
      kps[i].pt.y += min_border_y;
      kps[i].octave = lev;
      kps[i].size = scaled_patch_size;
    }
  }

  // compute orientations
  for (int lev = 0; lev < num_levs_; ++lev)
    computeOrientation(img_pyramid_[lev], all_kps[lev], umax_);
}

void OrbExtractor::ComputeKeyPointsOld(
    std::vector<std::vector<KeyPoint> >& all_kps) {
  all_kps.resize(num_levs_);

  float img_ratio = (float)img_pyramid_[0].cols / img_pyramid_[0].rows;

  for (int lev = 0; lev < num_levs_; ++lev) {
    const int nDesiredFeatures = num_feats_per_lev_[lev];

    const int levelCols = sqrt((float)nDesiredFeatures / (5 * img_ratio));
    const int levelRows = img_ratio * levelCols;

    const int min_border_x = kEdgeThreshold;
    const int min_border_y = min_border_x;
    const int max_border_x = img_pyramid_[lev].cols - kEdgeThreshold;
    const int max_border_y = img_pyramid_[lev].rows - kEdgeThreshold;

    const int W = max_border_x - min_border_x;
    const int H = max_border_y - min_border_y;
    const int cellW = ceil((float)W / levelCols);
    const int cellH = ceil((float)H / levelRows);

    const int nCells = levelRows * levelCols;
    const int nfeaturesCell = ceil((float)nDesiredFeatures / nCells);

    vector<vector<vector<KeyPoint> > > cellKeyPoints(
        levelRows, vector<vector<KeyPoint> >(levelCols));

    vector<vector<int> > nToRetain(levelRows, vector<int>(levelCols, 0));
    vector<vector<int> > nTotal(levelRows, vector<int>(levelCols, 0));
    vector<vector<bool> > bNoMore(levelRows, vector<bool>(levelCols, false));
    vector<int> iniXCol(levelCols);
    vector<int> iniYRow(levelRows);
    int nNoMore = 0;
    int nToDistribute = 0;

    float h_y = cellH + 6;

    for (int i = 0; i < levelRows; i++) {
      const float ini_y = min_border_y + i * cellH - 3;
      iniYRow[i] = ini_y;

      if (i == levelRows - 1) {
        h_y = max_border_y + 3 - ini_y;
        if (h_y <= 0) continue;
      }

      float h_x = cellW + 6;

      for (int j = 0; j < levelCols; j++) {
        float ini_x;

        if (i == 0) {
          ini_x = min_border_x + j * cellW - 3;
          iniXCol[j] = ini_x;
        } else {
          ini_x = iniXCol[j];
        }

        if (j == levelCols - 1) {
          h_x = max_border_x + 3 - ini_x;
          if (h_x <= 0) continue;
        }

        Mat cellImage = img_pyramid_[lev]
                            .rowRange(ini_y, ini_y + h_y)
                            .colRange(ini_x, ini_x + h_x);

        cellKeyPoints[i][j].reserve(nfeaturesCell * 5);

        FAST(cellImage, cellKeyPoints[i][j], ini_th_fast_, true);

        if (cellKeyPoints[i][j].size() <= 3) {
          cellKeyPoints[i][j].clear();

          FAST(cellImage, cellKeyPoints[i][j], min_th_fast_, true);
        }

        const int nKeys = cellKeyPoints[i][j].size();
        nTotal[i][j] = nKeys;

        if (nKeys > nfeaturesCell) {
          nToRetain[i][j] = nfeaturesCell;
          bNoMore[i][j] = false;
        } else {
          nToRetain[i][j] = nKeys;
          nToDistribute += nfeaturesCell - nKeys;
          bNoMore[i][j] = true;
          nNoMore++;
        }
      }
    }

    // Retain by score

    while (nToDistribute > 0 && nNoMore < nCells) {
      int nNewFeaturesCell =
          nfeaturesCell + ceil((float)nToDistribute / (nCells - nNoMore));
      nToDistribute = 0;

      for (int i = 0; i < levelRows; i++) {
        for (int j = 0; j < levelCols; j++) {
          if (!bNoMore[i][j]) {
            if (nTotal[i][j] > nNewFeaturesCell) {
              nToRetain[i][j] = nNewFeaturesCell;
              bNoMore[i][j] = false;
            } else {
              nToRetain[i][j] = nTotal[i][j];
              nToDistribute += nNewFeaturesCell - nTotal[i][j];
              bNoMore[i][j] = true;
              nNoMore++;
            }
          }
        }
      }
    }

    vector<KeyPoint>& kps = all_kps[lev];
    kps.reserve(nDesiredFeatures * 2);

    const int scaled_patch_size = kPatchSize * scale_factors_[lev];

    // Retain by score and transform coordinates
    for (int i = 0; i < levelRows; i++) {
      for (int j = 0; j < levelCols; j++) {
        vector<KeyPoint>& keysCell = cellKeyPoints[i][j];
        KeyPointsFilter::retainBest(keysCell, nToRetain[i][j]);
        if ((int)keysCell.size() > nToRetain[i][j])
          keysCell.resize(nToRetain[i][j]);

        for (size_t k = 0, kend = keysCell.size(); k < kend; k++) {
          keysCell[k].pt.x += iniXCol[j];
          keysCell[k].pt.y += iniYRow[i];
          keysCell[k].octave = lev;
          keysCell[k].size = scaled_patch_size;
          kps.push_back(keysCell[k]);
        }
      }
    }

    if ((int)kps.size() > nDesiredFeatures) {
      KeyPointsFilter::retainBest(kps, nDesiredFeatures);
      kps.resize(nDesiredFeatures);
    }
  }

  // and compute orientations
  for (int lev = 0; lev < num_levs_; ++lev)
    computeOrientation(img_pyramid_[lev], all_kps[lev], umax_);
}

static void computeDescriptors(const Mat& image, vector<KeyPoint>& kps,
                               Mat& descriptors, const vector<Point>& patterns_) {
  descriptors = Mat::zeros((int)kps.size(), 32, CV_8UC1);

  for (size_t i = 0; i < kps.size(); i++)
    computeOrbDescriptor(kps[i], image, &patterns_[0],
                         descriptors.ptr((int)i));
}

int OrbExtractor::operator()(InputArray img, InputArray msk,
                             vector<KeyPoint>& kps,
                             OutputArray descs,
                             std::vector<int>& lapping_areas) {
  // cout << "[OrbExtractor]: Max Features: " << num_feats_ << endl;
  if (img.empty()) return -1;

  Mat image = img.getMat();
  assert(image.type() == CV_8UC1);

  // Pre-compute the scale pyramid
  ComputePyramid(image);

  vector<vector<KeyPoint> > all_kps;
  ComputeKeyPointsOctTree(all_kps);
  // ComputeKeyPointsOld(all_kps);

  Mat descriptors;

  int nkeypoints = 0;
  for (int lev = 0; lev < num_levs_; ++lev)
    nkeypoints += (int)all_kps[lev].size();
  if (nkeypoints == 0)
    descs.release();
  else {
    descs.create(nkeypoints, 32, CV_8U);
    descriptors = descs.getMat();
  }

  //kps.clear();
  //kps.reserve(nkeypoints);
  kps = vector<cv::KeyPoint>(nkeypoints);

  int offset = 0;
  // Modified for speeding up stereo fisheye matching
  int monoIndex = 0, stereoIndex = nkeypoints - 1;
  for (int lev = 0; lev < num_levs_; ++lev) {
    vector<KeyPoint>& kps = all_kps[lev];
    int nkeypointsLevel = (int)kps.size();

    if (nkeypointsLevel == 0) continue;

    // preprocess the resized image
    Mat workingMat = img_pyramid_[lev].clone();
    GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

    // Compute the descriptors
    // Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
    Mat desc = cv::Mat(nkeypointsLevel, 32, CV_8U);
    computeDescriptors(workingMat, kps, desc, patterns_);

    offset += nkeypointsLevel;

    float scale =
        scale_factors_[lev];  // getScale(lev, firstLevel, scale_factor_);
    int i = 0;
    for (vector<KeyPoint>::iterator keypoint = kps.begin(),
                                    keypointEnd = kps.end();
         keypoint != keypointEnd; ++keypoint) {
      // Scale keypoint coordinates
      if (lev != 0) {
        keypoint->pt *= scale;
      }

      if (keypoint->pt.x >= lapping_areas[0] &&
          keypoint->pt.x <= lapping_areas[1]) {
        kps.at(stereoIndex) = (*keypoint);
        desc.row(i).copyTo(descriptors.row(stereoIndex));
        stereoIndex--;
      } else {
        kps.at(monoIndex) = (*keypoint);
        desc.row(i).copyTo(descriptors.row(monoIndex));
        monoIndex++;
      }
      i++;
    }
  }
  // cout << "[OrbExtractor]: extracted " << kps.size() << " KeyPoints"
  // << endl;
  return monoIndex;
}

void OrbExtractor::ComputePyramid(cv::Mat img) {
  for (int lev = 0; lev < num_levs_; ++lev) {
    float scale = inv_scale_factors_[lev];
    Size sz(cvRound((float)img.cols * scale), cvRound((float)img.rows * scale));
    Size wholeSize(sz.width + kEdgeThreshold * 2,
                   sz.height + kEdgeThreshold * 2);
    // NOTE: cv::Mat
    Mat tmp(wholeSize, img.type());
    img_pyramid_[lev] =
        tmp(Rect(kEdgeThreshold, kEdgeThreshold, sz.width, sz.height));

    // Compute the resized image
    if (lev != 0) {
      resize(img_pyramid_[lev - 1], img_pyramid_[lev], sz, 0, 0,
             INTER_LINEAR);

      copyMakeBorder(img_pyramid_[lev], tmp, kEdgeThreshold, kEdgeThreshold,
                     kEdgeThreshold, kEdgeThreshold,
                     BORDER_REFLECT_101 + BORDER_ISOLATED);
    } else {
      copyMakeBorder(img, tmp, kEdgeThreshold, kEdgeThreshold, kEdgeThreshold,
                     kEdgeThreshold, BORDER_REFLECT_101);
    }
  }
}

}  // namespace ORB_SLAM_FUSION
