#ifndef LANE_DETECTION_CLASSLANEDETECTION_HPP
#define LANE_DETECTION_CLASSLANEDETECTION_HPP

#include "opencv2/opencv.hpp"

class LaneDetection
{
public:
    LaneDetection() = default;

private:
    cv::Mat cv_mat_;
};

#endif //LANE_DETECTION_CLASSLANEDETECTION_HPP
