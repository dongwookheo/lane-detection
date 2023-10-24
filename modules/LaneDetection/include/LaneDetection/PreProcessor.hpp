#ifndef LANE_DETECTION_CLASSPREPROCESSOR_HPP
#define LANE_DETECTION_CLASSPREPROCESSOR_HPP

#include "opencv2/opencv.hpp"

class PreProcessor
{
public:
    PreProcessor() = default;

private:
    cv::Mat cv_mat_;
};

#endif //LANE_DETECTION_CLASSPREPROCESSOR_HPP
