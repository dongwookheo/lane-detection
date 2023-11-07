#ifndef LANE_DETECTION__LANE_DETECTION_HELPER_H
#define LANE_DETECTION__LANE_DETECTION_HELPER_H

#include <vector>
#include "opencv2/opencv.hpp"

// global variables
namespace {
    constexpr uint32_t k_frame_width = 640;
    constexpr uint32_t k_frame_height = 480;
    constexpr uint32_t k_roi_frame_height = (k_frame_height>>3)*5;
    constexpr uint32_t k_lane_width = 490;
    constexpr uint32_t k_offset = 400;
}

void divideLeftRightLine(const std::vector<cv::Vec4i>& lines, std::vector<cv::Vec4i>& left_lines, std::vector<cv::Vec4i>& right_lines);
void calculateSlopeAndIntercept(const std::vector<cv::Vec4i>& lines, double& average_slope, double& average_intercept);
void drawLines(cv::Mat& frame, double slope, double intercept, const cv::Scalar& color);
void calculatePos(int32_t& pos, double slope, double intercept, bool is_left);
void refinePos(double& left_slope, double& left_intercept, double& right_slope, double& right_intercept, int32_t& lpos, int32_t& rpos);
int32_t calculateError(int32_t &centor_pos, int32_t lpos, int32_t rpos);

#endif //LANE_DETECTION__LANE_DETECTION_HELPER_H
