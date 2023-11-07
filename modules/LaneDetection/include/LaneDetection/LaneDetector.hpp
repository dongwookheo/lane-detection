//
// Created by nahye on 23. 11. 7.
//

#ifndef LANE_DETECTION_LANEDETECTOR_HPP
#define LANE_DETECTION_LANEDETECTOR_HPP

#include "opencv2/opencv.hpp"
#include "LaneDetection/Common.hpp"

namespace XyCar
{

class LaneDetector
{
public:
    using Param = std::tuple<PREC, PREC, PREC, PREC>;

    LaneDetector();

    std::pair<int32_t, int32_t> find_pos(const cv::Mat &canny_crop, bool is_refining = true)
    {
        std::vector <cv::Vec4i> lines;
        cv::HoughLinesP(canny_crop, lines, 1, CV_PI / 180, 60, 60, 5);

        evaluate(lines);

        if (is_refining)
            refinePos();

        return { state_.left_pos_, state_.right_pos_ };
    }

private:
    void divideLeftRightLine(const std::vector<cv::Vec4i> &lines, std::vector <cv::Vec4i> &left_lines, std::vector <cv::Vec4i> &right_lines);

    void calculateSlopeAndIntercept(const std::vector <cv::Vec4i> &lines, bool is_left = true);

    void calculatePos(bool is_left = true);

    void refinePos();

    void evaluate(const std::vector <cv::Vec4i> &lines)
    {
        std::vector <cv::Vec4i> left_lines, right_lines;
        divideLeftRightLine(lines, left_lines, right_lines);

        calculateSlopeAndIntercept(left_lines);
        calculateSlopeAndIntercept(right_lines, false);
        kalman_.predict();

        // calculate lpos, rpos
        calculatePos();
        calculatePos(false);
        kalman_.update();
    }

    KalmanFilter kalman_;
    State state_;
};
}

#endif //TEP_LANEDETECTOR_HPP