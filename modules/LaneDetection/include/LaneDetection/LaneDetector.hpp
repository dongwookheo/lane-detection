//
// Created by nahye on 23. 11. 7.
//

#ifndef LANE_DETECTION__LANEDETECTOR_HPP
#define LANE_DETECTION__LANEDETECTOR_HPP

#include "opencv2/opencv.hpp"
#include "LaneDetection/Common.hpp"

namespace XyCar
{
/**
 * @details Class responsible for detecting lanes in an image.
 */
class LaneDetector
{
public:
    using Param = std::tuple<PREC, PREC, PREC, PREC>;

    LaneDetector();

    /**
     * @details Estimate the lane from the edge image and return the coordinates of the left & right lanes.
     * @param[in] canny_crop Canny edge ROI image.
     * @param[in] is_refining Flag about whether to refine position of lane.
     * @return std::pair<int32_t, int32_t>
     */
    std::pair<int32_t, int32_t> find_pos(const cv::Mat& canny_crop, bool is_refining = true)
    {
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(canny_crop, lines, 1, CV_PI / 180, 60, 60, 5);

        evaluate(lines);

        if (is_refining)
            refinePos();

        return { state_.left_pos_, state_.right_pos_ };
    }

private:
    /**
     * @details Divide 'lines' into 'left_lines' and 'right_lines' based on slope.
     * @param[in] lines Coordinates consisting of starting and ending points. (x, y)
     * @param[out] left_lines Coordinates of left lines consisting of starting and ending points (x, y).
     * @param[out] right_lines Coordinates of right lines consisting of starting and ending points (x, y).
     * @return void
     */
    void divideLeftRightLine(const std::vector<cv::Vec4i>& lines, std::vector<cv::Vec4i>& left_lines, std::vector<cv::Vec4i>& right_lines);

    /**
     * @details Calculates the slope and intercept of 'lines',
     * and returns an estimated lane calculated by weighted average.
     * @param[in] lines Coordinates consisting of starting and ending points (x, y).
     * @param[in] is_left Flag of left lane.
     * @return void
     */
    void calculateSlopeAndIntercept(const std::vector<cv::Vec4i>& lines, bool is_left = true);

    /**
     * @details Do exception handling to lane position('pos'),
     * using the 'slope' and 'intercept' of lanes.
     * @param[in] is_left Flag of left lane.
     * @return void
     */
    void calculatePos(bool is_left = true);

    /**
     * @details Estimate left and right lanes based on exception handling.
     * @return void
     */
    void refinePos();

    /**
     * @details Divide lanes into left & right,
     * and estimate by applying filter (or our algorithm).
     * @param[in] lines Coordinates consisting of starting and ending points (x, y).
     * @return void
     */
    void evaluate(const std::vector<cv::Vec4i>& lines)
    {
        std::vector<cv::Vec4i> left_lines, right_lines;
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

#endif // LANE_DETECTION__LANEDETECTOR_HPP
