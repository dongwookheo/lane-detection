#include "LaneDetection/lane_detection_helper.h"

/* @details  Divide 'lines' into 'left_lines' and 'right_lines' based on slope.
* @param[in]  lines  Coordinates consisting of starting and ending points (x, y).
* @param[out]  left_lines  Coordinates of left lines consisting of starting and ending points (x, y).
* @param[out]  right_lines  Coordinates of right lines consisting of starting and ending points (x, y).
* @param[out]  stop_lines  Coordinates of stop lines consisting of starting and ending points (x, y).
* @return  void
*/
void divideLeftRightLine(const std::vector<cv::Vec4i>& lines, std::vector<cv::Vec4i>& left_lines, std::vector<cv::Vec4i>& right_lines, std::vector<cv::Vec4i>& stop_lines)
{
    constexpr double k_low_slope_threshold = 0.1;
    constexpr double k_stop_slpoe_threshold = 0.15;

    constexpr int32_t k_half_frame = k_frame_width / 2;
    constexpr int32_t k_threshold_location = k_frame_width / 5;

    for(cv::Vec4i line : lines)
    {
        int32_t x1 = line[0];
        int32_t y1 = line[1];
        int32_t x2 = line[2];
        int32_t y2 = line[3];

        if(x2 - x1 == 0)
            continue;

        double slope = static_cast<double>(y2 - y1) / (x2 - x1);

        if((slope < -k_low_slope_threshold) && (x1 < k_half_frame))
            left_lines.emplace_back(x1,y1,x2,y2);

        else if((slope > k_low_slope_threshold) && (x2 > k_half_frame))
            right_lines.emplace_back(x1,y1,x2,y2);

        else if((abs(slope) <= k_stop_slpoe_threshold) && (x1 > k_threshold_location) && (x2 < k_threshold_location * 4))
            stop_lines.emplace_back(x1,y1,x2,y2);
    }
}

/* @details  Find the stop line.
* @param[in]  stop_lines  Coordinates of stop lines consisting of starting and ending points (x, y).
* @return  bool The flag of stop.
*/
bool findStopLine(const std::vector<cv::Vec4i> &stoplines)
{
    if(stoplines.size() >= 2)
        return true;
    else
        return false;
}

/* @details  Calculates the slope and intercept of 'lines',
*           and returns an estimated lane calculated by weighted average.
* @param[in]  lines  Coordinates consisting of starting and ending points (x, y).
* @param[out]  average_slope  Slope of a lane calculated by weighted average.
* @param[out]  average_intercept  Intercept of a lane calculated by weighted average.
* @return  void
*/
void calculateSlopeAndIntercept(const std::vector<cv::Vec4i>& lines, double& average_slope, double& average_intercept)
{
    double length_sum = 0.0, slope_sum = 0.0, intercept_sum = 0.0;

    for(const cv::Vec4i line : lines)
    {
        int32_t x1 = line[0];
        int32_t y1 = line[1];
        int32_t x2 = line[2];
        int32_t y2 = line[3];

        if(x2 - x1 == 0)
            continue;

        int32_t diff_y = y2 - y1;
        int32_t diff_x = x2 - x1;
        double slope = static_cast<double>(diff_y) / (diff_x);
        double intercept = y1 + k_roi_frame_height - slope * x1;
        double line_length = sqrt(diff_y * diff_y) + (diff_x * diff_x);

        length_sum += line_length;
        slope_sum += slope * line_length;
        intercept_sum += intercept * line_length;
    }

    if(std::round(length_sum) != 0)
    {
        average_slope = slope_sum / length_sum;
        average_intercept = intercept_sum / length_sum;
    }
}

/* @details  Draw a lane on 'frame'.
* @param[out] frame
* @param[in]  slope  The slope of a lane.
* @param[in]  intercept  The intercept of a lane.
* @param[in]  color  The color of lane.
* @return  void
*/
void drawLines(cv::Mat& frame, double slope, double intercept, const cv::Scalar& color)
{
    if(slope == 0) return;
    int32_t y1 = k_frame_height;
    int32_t y2 = std::round(y1>>1);
    int32_t x1 = std::round((y1 - intercept) / slope);
    int32_t x2 = std::round((y2 - intercept) / slope);
    cv::line(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 2, cv::LINE_8);
}

/* @details  Do exception handling to lane position('pos').
*           Using the 'slope' and 'intercept'.
* @param[out]  pos  The position of lane (x coordinate).
* @param[in]  slope  The slope of a lane.
* @param[in]  intercept  The intercept of a lane.
* @param[in]  is_left  The flag for left lane.
* @return  void
*/
void calculatePos(int32_t& pos, double slope, double intercept, bool is_left)
{
    if(std::round(slope) == 0 && std::round(intercept) == 0){
        if (is_left)
            pos = -1;
        else
            pos = k_frame_width + 1;
    }
    else{
        pos = static_cast<int32_t>((k_offset - intercept)/ slope);
        if (is_left && (pos < 0))
            pos = 0;
        if ((!is_left) && (pos > k_frame_width))
            pos = k_frame_width;
    }
}

/* @details  Estimate left and right lanes based on exception handling.
* @param[in, out]  left_slope  The slope of a left lane.
* @param[out]  left_intercept  The intercept of a left lane.
* @param[in, out]  right_slope  The slope of a right lane.
* @param[out]  right_intercept  The intercept of a right lane.
* @param[in]  lpos  The x coordinate of left lane.
* @param[in]  rpos  The x coordinate of right lane.
* @return  void
*/
void refinePos(double& left_slope, double& left_intercept, double& right_slope, double& right_intercept, int32_t& lpos, int32_t& rpos)
{

    constexpr double k_under_limit = 0.6;
    constexpr double k_upper_limit = 1.0;

    if(lpos < 0){
        if((rpos <= k_frame_width) && (k_under_limit < abs(right_slope)) && (abs(right_slope) < k_upper_limit)){
            lpos = rpos - k_lane_width;
            left_slope = -right_slope;
            left_intercept = k_offset - left_slope * lpos;
            if(lpos < 0) lpos = 0;
        }
        else {
            lpos = 0;
        }
    }
    else if(rpos > k_frame_width){
        if((lpos >= 0) && (k_under_limit < abs(left_slope)) && (abs(left_slope) < k_upper_limit)){
            rpos = lpos + k_lane_width;
            right_slope = -left_slope;
            right_intercept = k_offset - right_slope * rpos;
            if(rpos > k_frame_width) rpos = k_frame_width;
        }
        else {
            rpos = k_frame_width;
        }
    }
}

/* @details  Calculate difference of centor frame and pos frame.
* @param[out]  centor_pos  The centor of lpos and rpos.
* @param[in]  lpos  The x coordinate of left lane.
* @param[in]  rpos  The x coordinate of right lane.
* @return  error The difference of centor frame and centor_pos
*/
int32_t calculateError(int32_t &centor_pos, int32_t lpos, int32_t rpos)
{
    centor_pos = static_cast<int32_t>((rpos + lpos) / 2);

    int32_t error = k_frame_width / 2 - centor_pos;

    // std::cout << error << std::endl;

    return error;
}
