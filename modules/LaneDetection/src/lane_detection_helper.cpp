#include "LaneDetection/lane_detection_helper.h"

/* @details  Divide 'lines' into 'left_lines' and 'right_lines' based on slope.
* @param[in]  lines  Coordinates consisting of starting and ending points (x, y).
* @param[out]  left_lines  Coordinates of left lines consisting of starting and ending points (x, y).
* @param[out]  right_lines  Coordinates of right lines consisting of starting and ending points (x, y).
* @return  void
*/
void divideLeftRightLine(std::vector<cv::Vec4i>& lines, std::vector<cv::Vec4i>& left_lines, std::vector<cv::Vec4i>& right_lines)
{
    double low_slope_threshold = 0.1;

    for(cv::Vec4i line : lines)
    {
        int x1 = line[0]; int y1 = line[1];
        int x2 = line[2]; int y2 = line[3];

        if(x2 - x1 == 0)
            continue;

        double slope = static_cast<double>(y2 - y1) / (x2 - x1);

        if((slope < -low_slope_threshold) && (x1 < k_frame_width / 2))
            left_lines.push_back({x1,y1,x2,y2});

        else if((slope > low_slope_threshold) && (x2 > k_frame_width /2))
            right_lines.push_back({x1,y1,x2,y2});
    }
}

/* @details  Calculates the slope and intercept of 'lines',
*           and returns an estimated lane calculated by weighted average.
* @param[in]  lines  Coordinates consisting of starting and ending points (x, y).
* @param[out]  average_slope  Slope of a lane calculated by weighted average.
* @param[out]  average_intercept  Intercept of a lane calculated by weighted average.
* @return  void
*/
void calculateSlopeAndIntercept(const std::vector<cv::Vec4i> lines, double& average_slope, double& average_intercept)
{
    double length_sum = 0.0, slope_sum = 0.0, intercept_sum = 0.0;

    for(cv::Vec4i line : lines)
    {
        int x1 = line[0]; int y1 = line[1];
        int x2 = line[2]; int y2 = line[3];

        double slope = static_cast<double>(y2 - y1) / (x2 - x1);
        double intercept = y1 + k_roi_frame_height - slope * x1;
        double line_length = sqrt((y2-y1)*(y2-y1) + (x2-x1)*(x2-x1));

        length_sum += line_length;
        slope_sum += slope * line_length;
        intercept_sum += intercept * line_length;
    }

    if(cvRound(length_sum) != 0)
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
void drawLines(cv::Mat& frame, const double& slope, const double& intercept, const cv::Scalar& color)
{
    int y1 = k_frame_height;
    int y2 = cvRound(y1>>1);
    int x1 = cvRound((y1 - intercept) / slope);
    int x2 = cvRound((y2 - intercept) / slope);
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
void calculatePos(const double& slope, const double& intercept, int32_t& pos, bool left = false, bool right = false)
{
    if(cvRound(slope) == 0 && cvRound(intercept) == 0){
        if (left)
            pos = -1;
        else if (right)
            pos = k_frame_width << 1;
    }
    else{
        pos = static_cast<int32_t>((k_offset - intercept) / slope);
        if (left){
            if (pos < 0) pos = 0;
        }
        if (right){
            if(pos > k_frame_width) pos = k_frame_width;
        }
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
void estimatePos(double& left_slope, double& left_intercept, double& right_slope, double& right_intercept, int32_t& lpos, int32_t& rpos)
{
    if(lpos < 0){
        if((rpos <= k_frame_width) && (0.6 < abs(right_slope)) && (abs(right_slope) < 1)){
            lpos = rpos - k_lane_width;
            left_slope = -right_slope;
            left_intercept = k_offset - left_slope * lpos;
        }
        else {
            lpos = 0;
        }
    }
    else if(rpos > k_frame_width){
        if((lpos >= 0) && (0.6 < abs(left_slope)) && (abs(left_slope) < 1)){
            rpos = lpos + k_lane_width;
            right_slope = -left_slope;
            right_intercept = k_offset - right_slope * rpos;
        }
        else {
            rpos = k_frame_width;
        }
    }
}
