// 시스템 헤더
#include <iostream>
#include <fstream>

// 서드 파티 헤더
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"

// 사용자 정의 헤더
#include "LaneDetection/LaneDetector.hpp"
#include "LaneDetection/PreProcessor.hpp"

// 전역 변수
namespace {
    constexpr uint32_t frame_width = 640, frame_height = 480;
    constexpr uint32_t roi_frame_height = (frame_height>>3)*5;
    constexpr uint32_t lane_width = 490;
    constexpr uint32_t offset = 400;
}

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

        if((slope < -low_slope_threshold) && (x1 < frame_width / 2))
            left_lines.push_back({x1,y1,x2,y2});

        else if((slope > low_slope_threshold) && (x2 > frame_width /2))
            right_lines.push_back({x1,y1,x2,y2});
    }
}

void calculateSlopeAndIntercept(const std::vector<cv::Vec4i> lines, double& average_slope, double& average_intercept)
{
    double length_sum = 0.0, slope_sum = 0.0, intercept_sum = 0.0;

    for(cv::Vec4i line : lines)
    {
        int x1 = line[0]; int y1 = line[1];
        int x2 = line[2]; int y2 = line[3];

        double slope = static_cast<double>(y2 - y1) / (x2 - x1);
        double intercept = y1 + roi_frame_height - slope * x1;
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

void drawLines(cv::Mat& frame, const double& slope, const double& intercept, const cv::Scalar& color)
{
    int y1 = frame_height;
    int y2 = cvRound(y1>>1);
    int x1 = cvRound((y1 - intercept) / slope);
    int x2 = cvRound((y2 - intercept) / slope);
    cv::line(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 2, cv::LINE_8);
}

void calculatePos(const double& slope, const double& intercept, int32_t& pos, bool left = false, bool right = false)
{
    if(cvRound(slope) == 0 && cvRound(intercept) == 0){
        if (left)
            pos = -1;
        else if (right)
            pos = frame_width << 1;
    }
    else{
        pos = static_cast<int32_t>((offset - intercept)/ slope);
        if (left){
            if (pos < 0) pos = 0;
        }
        if (right){
            if(pos > frame_width) pos = frame_width;
        }
    }
}

void estimatePos(double& left_slope, double& left_intercept, double& right_slope, double& right_intercept, int32_t& lpos, int32_t& rpos)
{
    if(lpos < 0){
        if((rpos <= frame_width) && (0.6 < abs(right_slope)) && (abs(right_slope) < 1)){
            lpos = rpos - lane_width;
            left_slope = -right_slope;
            left_intercept = offset - left_slope * lpos;
        }
        else {
            lpos = 0;
        }
    }
    else if(rpos > frame_width){
        if((lpos >= 0) && (0.6 < abs(left_slope)) && (abs(left_slope) < 1)){
            rpos = lpos + lane_width;
            right_slope = -left_slope;
            right_intercept = offset - right_slope * rpos;
        }
        else {
            rpos = frame_width;
        }
    }
}

int main()
{
    cv::String file_path = "../examples/Sub_project.avi";
    cv::VideoCapture cap(file_path);
    if(!cap.isOpened())
    {
        std::cerr << "Camera open failed!" << std::endl;
        return -1;
    }

    std::ofstream csvfile("../result/result.csv");
    csvfile << "lpos, rpos \n";

    cv::uint32_t count_frame = 0;
    cv::Mat frame;
    cv::Mat cropped_frame(cv::Size(frame_width, frame_height), CV_8UC3);
    cv::Mat mask_lidar = cv::imread("../examples/mask.png", CV_8UC1);

    int32_t lpos = 0, rpos = frame_width;

    while(true)
    {
        cap >> frame;
        if(frame.empty())
        {
            std::cerr << "Frame empty" << std::endl;
            break;
        }

        ++count_frame;

        frame.copyTo(cropped_frame);
        cropped_frame = cropped_frame(cv::Rect(0, roi_frame_height, frame_width, frame_height - roi_frame_height));

        // gray image
        cv::cvtColor(cropped_frame, cropped_frame, cv::COLOR_BGR2GRAY);

        // binarization
        cv::equalizeHist(cropped_frame, cropped_frame);
        cv::threshold(cropped_frame, cropped_frame, 65, 255, cv::THRESH_BINARY_INV);

        // lidar mask
        cv::bitwise_and(cropped_frame, mask_lidar(cv::Rect(0,(mask_lidar.rows>>3)*5,mask_lidar.cols,(mask_lidar.rows>>3)*3)), cropped_frame);

        // blur (gaussian)
        cv::GaussianBlur(cropped_frame, cropped_frame, cv::Size(), 5);

        // canny edge
        cv::Mat canny_crop;
        cv::Canny(cropped_frame, canny_crop, 50, 150);

        // houghLinesP
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(canny_crop, lines, 1, CV_PI/180, 60, 60, 5);

        // divide left, right lines
        std::vector<cv::Vec4i> left_lines, right_lines;
        divideLeftRightLine(lines, left_lines, right_lines);

        // calculate slop and intercept using weighted average
        double left_average_slope = 0.0, left_average_intercept = 0.0;
        double right_average_slope = 0.0, right_average_intercept = 0.0;
        calculateSlopeAndIntercept(left_lines, left_average_slope, left_average_intercept);
        calculateSlopeAndIntercept(right_lines, right_average_slope, right_average_intercept);

        drawLines(frame, left_average_slope, left_average_intercept, cv::Scalar(0, 0, 255));
        drawLines(frame, right_average_slope, right_average_intercept, cv::Scalar(0, 0, 255));

        // calculate lpos, rpos
        calculatePos(left_average_slope, left_average_intercept, lpos, true, false);
        calculatePos(right_average_slope, right_average_intercept, rpos, false, true);

        estimatePos(left_average_slope, left_average_intercept, right_average_slope, right_average_intercept, lpos, rpos);

        drawLines(frame, left_average_slope, left_average_intercept, cv::Scalar(255, 0, 0));
        drawLines(frame, right_average_slope, right_average_intercept, cv::Scalar(255, 0, 0));

        cv::rectangle(frame, cv::Rect(cv::Point(lpos-5, 395),cv::Point(lpos+5, 405)), cv::Scalar(0, 255, 0));
        cv::rectangle(frame, cv::Rect(cv::Point(rpos-5, 395),cv::Point(rpos+5, 405)), cv::Scalar(0, 255, 0));

        // save csv file
        if (count_frame % 30 == 0)
            csvfile << lpos << "," << rpos << "\n";

        // 기준 line
        cv::line(frame, cv::Point(0,offset), cv::Point(frame_width,offset), cv::Scalar(0,255,0), 1, cv::LINE_4);

        cv::imshow("frame", frame);

        if(cv::waitKey(1) == 27)   // ESC
            break;
    }
    cap.release();
    cv::destroyAllWindows();
    csvfile.close();

    return 0;
}