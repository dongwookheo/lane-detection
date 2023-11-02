// system header
#include <cassert>
#include <cmath>
#include <iostream>
#include <fstream>

// third party header
#include "opencv2/opencv.hpp"

// user defined header
#include "LaneDetection/LaneDetector.hpp"
#include "LaneDetection/PreProcessor.hpp"
#include "LaneDetection/lane_detection_helper.h"

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
    csvfile << "count, frame, lpos, rpos \n";

    cv::uint32_t count_frame = 0;
    cv::Mat frame;
    cv::Mat cropped_frame(cv::Size(k_frame_width, k_frame_height), CV_8UC3);
    cv::Mat mask_lidar = cv::imread("../examples/mask.png", CV_8UC1);

    int32_t lpos = 0, rpos = k_frame_width;

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
        cropped_frame = cropped_frame(cv::Rect(0, k_roi_frame_height, k_frame_width, k_frame_height - k_roi_frame_height));

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
        double left_average_slope = 0.0;
        double left_average_intercept = 0.0;
        double right_average_slope = 0.0;
        double right_average_intercept = 0.0;
        calculateSlopeAndIntercept(left_lines, left_average_slope, left_average_intercept);
        calculateSlopeAndIntercept(right_lines, right_average_slope, right_average_intercept);

        drawLines(frame, left_average_slope, left_average_intercept, cv::Scalar(0, 0, 255));
        drawLines(frame, right_average_slope, right_average_intercept, cv::Scalar(0, 0, 255));

        // calculate lpos, rpos
        calculatePos(lpos, left_average_slope, left_average_intercept, true);
        calculatePos(rpos, right_average_slope, right_average_intercept, false);

        refinePos(left_average_slope, left_average_intercept, right_average_slope, right_average_intercept, lpos, rpos);

        drawLines(frame, left_average_slope, left_average_intercept, cv::Scalar(255, 0, 0));
        drawLines(frame, right_average_slope, right_average_intercept, cv::Scalar(255, 0, 0));

        cv::rectangle(frame, cv::Rect(cv::Point(lpos-5, 395),cv::Point(lpos+5, 405)), cv::Scalar(0, 255, 0));
        cv::rectangle(frame, cv::Rect(cv::Point(rpos-5, 395),cv::Point(rpos+5, 405)), cv::Scalar(0, 255, 0));

        // save csv file
        if (count_frame % 30 == 0)
            csvfile << (count_frame / 30 - 1) << ","<< count_frame << "," <<lpos << "," << rpos << "\n";

        // 기준 line
        cv::line(frame, cv::Point(0, k_offset), cv::Point(k_frame_width, k_offset), cv::Scalar(0, 255, 0), 1, cv::LINE_4);

        cv::imshow("frame", frame);

        if(cv::waitKey(1) == 27)   // ESC
            break;
    }
    cap.release();
    cv::destroyAllWindows();
    csvfile.close();

    return 0;
}