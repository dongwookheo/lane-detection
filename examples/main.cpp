// 시스템 헤더
#include <iostream>

// 서드 파티 헤더
#include "opencv2/opencv.hpp"

// 사용자 정의 헤더
#include "LaneDetection/LaneDetector.hpp"
#include "LaneDetection/PreProcessor.hpp"

int main()
{
    cv::String file_path = "../examples/Sub_project.avi";
    cv::VideoCapture cap(file_path);
    if(!cap.isOpened())
    {
        std::cerr << "Camera open failed!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    cv::Mat crop(cv::Size(640, 480), CV_8UC3);
    cv::Mat mask_lidar = cv::imread("../examples/mask.png", CV_8UC1);
    while(true)
    {
        cap >> frame;
        if(frame.empty())
        {
            std::cerr << "Frame empty" << std::endl;
            break;
        }

        frame.copyTo(crop, mask_lidar);
        crop = crop(cv::Rect(0, frame.rows>>1, frame.cols, frame.rows>>1));
        cv::cvtColor(crop, crop, cv::COLOR_BGR2HLS);

        cv::imshow("frame", frame);
        cv::imshow("crop", crop);
        if(cv::waitKey(10) == 27)   // ESC
            break;
    }
    cap.release();
    cv::destroyAllWindows();

    return 0;
}