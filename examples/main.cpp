// 시스템 헤더
#include <iostream>

// 서드 파티 헤더
#include "opencv2/opencv.hpp"

// 사용자 정의 헤더
#include "LaneDetection/LaneDetector.hpp"
#include "LaneDetection/PreProcessor.hpp"

int main()
{
    cv::VideoCapture cap("../examples/Sub_project.avi");
    if(!cap.isOpened())
    {
        std::cerr << "Camera open failed!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while(true)
    {
        cap >> frame;
        if(frame.empty())
        {
            std::cerr << "Frame empty" << std::endl;
            break;
        }

        cv::imshow("frame", frame);
        if(cv::waitKey(10) == 'q')
            break;
    }
    cap.release();
    cv::destroyAllWindows();

    return 0;
}