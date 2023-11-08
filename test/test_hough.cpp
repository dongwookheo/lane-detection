// 시스템 헤더
#include <iostream>

// 서드 파티 헤더
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"

// 사용자 정의 헤더
#include "LaneDetection/LaneDetector.hpp"


int main()
{
    cv::String file_path = "../examples/Sub_project.avi";
    cv::VideoCapture cap(file_path);
    if(!cap.isOpened())
    {
        std::cerr << "Camera open failed!" << std::endl;
        return -1;
    }

    cv::Mat frame, frame2;
    cv::Mat crop(cv::Size(640, 480), CV_8UC3);
    cv::Mat mask_lidar = cv::imread("../examples/mask.png", CV_8UC1);

    double sum_t = 0.0;
    double cnt = 0.0;

    while(true)
    {
        cap >> frame;
        if(frame.empty())
        {
            std::cerr << "Frame empty" << std::endl;
            break;
        }
        frame2 = frame.clone();

        crop = frame(cv::Rect(0, frame.rows>>1, frame.cols, frame.rows>>1)).clone();

        cv::Mat crop_gray, crop_hls;
        cv::cvtColor(crop, crop_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(crop, crop_hls, cv::COLOR_BGR2HLS);


    //nahye : HLS 영상의 L로 equalize
    std::vector<cv::Mat> hls_planes;
    cv::split(crop_hls, hls_planes);

    // tm.start();
    cv::Mat equal_hls, dst_equal_hls;
    cv::equalizeHist(hls_planes[1], equal_hls);
    cv::threshold(equal_hls, dst_equal_hls, 65, 255, cv::THRESH_BINARY_INV);

    cv::Mat dst_morpho;
    cv::morphologyEx(dst_equal_hls, dst_morpho, cv::MORPH_OPEN, cv::Mat(), cv::Point(-1,-1), 1);

    cv::Mat canny_crop;
    cv::Canny(dst_morpho, canny_crop, 50, 150);

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(canny_crop, lines, 1, CV_PI/180, 100, 50, 5);

    for(cv::Vec4i line : lines) {
        cv::line(frame, cv::Point(line[0], line[1]+240), cv::Point(line[2], line[3]+240), cv::Scalar(0,0,255), 2, cv::LINE_8);
    }
    cv::bitwise_and(canny_crop, mask_lidar(cv::Rect(0,mask_lidar.rows/2,mask_lidar.cols,mask_lidar.rows/2)), canny_crop);

    cv::imshow("simple_th", frame);
    // cv::imshow("equal_hls", equal_hls);
    // cv::imshow("dst_equal_hls", dst_equal_hls);
    // cv::imshow("morpho", dst_morpho);
    // cv::imshow("gauissian", dst_gaussian);
    // cv::imshow("canny", canny_crop);

    //dongwook
    cv::adaptiveThreshold(crop_gray, crop_gray, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 39, 13);
    cv::GaussianBlur(crop_gray, crop_gray, cv::Size(), 3);

    cv::Mat canny_crop2;
    cv::Canny(crop_gray, canny_crop2, 50, 150);

    std::vector<cv::Vec4i> lines2;
    cv::HoughLinesP(canny_crop2, lines2, 1, CV_PI/180, 100, 50, 5);

    for(cv::Vec4i line : lines2) {
        cv::line(frame2, cv::Point(line[0], line[1]+240), cv::Point(line[2], line[3]+240), cv::Scalar(0,0,255), 2, cv::LINE_8);
    }

    cv::bitwise_and(canny_crop2, mask_lidar(cv::Rect(0,mask_lidar.rows/2,mask_lidar.cols,mask_lidar.rows/2)), canny_crop2);

    cv::imshow("adaptive_th", frame2);

        if(cv::waitKey(10) == 27)   // ESC
            break;
    }
    cap.release();
    cv::destroyAllWindows();

    return 0;
}