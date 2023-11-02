// 시스템 헤더
#include <iostream>
#include <ostream>
#include <vector>

// 서드 파티 헤더
#include "opencv2/core.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
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

        // frame.copyTo(crop, mask_lidar);
        // crop = crop(cv::Rect(0, frame.rows>>1, frame.cols, frame.rows>>1));

        crop = frame(cv::Rect(0, frame.rows>>1, frame.cols, frame.rows>>1)).clone();

        cv::Mat crop_gray, crop_hls;
        cv::cvtColor(crop, crop_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(crop, crop_hls, cv::COLOR_BGR2HLS);

        cv::TickMeter tm;

#if 1
//method 3 : HLS 영상의 L로 equalize
    std::vector<cv::Mat> hls_planes;
    cv::split(crop_hls, hls_planes);

    // tm.start();
    cv::Mat equal_hls, dst_equal_hls;
    cv::equalizeHist(hls_planes[1], equal_hls);
    cv::threshold(equal_hls, dst_equal_hls, 65, 255, cv::THRESH_BINARY_INV);

    cv::Mat dst_morpho;
    // cv::erode(dst_equal_hls, dst_morpho, cv::Mat(), cv::Point(-1,-1), 1);
    cv::morphologyEx(dst_equal_hls, dst_morpho, cv::MORPH_OPEN, cv::Mat(), cv::Point(-1,-1), 1);

    // tm.stop();
    // sum_t += tm.getTimeMilli();
    // cnt += 1;
    //모폴로지를 수행한 후라서 가우시안 블러를 하지 전과 후가 차이가 별로 없다. 일단은.
    // cv::Mat dst_gaussian;
    // cv::GaussianBlur(dst_morpho, dst_gaussian, cv::Size(), 3);

    // 캐니 에지
    cv::Mat canny_crop;
    // cv::Canny(dst_gaussian, canny_crop1, 50, 150);
    cv::Canny(dst_morpho, canny_crop, 50, 150);

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(canny_crop, lines, 1, CV_PI/180, 100, 50, 5);

    for(cv::Vec4i line : lines) {
        cv::line(frame, cv::Point(line[0], line[1]+240), cv::Point(line[2], line[3]+240), cv::Scalar(0,0,255), 2, cv::LINE_8);
    }
    cv::bitwise_and(canny_crop, mask_lidar(cv::Rect(0,mask_lidar.rows/2,mask_lidar.cols,mask_lidar.rows/2)), canny_crop);


    cv::imshow("simple_th", frame);
    // cv::imshow("equal_hls", equal_hls);
    cv::imshow("dst_equal_hls", dst_equal_hls);
    cv::imshow("morpho", dst_morpho);
    // cv::imshow("gauissian", dst_gaussian);
    cv::imshow("canny", canny_crop);

#else
//method 4: HLS 영상의 L로 stretch
    std::vector<cv::Mat> hls_planes;
    cv::split(crop_hls, hls_planes);


    double hls_minv, hls_maxv;
    cv::minMaxLoc(hls_planes[1], &hls_minv, &hls_maxv);

    tm.start();
    stretch_hls = (hls_planes[1] - hls_minv) * 255 / (hls_maxv - hls_minv);

    double st_hls_th = cv::getTrackbarPos("st_hls_threshold", "dst_stretch_hls");
    cv::threshold(stretch_hls, dst_stretch_hls, st_hls_th, 255, cv::THRESH_BINARY_INV);

    tm.stop();
    sum_t += tm.getTimeMilli();
    cnt += 1;

    cv::imshow("stretch_hls", stretch_hls);
    cv::imshow("dst_stretch_hls", dst_stretch_hls);

    cv::imshow("hist_st_hls", getGrayHistImage(calcGrayHist(stretch_hls)));

#endif

        // cv::imshow("frame", frame);

        if(cv::waitKey(10) == 27)   // ESC
            break;
    }
    cap.release();
    cv::destroyAllWindows();

    // std::cout << "----------avg_time: "<< sum_t/cnt << std::endl;

    return 0;
}