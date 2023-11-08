// 시스템 헤더
#include <iostream>
#include <ostream>
#include <vector>

// 서드 파티 헤더
#include "opencv2/core.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"

// 사용자 정의 헤더
#include "LaneDetection/LaneDetector.hpp"

cv::Mat equal_gray, stretch_gray;
cv::Mat dst_equal_gray, dst_stretch_gray;
cv::Mat equal_hls, stretch_hls;
cv::Mat dst_equal_hls, dst_stretch_hls;
int t_value = 128;

//gray histogram 구하는 함수
cv::Mat calcGrayHist(const cv::Mat& img)
{
    CV_Assert(img.type() == CV_8U);

    cv::Mat hist;
    int channels[] = {0};
    int dims = 1;
    const int histSize[] = {256};
    float graylevel[] = {0,256};
    const float* ranges[] = { graylevel };

    calcHist(&img, 1, channels, cv::noArray(), hist ,dims, histSize, ranges);

    return hist;
}

//histogram 이미지 만들어 주는 함수
cv::Mat getGrayHistImage(const cv::Mat& hist)
{
    CV_Assert(hist.type() == CV_32FC1); //if false -> print
    CV_Assert(hist.size() == cv::Size(1, 256));

    double histMax = 0;
    minMaxLoc(hist, 0, &histMax); //find min, max val //if don't need, set 0

    cv::Mat imgHist(100, 256, CV_8UC1, cv::Scalar(255));
    for(int i = 0; i < 256; i++){
        line(imgHist, cv::Point(i, 100), cv::Point(i, 100 - cvRound(hist.at<float>(i,0) * 100 / histMax)), cv::Scalar(0));
    }

    return imgHist;
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

    cv::Mat frame;
    cv::Mat crop(cv::Size(640, 480), CV_8UC3);
    cv::Mat mask_lidar = cv::imread("../examples/mask.png", CV_8UC1);

    //method 1 trackbar
    cv::namedWindow("dst_equal_gray");
    cv::createTrackbar("eq_gray_threshold", "dst_equal_gray", &t_value, 255);

    //method2 trackbar
    cv::namedWindow("dst_stretch_gray");
    cv::createTrackbar("st_gray_threshold", "dst_stretch_gray", &t_value, 255);

    //method3 trackbar
    cv::namedWindow("dst_equal_hls");
    cv::createTrackbar("eq_hls_threshold", "dst_equal_hls", &t_value, 255);

    //method4 trackbar
    cv::namedWindow("dst_stretch_hls");
    cv::createTrackbar("st_hls_threshold", "dst_stretch_hls", &t_value, 255);


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
// method 1 : gray 영상에서 equalize 수행
        tm.start();
        cv::equalizeHist(crop_gray, equal_gray);

        double eq_gray_th = cv::getTrackbarPos("eq_gray_threshold","dst_equal_gray");
        cv::threshold(equal_gray, dst_equal_gray, eq_gray_th, 255, cv::THRESH_BINARY_INV);

        tm.stop();
        sum_t += tm.getTimeMilli();
        cnt += 1;

        cv::imshow("equal_gray", equal_gray);
        cv::imshow("dst_equal_gray", dst_equal_gray);

        cv::imshow("hist_eq_gray", getGrayHistImage(calcGrayHist(equal_gray)));

#elif 0
//method 2 : gray 영상에서 stretch 수행
        tm.start();
        double gray_minv, gray_maxv;
        cv::minMaxLoc(crop_gray, &gray_minv, &gray_maxv);
        stretch_gray = (crop_gray - gray_minv) * 255 / (gray_maxv - gray_minv);

        double st_gray_th = cv::getTrackbarPos("st_gray_threshold", "dst_stretch_gray");
        cv::threshold(stretch_gray, dst_stretch_gray, st_gray_th, 255, cv::THRESH_BINARY_INV);

        tm.stop();
        sum_t += tm.getTimeMilli();
        cnt += 1;

        cv::imshow("stretch_gray", stretch_gray);
        cv::imshow("dst_stretch_gray", dst_stretch_gray);

        cv::imshow("hist_st_gray", getGrayHistImage(calcGrayHist(stretch_gray)));

#elif 1
//method 3 : HLS 영상의 L로 equalize
    std::vector<cv::Mat> hls_planes;
    cv::split(crop_hls, hls_planes);

    // tm.start();

    cv::equalizeHist(hls_planes[1], equal_hls);

    double eq_hls_th = cv::getTrackbarPos("eq_hls_threshold","dst_equal_hls");
    cv::threshold(equal_hls, dst_equal_hls, eq_hls_th, 255, cv::THRESH_BINARY_INV);

    cv::Mat dst_morpho;
    // cv::erode(dst_equal_hls, dst_morpho, cv::Mat(), cv::Point(-1,-1), 1);
    cv::morphologyEx(dst_equal_hls, dst_morpho, cv::MORPH_OPEN, cv::Mat(), cv::Point(-1,-1), 2);
    imshow("erode", dst_morpho);

    // tm.stop();
    // sum_t += tm.getTimeMilli();
    // cnt += 1;

    cv::imshow("equal_hls", equal_hls);
    cv::imshow("dst_equal_hls", dst_equal_hls);

    // cv::imshow("hist_eq_hls", getGrayHistImage(calcGrayHist(equal_hls)));

#elif 1
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

#else
        cv::equalizeHist(crop_gray, equal_gray);
        cv::imshow("hist_eq_gray", getGrayHistImage(calcGrayHist(equal_gray)));

        double gray_minv, gray_maxv;
        cv::minMaxLoc(crop_gray, &gray_minv, &gray_maxv);
        stretch_gray = (crop_gray - gray_minv) * 255 / (gray_maxv - gray_minv);
        cv::imshow("hist_st_gray",
                   getGrayHistImage(calcGrayHist(stretch_gray)));

        std::vector<cv::Mat> hls_planes;
        cv::split(crop_hls, hls_planes);

        cv::equalizeHist(hls_planes[1], equal_hls);
        cv::imshow("hist_eq_hls", getGrayHistImage(calcGrayHist(equal_hls)));

        double hls_minv, hls_maxv;
        cv::minMaxLoc(hls_planes[1], &hls_minv, &hls_maxv);
        stretch_hls = (hls_planes[1] - hls_minv) * 255 / (hls_maxv - hls_minv);
        cv::imshow("hist_st_hls", getGrayHistImage(calcGrayHist(stretch_hls)));

        // std::cout << "minv: " << gray_minv << ", maxv: " << gray_maxv << std::endl;
        // std::string text_minmax = cv::format("min: %f, max: %f", minv, maxv);
        // cv::putText(hls_planes[0], text_minmax, cv::Point(10,50),cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255), 1, cv::LINE_AA);


        //평균 밝기
        // cv::Scalar avg_h, avg_eh;
        // avg_h = cv::mean(hls_planes[0]) / hls_planes[0].rows * hls_planes[0].cols;
        // avg_eh = cv::sum(equal_H) / equal_H.rows * equal_H.cols;

        // cv::Scalar avg_gray = cv::mean(crop_gray);
        // std::cout << avg_gray[0] << std::endl;

        // cv::putText(hls_planes[0], format("%f", avg_h), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255), 1, cv::LINE_AA);

#endif
        // cv::imshow("frame", frame);

        // cv::imshow("crop_gray", crop_gray);
        // cv::imshow("hist_gray", getGrayHistImage(calcGrayHist(crop_gray)));

        // cv::imshow("crop_hls", hls_planes[1]);
        // cv::imshow("hist_hls", getGrayHistImage(calcGrayHist(hls_planes[1])));

        if(cv::waitKey(10) == 27)   // ESC
            break;
    }
    cap.release();
    cv::destroyAllWindows();

    std::cout << "----------avg_time: "<< sum_t/cnt << std::endl;

    return 0;
}