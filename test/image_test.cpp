// 시스템 헤더
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

// 서드 파티 헤더
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"

// 사용자 정의 헤더
#include "LaneDetection/LaneDetector.hpp"
#include "LaneDetection/PreProcessor.hpp"

// 전역 변수
namespace {
    double left_estimation_slope = 0.0;
    double left_estimation_intercept = 0.0;
    double right_estimation_slope = 0.0;
    double right_estimation_intercept = 0.0;
    // alpha: 0.1 ~ 0.9: 0.1 에 가까울수록 현재 값을 더 잘 반영
    float alpha = 0.1;
}

int main()
{
    cv::String file_path = "../examples/line_images/img4.png";

    cv::Mat frame, frame2, frame3;
    frame = cv::imread(file_path);
    cv::Mat crop(cv::Size(640, 480), CV_8UC3);
    cv::Mat mask_lidar = cv::imread("../examples/mask.png", CV_8UC1);

    if(frame.empty())
    {
        std::cerr << "image empty" << std::endl;
        return -1;
    }

    frame2 = frame.clone();
    frame3 = frame.clone();
    // ++count_frame;
    frame.copyTo(crop);
    crop = crop(cv::Rect(0, (frame.rows>>3)*5, frame.cols, (frame.rows>>3)*3));

    cv::cvtColor(crop, crop, cv::COLOR_BGR2GRAY);

    // tm.start();
    cv::equalizeHist(crop, crop);

    cv::threshold(crop, crop, 65, 255, cv::THRESH_BINARY_INV);

    // tm.stop();
    // std::cout << tm.getTimeMilli() << "ms. " << std::endl;
    // tm.reset();

    cv::bitwise_and(crop, mask_lidar(cv::Rect(0,(mask_lidar.rows>>3)*5,mask_lidar.cols,(mask_lidar.rows>>3)*3)), crop);

    // 가우시안 블러
    cv::GaussianBlur(crop, crop, cv::Size(), 5);

    // 캐니 에지
    cv::Mat canny_crop;
    cv::Canny(crop, canny_crop, 50, 150);

    std::vector<cv::Vec4i> lines;
    // cv::HoughLinesP(canny_crop, lines, 1, CV_PI/180, 100, 30, 5);

    cv::HoughLinesP(canny_crop, lines, 1, CV_PI/180, 60, 60, 5);
    // cv::HoughLinesP(canny_crop2, lines2, 1, CV_PI/180, 100, 30, 5);
    cv::imshow("hough", canny_crop);


    double total_left_length = 0.0, total_right_length = 0.0;
    double left_slope_sum = 0.0, right_slope_sum = 0.0;
    double left_intercept_sum = 0.0, right_intercept_sum = 0.0;
    std::vector<cv::Vec4i> passed_lines;
    for(cv::Vec4i line : lines)
    {
        int x1 = line[0]; int y1 = line[1];
        int x2 = line[2]; int y2 = line[3];

        // std::cout << "x1: " << x1 << "x2:  " << x2 << std::endl;
        // std::cout << "y1: " << y1 << "y2:  " << y2 << std::endl;

        if(x2 - x1 == 0) {
            continue;
        }

        double slope = static_cast<double>(y2 - y1) / (x2 - x1);
        double intercept = y1+(frame.rows>>3)*5 - slope * x1;
        double line_length = sqrt((y2-y1)*(y2-y1) + (x2-x1)*(x2-x1));

        if((slope < 0) && (x1 < frame.cols / 2))
        // if((slope < 0))
        {
            total_left_length += line_length;
            left_slope_sum += slope * line_length;
            left_intercept_sum += intercept * line_length;
            cv::line(frame3, cv::Point(line[0], line[1]+(frame.rows>>3)*5), cv::Point(line[2], line[3]+(frame.rows>>3)*5), cv::Scalar(0,0,255), 2, cv::LINE_8);
            cv::putText(frame3, cv::format("%f", slope), cv::Point(line[0], line[1]+(frame.rows>>3)*5), cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar::all(-1), 1, cv::LINE_AA);
        }
        else if((slope > 0) && (x2 > frame.cols /2))
        // else if((slope > 0) )
        {
            total_right_length += line_length;
            right_slope_sum += slope * line_length;
            right_intercept_sum += intercept * line_length;
            cv::line(frame3, cv::Point(line[0], line[1]+(frame.rows>>3)*5), cv::Point(line[2], line[3]+(frame.rows>>3)*5), cv::Scalar(0,0,255), 2, cv::LINE_8);
            cv::putText(frame3, cv::format("%f", slope), cv::Point(line[0], line[1]+(frame.rows>>3)*5), cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar::all(-1), 1, cv::LINE_AA);

        }


        // if((slope < 0) && (x2 < frame.cols / 2)){
        //     cv::line(frame3, cv::Point(line[0], line[1]+(frame.rows>>3)*5), cv::Point(line[2], line[3]+(frame.rows>>3)*5), cv::Scalar(0,0,255), 2, cv::LINE_8);
        //     cv::putText(frame3, cv::format("%f", slope), cv::Point(line[0], line[1]+(frame.rows>>3)*5), cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar::all(-1), 1, cv::LINE_AA);
        // }

        // else if((slope > 0) && (x1 > frame.cols /2)){
        //     cv::line(frame3, cv::Point(line[0], line[1]+(frame.rows>>3)*5), cv::Point(line[2], line[3]+(frame.rows>>3)*5), cv::Scalar(0,0,255), 2, cv::LINE_8);
        //     cv::putText(frame3,cv::format("%f", slope), cv::Point(line[0], line[1]+(frame.rows>>3)*5), cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar::all(-1), 1, cv::LINE_AA);
        // }

    //    cv::line(frame2, cv::Point(line[0], line[1]+(frame.rows>>3)*5), cv::Point(line[2], line[3]+(frame.rows>>3)*5), cv::Scalar(0,0,255), 2, cv::LINE_8);
    }

    double left_average_slope = 0.0;
    double left_average_intercept = 0.0;
    if(total_left_length != 0)
    {
        left_average_slope = left_slope_sum / total_left_length;
        left_average_intercept = left_intercept_sum / total_left_length;
    }
    double right_average_slope = 0.0;
    double right_average_intercept = 0.0;
    if(total_right_length != 0)
    {
        right_average_slope = right_slope_sum / total_right_length;
        right_average_intercept = right_intercept_sum / total_right_length;
    }

    [frame, left_average_slope, left_average_intercept, right_average_intercept, right_average_slope]()
    {
        // 가중 평균에 따라 차선을 그려주는 함수
        int y1 = frame.rows;
        int y2 = cvRound(y1>>1);
        int x1 = cvRound((y1 - left_average_intercept) / left_average_slope);
        int x2 = cvRound((y2 - left_average_intercept) / left_average_slope);
        cv::line(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0,0,255), 2, cv::LINE_8);

        y1 = frame.rows;
        y2 = cvRound(y1>>1);
        x1 = cvRound((y1 - right_average_intercept) / right_average_slope);
        x2 = cvRound((y2 - right_average_intercept) / right_average_slope);
        cv::line(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0,0,255), 2, cv::LINE_8);
    }();

    [&]()
    {
        // 로우 패스 필터 (재귀적 표현이 간단)
        if(left_estimation_slope == 0.0) { left_estimation_slope = left_average_slope; }
        if(left_estimation_intercept == 0.0) { left_estimation_intercept = left_average_intercept; }

//            if(fabs(left_average_slope) >= 1e-9 && fabs(left_average_intercept) >= 1e-9){}

        left_estimation_slope = alpha * left_estimation_slope +
                (1 - alpha) * left_average_slope;
        left_estimation_intercept = alpha * left_estimation_intercept +
                (1 - alpha) * left_average_intercept;


        int y1 = frame.rows;
        int y2 = cvRound(y1>>1);
        int x1 = cvRound((y1 - left_estimation_intercept) / left_estimation_slope);
        int x2 = cvRound((y2 - left_estimation_intercept) / left_estimation_slope);
        cv::line(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255,0,0), 2, cv::LINE_8);

        if (right_estimation_slope == 0.0) { right_estimation_slope = right_average_slope; }
        if (right_estimation_intercept == 0.0) { right_estimation_intercept = right_average_intercept; }

//            if(fabs(right_average_slope) >= 1e-9 && fabs(right_average_intercept) >= 1e-9){}

        right_estimation_slope = alpha * right_estimation_slope +
                (1 - alpha) * right_average_slope;
        right_estimation_intercept = alpha * right_estimation_intercept +
                (1 - alpha) * right_average_intercept;


        y1 = frame.rows;
        y2 = cvRound(y1>>1);
        x1 = cvRound((y1 - right_estimation_intercept) / right_estimation_slope);
        x2 = cvRound((y2 - right_estimation_intercept) / right_estimation_slope);
        cv::line(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255,0,0), 2, cv::LINE_8);
    }();

    // cv::bitwise_and(canny_crop, mask_lidar(cv::Rect(0,(mask_lidar.rows>>3)*5,mask_lidar.cols,(mask_lidar.rows>>3)*3)), canny_crop);


    // 기준 line
    cv::line(frame, cv::Point(0,400), cv::Point(640,400), cv::Scalar(0,255,0), 1, cv::LINE_4);

    //가로 중간
    cv::line(frame2, cv::Point(320,0), cv::Point(320,480), cv::Scalar(0,255,0), 1, cv::LINE_4);


    // if(count_frame % 30 == 0)
    //     cv::putText(frame, cv::format("frame: %d", count_frame), cv::Point(20,50), cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar::all(-1), 1, cv::LINE_AA);

    cv::imshow("frame_nahye", frame);
//        cv::imshow("crop", crop);
    cv::imshow("canny_crop_nahye", canny_crop);
    // cv::imshow("frame2", frame2);
    cv::imshow("frame3", frame3);

    while(1){
        if(cv::waitKey(1) == 27)   // ESC
            break;
    }
    cv::destroyAllWindows();
// std::cout <<"total frame: " << count_frame << std::endl;

    return 0;
}