// 시스템 헤더
#include <iostream>

// 서드 파티 헤더
#include "opencv2/core.hpp"
#include "opencv2/core/operations.hpp"
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

    namespace kalman{
        double dt = 1;
        cv::Mat A = (cv::Mat_<double>(4,4) <<
                1, dt, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, dt,
                0, 0, 0, 1);
        cv::Mat H = (cv::Mat_<double>(2,4) << 1, 0, 0, 0, 0, 0, 1, 0);
        cv::Mat Q = cv::Mat::eye(4, 4, CV_64F);
        cv::Mat R = (cv::Mat_<double>(2, 2) << 50, 0, 0, 50);

        cv::Mat P = 100 * cv::Mat::eye(4, 4, CV_64F);
        cv::Mat left_prediction, right_prediction, P_prediction;
        cv::Mat K, left_measure, left_estimation, right_measure, right_estimation;
    } // kalman
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

    cv::uint32_t count_frame = 0;
    cv::Mat frame;
    cv::Mat crop(cv::Size(640, 480), CV_8UC3);
    cv::Mat mask_lidar = cv::imread("../examples/mask.png", CV_8UC1);

    uint32_t lpos = 0, rpos = 640;

    while(true)
    {
        cap >> frame;
        if(frame.empty())
        {
            std::cerr << "Frame empty" << std::endl;
            break;
        }

        ++count_frame;
        frame.copyTo(crop);
        crop = crop(cv::Rect(0, (frame.rows>>3)*5, frame.cols, (frame.rows>>3)*3));

        cv::cvtColor(crop, crop, cv::COLOR_BGR2GRAY);

        //이진화
        cv::equalizeHist(crop, crop);
        cv::threshold(crop, crop, 65, 255, cv::THRESH_BINARY_INV);

        //라이다 마스크 적용
        cv::bitwise_and(crop, mask_lidar(cv::Rect(0,(mask_lidar.rows>>3)*5,mask_lidar.cols,(mask_lidar.rows>>3)*3)), crop);

        // 가우시안 블러
        cv::GaussianBlur(crop, crop, cv::Size(), 5);

        // 캐니 에지
        cv::Mat canny_crop;
        cv::Canny(crop, canny_crop, 50, 150);

        //직선 검출
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(canny_crop, lines, 1, CV_PI/180, 60, 60, 5);


        double total_left_length = 0.0, total_right_length = 0.0;
        double left_slope_sum = 0.0, right_slope_sum = 0.0;
        double left_intercept_sum = 0.0, right_intercept_sum = 0.0;

        for(cv::Vec4i line : lines)
        {
            int x1 = line[0]; int y1 = line[1];
            int x2 = line[2]; int y2 = line[3];

            if(x2 - x1 == 0) {
                continue;
            }

            double slope = static_cast<double>(y2 - y1) / (x2 - x1);
            double intercept = y1+(frame.rows>>3)*5 - slope * x1;
            double line_length = sqrt((y2-y1)*(y2-y1) + (x2-x1)*(x2-x1));

            if((slope < -0.1) && (x1 < frame.cols / 2))
            {
                total_left_length += line_length;
                left_slope_sum += slope * line_length;
                left_intercept_sum += intercept * line_length;

            }
            else if((slope > 0.1) && (x2 > frame.cols /2))
            {
                total_right_length += line_length;
                right_slope_sum += slope * line_length;
                right_intercept_sum += intercept * line_length;
            }
        }

        double left_average_slope = 0.0;
        double left_average_intercept = 0.0;
        if(cvRound(total_left_length) != 0)
        {
            left_average_slope = left_slope_sum / total_left_length;
            left_average_intercept = left_intercept_sum / total_left_length;
        }

        double right_average_slope = 0.0;
        double right_average_intercept = 0.0;
        if(cvRound(total_right_length) != 0)
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
            using namespace kalman;
            // 칼만 필터
            if(count_frame == 1)
            {
                left_estimation_slope = left_average_slope;
                left_estimation_intercept = left_average_intercept;
                right_estimation_slope = right_average_slope;
                right_estimation_intercept = right_average_intercept;
            }

            // #1 Prediction
            left_prediction = A * (cv::Mat_<double>(4,1) <<
                                                         left_estimation_slope, -0.0005, left_estimation_intercept, -0.1);
            right_prediction = A * (cv::Mat_<double>(4,1) <<
                                                          right_estimation_slope, -0.0005, right_estimation_intercept, -0.1);
            P_prediction = A * P * A.t() + Q;

            // #2 Kalman Gain
            K = P_prediction * H.t() * (H * P_prediction * H.t() + R).inv();

            // #3 Estimation
            if (cvRound(total_left_length)!=0)
            {
                left_measure = (cv::Mat_<double>(2, 1) << left_average_slope, left_average_intercept);
                left_estimation = left_prediction + K * (left_measure - H * left_prediction);
                left_estimation_slope = left_estimation.at<double>(0,0);
                left_estimation_intercept = left_estimation.at<double>(2,0);
            }
            else
            {
                left_estimation = left_prediction; // 감지 하지 못하는 경우, 예측값 사용
                left_estimation_slope = left_estimation.at<double>(0,0);
                left_estimation_intercept = left_estimation.at<double>(2,0);
            }
            if (cvRound(total_right_length)!=0)
            {
                right_measure = (cv::Mat_<double>(2, 1) << right_average_slope, right_average_intercept);
                right_estimation = right_prediction + K * (right_measure - H * right_prediction);
                right_estimation_slope = right_estimation.at<double>(0,0);
                right_estimation_intercept = right_estimation.at<double>(2,0);
            }
            else
            {
                right_estimation = right_prediction;    // 감지 하지 못하는 경우, 예측값 사용
                right_estimation_slope = right_estimation.at<double>(0, 0);
                right_estimation_intercept = right_estimation.at<double>(2, 0);
            }

            // #4 Error covariance
            P = P_prediction - K * H * P_prediction;

            int y1 = frame.rows;
            int y2 = cvRound(y1>>1);
            int x1 = cvRound((y1 - left_estimation_intercept) / left_estimation_slope);
            int x2 = cvRound((y2 - left_estimation_intercept) / left_estimation_slope);
            cv::line(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255,0,0), 2, cv::LINE_8);

            y1 = frame.rows;
            y2 = cvRound(y1>>1);
            x1 = cvRound((y1 - right_estimation_intercept) / right_estimation_slope);
            x2 = cvRound((y2 - right_estimation_intercept) / right_estimation_slope);
            cv::line(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255,0,0), 2, cv::LINE_8);

            if(left_estimation_intercept == 0 && left_estimation_slope == 0){
                lpos = 0;
            }
            else{
                lpos = static_cast<uint32_t>((400 - left_estimation_intercept)/ left_estimation_slope);
            }

            if(right_estimation_intercept == 0 && right_estimation_slope == 0){
                rpos = 640;
            }
            else{
                rpos = static_cast<uint32_t>((400 - right_estimation_intercept)/ right_estimation_slope);
            }

            std::cout << cv::format("%d_frame : (lpos = %d, rpos = %d)", count_frame, lpos, rpos) << std::endl;
            cv::rectangle(frame, cv::Rect(cv::Point(lpos-5, 395),cv::Point(lpos+5, 405)), cv::Scalar(0, 255, 0), 2);
            cv::rectangle(frame, cv::Rect(cv::Point(rpos-5, 395),cv::Point(rpos+5, 405)), cv::Scalar(0, 255, 0), 2);
        }();

        // 기준 line
        cv::line(frame, cv::Point(0,400), cv::Point(640,400), cv::Scalar(0,255,0), 1, cv::LINE_4);

        cv::imshow("frame", frame);
        cv::imshow("canny_crop", canny_crop);

        if(cv::waitKey(1) == 27)   // ESC
            break;
    }
    cap.release();
    cv::destroyAllWindows();

    return 0;
}