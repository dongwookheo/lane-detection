// system header
#include <cmath>
#include <iostream>

// third party header
#include "opencv2/opencv.hpp"

// 사용자 정의 헤더
#include "LaneDetection/LaneDetector.hpp"
#include "LaneDetection/PreProcessor.hpp"

// 전역 변수
namespace
{
    double left_estimation_slope = 0.0;
    double left_estimation_intercept = 0.0;
    double right_estimation_slope = 0.0;
    double right_estimation_intercept = 0.0;
    constexpr int16_t k_offset = 400;
}
namespace kalman
{
    double dt = 1;
    cv::Mat transition_matrix = (cv::Mat_<double>(4,4) <<
            1, dt, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, dt,
            0, 0, 0, 1);
    cv::Mat measurement_matrix = (cv::Mat_<double>(2,4) << 1, 0, 0, 0, 0, 0, 1, 0);
    cv::Mat process_noise_matrix = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat measurement_noise_matrix = (cv::Mat_<double>(2, 2) << 50, 0, 0, 50);

    cv::Mat covariance_matrix = 100 * cv::Mat::eye(4, 4, CV_64F);
    cv::Mat left_prediction, right_prediction, covariance_matrix_prediction;
    cv::Mat kalman_gain;
} // kalman



int main()
{
    cv::String file_path = "../examples/Sub_project.avi";
    cv::VideoCapture cap(file_path);
    if(!cap.isOpened())
    {
        std::cerr << "Camera open failed!" << std::endl;
        return -1;
    }

    uint16_t count_frame = 0;
    cv::Mat frame;
    cv::Mat crop(cv::Size(640, 480), CV_8UC3);
    cv::Mat mask_lidar = cv::imread("../examples/mask.png", CV_8UC1);

    uint16_t lpos = 0;
    uint16_t rpos = 640;

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

        // binarization
        cv::equalizeHist(crop, crop);
        cv::threshold(crop, crop, 65, 255, cv::THRESH_BINARY_INV);

        // apply mask
        cv::bitwise_and(crop, mask_lidar(cv::Rect(0,(mask_lidar.rows>>3)*5,mask_lidar.cols,(mask_lidar.rows>>3)*3)), crop);

        // gussian blur
        cv::GaussianBlur(crop, crop, cv::Size(), 5);

        // canny
        cv::Mat canny_crop;
        cv::Canny(crop, canny_crop, 50, 150);

        // line detect with hough transform
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(canny_crop, lines, 1, CV_PI/180, 60, 60, 5);

        double total_left_length = 0.0;
        double total_right_length = 0.0;
        double left_slope_sum = 0.0;
        double right_slope_sum = 0.0;
        double left_intercept_sum = 0.0;
        double right_intercept_sum = 0.0;

        for(const cv::Vec4i& line : lines)
        {
            int32_t x1 = line[0];
            int32_t y1 = line[1];
            int32_t x2 = line[2];
            int32_t y2 = line[3];

            if(x2 == x1)
                continue;

            int32_t diff_x = x2 - x1;
            int32_t diff_y = y2 - y1;
            double slope = static_cast<double>(y2 - y1) / (x2 - x1);
            double intercept = y1 + (frame.rows>>3) * 5 - slope * x1;
            double line_length = sqrt(diff_y * diff_y + diff_x * diff_x);

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
        if(std::round(total_left_length) != 0)
        {
            left_average_slope = left_slope_sum / total_left_length;
            left_average_intercept = left_intercept_sum / total_left_length;
        }

        double right_average_slope = 0.0;
        double right_average_intercept = 0.0;
        if(std::round(total_right_length) != 0)
        {
            right_average_slope = right_slope_sum / total_right_length;
            right_average_intercept = right_intercept_sum / total_right_length;
        }


        [frame, left_average_slope, left_average_intercept, right_average_intercept, right_average_slope]()
        {
            // 가중 평균에 따라 차선을 그려주는 함수
            int32_t y1 = frame.rows;
            int32_t y2 = std::round(y1>>1);
            int32_t x1 = std::round((y1 - left_average_intercept) / left_average_slope);
            int32_t x2 = std::round((y2 - left_average_intercept) / left_average_slope);
            cv::line(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0,0,255), 2, cv::LINE_8);

            x1 = std::round((y1 - right_average_intercept) / right_average_slope);
            x2 = std::round((y2 - right_average_intercept) / right_average_slope);
            cv::line(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0,0,255), 2, cv::LINE_8);
        }();

        [=, &lpos, &rpos]()
        {
            // 칼만 필터
            if(count_frame == 1)
            {
                left_estimation_slope = left_average_slope;
                left_estimation_intercept = left_average_intercept;
                right_estimation_slope = right_average_slope;
                right_estimation_intercept = right_average_intercept;
            }

            // #1 Prediction
            kalman::left_prediction = kalman::transition_matrix * (cv::Mat_<double>(4,1) <<
                                                         left_estimation_slope, -0.0005, left_estimation_intercept, -0.1);
            kalman::right_prediction = kalman::transition_matrix * (cv::Mat_<double>(4,1) <<
                                                          right_estimation_slope, -0.0005, right_estimation_intercept, -0.1);
            static const auto transition_matrix_t = kalman::transition_matrix.t();
            kalman::covariance_matrix_prediction = kalman::transition_matrix * kalman::covariance_matrix * transition_matrix_t + kalman::process_noise_matrix;

            // #2 Kalman Gain
            static const auto measurement_matrix_t = kalman::measurement_matrix.t();
            kalman::kalman_gain = kalman::covariance_matrix_prediction * measurement_matrix_t * (kalman::measurement_matrix * kalman::covariance_matrix_prediction * measurement_matrix_t + kalman::measurement_noise_matrix).inv();

            // #3 Estimation
            auto updateEstimation =
                    [=](double total_line_length, double average_slope, double average_intercept,
                            cv::Mat& prediction, double& estimation_slope, double& estimation_intercept)
            {
                cv::Mat estimation = prediction;
                if (std::round(total_line_length) != 0)
                {
                    cv::Mat measurement = (cv::Mat_<double>(2, 1) << average_slope, average_intercept);
                    estimation += kalman::kalman_gain * (measurement - kalman::measurement_matrix * prediction);
                }

                estimation_slope = estimation.at<double>(0, 0);
                estimation_intercept = estimation.at<double>(2, 0);
            };

            updateEstimation(total_left_length, left_average_slope, left_average_intercept,
                             kalman::left_prediction, left_estimation_slope, left_estimation_intercept);

            updateEstimation(total_right_length, right_average_slope, right_average_intercept,
                             kalman::right_prediction, right_estimation_slope, right_estimation_intercept);

            // #4 Error covariance
            kalman::covariance_matrix = kalman::covariance_matrix_prediction - kalman::kalman_gain * kalman::measurement_matrix * kalman::covariance_matrix_prediction;

            int32_t y1 = frame.rows;
            int32_t y2 = std::round(y1>>1);
            int32_t x1 = std::round((y1 - left_estimation_intercept) / left_estimation_slope);
            int32_t x2 = std::round((y2 - left_estimation_intercept) / left_estimation_slope);
            cv::line(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255,0,0), 2, cv::LINE_8);

            x1 = std::round((y1 - right_estimation_intercept) / right_estimation_slope);
            x2 = std::round((y2 - right_estimation_intercept) / right_estimation_slope);
            cv::line(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255,0,0), 2, cv::LINE_8);

            auto checkSlopeAndIntercept = [](double intercept, double slope)
            {
                return ((intercept == 0) && (slope == 0));
            };

            auto calculatePosition =
                    [checkSlopeAndIntercept](double intercept, double slope, uint16_t defaultValue)
            {
                if(checkSlopeAndIntercept(intercept, slope))
                    return defaultValue;
                else
                    return static_cast<uint16_t>((k_offset - intercept) / slope);
            };

            lpos = calculatePosition(left_estimation_intercept, left_estimation_slope, 0);
            rpos = calculatePosition(right_estimation_intercept, right_estimation_slope, 640);
#if DEBUG
            std::cout << cv::format("%d_frame : (lpos = %d, rpos = %d)", count_frame, lpos, rpos) << std::endl;
#endif
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
