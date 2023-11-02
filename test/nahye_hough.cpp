// 시스템 헤더
#include <iostream>
#include <fstream>

// 서드 파티 헤더
#include "opencv2/core.hpp"
#include "opencv2/core/fast_math.hpp"
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
    // cv::TickMeter tm;
    cv::Mat frame, frame2, frame3;
    cv::Mat crop(cv::Size(640, 480), CV_8UC3);
    cv::Mat mask_lidar = cv::imread("../examples/mask.png", CV_8UC1);
    bool save_flag = 0;

    uint32_t lpos = 0, rpos = 640;

    while(true)
    {

        cap >> frame;
        if(frame.empty())
        {
            std::cerr << "Frame empty" << std::endl;
            break;
        }

        frame2 = frame.clone();
        frame3 = frame.clone();
        ++count_frame;
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



        double total_left_length = 0.0, total_right_length = 0.0;
        double left_slope_sum = 0.0, right_slope_sum = 0.0;
        double left_intercept_sum = 0.0, right_intercept_sum = 0.0;
        // std::vector<cv::Vec4i> left_lines;
        // std::vector<cv::Vec4i> right_lines;

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
                // left_lines.push_back({x1,y1,x2,y2});

                cv::line(frame3, cv::Point(line[0], line[1]+(frame.rows>>3)*5), cv::Point(line[2], line[3]+(frame.rows>>3)*5), cv::Scalar(0,0,255), 2, cv::LINE_8);
                cv::putText(frame3, cv::format("%f", slope), cv::Point(line[0], line[1]+(frame.rows>>3)*5), cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar::all(-1), 1, cv::LINE_AA);
            }
            else if((slope > 0.1) && (x2 > frame.cols /2))
            {
                total_right_length += line_length;
                right_slope_sum += slope * line_length;
                right_intercept_sum += intercept * line_length;
                // right_lines.push_back({x1,y1,x2,y2});

                cv::line(frame3, cv::Point(line[0], line[1]+(frame.rows>>3)*5), cv::Point(line[2], line[3]+(frame.rows>>3)*5), cv::Scalar(0,0,255), 2, cv::LINE_8);
                cv::putText(frame3, cv::format("%f", slope), cv::Point(line[0], line[1]+(frame.rows>>3)*5), cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar::all(-1), 1, cv::LINE_AA);

            }
            // else {
            //     std::cout << "problem : " << cv::format("%d = %f", count_frame, slope)<< std::endl;
            //     save_flag = 1;
            // }

            // cv::line(frame2, cv::Point(line[0], line[1]+(frame.rows>>3)*5), cv::Point(line[2], line[3]+(frame.rows>>3)*5), cv::Scalar(0,0,255), 2, cv::LINE_8);
            // cv::putText(frame2, cv::format("%f", slope), cv::Point(line[0], line[1]+(frame.rows>>3)*5), cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar::all(-1), 1, cv::LINE_AA);

        }

        double left_average_slope = 0.0;
        double left_average_intercept = 0.0;
        if(cvRound(total_left_length) != 0)
        {
            left_average_slope = left_slope_sum / total_left_length;
            left_average_intercept = left_intercept_sum / total_left_length;
        }
        else {
            // std::cout << "l slope == " << total_left_length << std::endl;
        }
        double right_average_slope = 0.0;
        double right_average_intercept = 0.0;
        if(cvRound(total_right_length) != 0)
        {
            right_average_slope = right_slope_sum / total_right_length;
            right_average_intercept = right_intercept_sum / total_right_length;
        }
        else {
            // std::cout << "r slope == 0" << std::endl;
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
//             if(cvRound(left_estimation_slope) == 0) {
//                 left_estimation_slope = left_average_slope;
//                 // std::cout << "---------------check--------------" << std::endl;
//                 // std::cout << count_frame <<"_left_average_slope:  " << left_average_slope << std::endl;
//             }
//             if(cvRound(left_estimation_intercept) == 0) {
//                 left_estimation_intercept = left_average_intercept;
//             }

//             left_estimation_slope = alpha * left_estimation_slope +
//                     (1 - alpha) * left_average_slope;
//             // std::cout << count_frame <<"left_estimation_slope:  " << left_estimation_slope << std::endl;

//             left_estimation_intercept = alpha * left_estimation_intercept +
//                     (1 - alpha) * left_average_intercept;

//             // int y1 = frame.rows;
//             // int y2 = cvRound(y1>>1);
//             // int x1 = cvRound((y1 - left_estimation_intercept) / left_estimation_slope);
//             // int x2 = cvRound((y2 - left_estimation_intercept) / left_estimation_slope);

//             // cv::line(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255,0,0), 2, cv::LINE_8);
//             // cv::putText(frame, cv::format("%f", left_estimation_slope), cv::Point(x2, y2), cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar::all(-1), 1, cv::LINE_AA);

//             if(cvRound(right_estimation_slope) == 0) {
//                 right_estimation_slope = right_average_slope;
//                 // std::cout << cv::format("%d: right_estimation_slope none", count_frame) << std::endl;
//             }
//             // if (right_estimation_slope == 0.0) { right_estimation_slope = right_average_slope; }
//             if (cvRound(right_estimation_intercept) == 0) {
//                 right_estimation_intercept = right_average_intercept;
//             }

// //            if(fabs(right_average_slope) >= 1e-9 && fabs(right_average_intercept) >= 1e-9){}

//             right_estimation_slope = alpha * right_estimation_slope +
//                     (1 - alpha) * right_average_slope;
//             right_estimation_intercept = alpha * right_estimation_intercept +
//                     (1 - alpha) * right_average_intercept;


            // y1 = frame.rows;
            // y2 = cvRound(y1>>1);
            // x1 = cvRound((y1 - right_estimation_intercept) / right_estimation_slope);
            // x2 = cvRound((y2 - right_estimation_intercept) / right_estimation_slope);
            // cv::line(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255,0,0), 2, cv::LINE_8);

            left_estimation_intercept = left_average_intercept;
            left_estimation_slope = left_average_slope;
            right_estimation_intercept = right_average_intercept;
            right_estimation_slope = right_average_slope;

            if(cvRound(left_estimation_intercept) == 0 && cvRound(left_estimation_slope) == 0){
                lpos = 0;
                // std::cout << cv::format("%d_lpos none", count_frame) << std::endl;
            }
            else{
                lpos = static_cast<uint32_t>((400 - left_estimation_intercept)/ left_estimation_slope);
            }

            if(cvRound(right_estimation_intercept) == 0 && cvRound(right_estimation_slope) == 0){
                rpos = 640;
                // std::cout << cv::format("%d_rpos none", count_frame) << std::endl;
            }
            else{
                rpos = static_cast<uint32_t>((400 - right_estimation_intercept)/ right_estimation_slope);
                if(rpos > 640){
                    rpos = 640;
                    save_flag = 1;
                    // std::cout << (400 - right_estimation_intercept)/ right_estimation_slope << std::endl;
                    // std::cout << "right_estimation_intercept: " << right_estimation_intercept << std::endl;
                    // std::cout << "right_estimation_slope: " << right_estimation_slope << std::endl;

                }
                cv::putText(frame, cv::format("first : %d", rpos), cv::Point(500, 50), cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar::all(-1), 1, cv::LINE_AA);

            }

            cv::putText(frame, cv::format("origin : %d", rpos), cv::Point(500, 75), cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar::all(-1), 1, cv::LINE_AA);


            if(lpos == 0 && rpos != 640){
                if((0.6 < abs(right_estimation_slope)) && (abs(right_estimation_slope) < 1)){
                    lpos = rpos - 490;
                    left_estimation_slope = -right_estimation_slope;
                    left_estimation_intercept = 400 - left_estimation_slope * lpos;
                }
            }
            else if(rpos == 640 && lpos != 0){
                if((0.6 < abs(left_estimation_slope)) && (abs(left_estimation_slope) < 1)){
                    rpos = lpos + 490;
                    right_estimation_slope = -left_estimation_slope;
                    right_estimation_intercept = 400 - right_estimation_slope * rpos;
                }
            }

            int y1 = frame.rows;
            int y2 = cvRound(y1>>1);
            int x1 = cvRound((y1 - left_estimation_intercept) / left_estimation_slope);
            int x2 = cvRound((y2 - left_estimation_intercept) / left_estimation_slope);

            cv::line(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255,0,0), 2, cv::LINE_8);

            x1 = cvRound((y1 - right_estimation_intercept) / right_estimation_slope);
            x2 = cvRound((y2 - right_estimation_intercept) / right_estimation_slope);

            cv::line(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255,0,0), 2, cv::LINE_8);

            cv::rectangle(frame, cv::Rect(cv::Point(lpos-5, 395),cv::Point(lpos+5, 405)), cv::Scalar(0, 255, 0));
            cv::rectangle(frame, cv::Rect(cv::Point(rpos-5, 395),cv::Point(rpos+5, 405)), cv::Scalar(0, 255, 0));

            cv::putText(frame, cv::format("change: %d", rpos), cv::Point(500, 100), cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar::all(-1), 1, cv::LINE_AA);

        }();


        // 기준 line
        cv::line(frame, cv::Point(0,400), cv::Point(640,400), cv::Scalar(0,255,0), 1, cv::LINE_4);

        //가로 중간
        // cv::line(frame2, cv::Point(320,0), cv::Point(320,480), cv::Scalar(0,255,0), 1, cv::LINE_4);


        // if(count_frame % 30 == 0)
        //     cv::putText(frame, cv::format("frame: %d", count_frame), cv::Point(20,50), cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar::all(-1), 1, cv::LINE_AA);

        cv::imshow("frame_nahye", frame);
//        cv::imshow("crop", crop);
        cv::imshow("canny_crop_nahye", canny_crop);
        // cv::imshow("frame2", frame2);
        cv::imshow("frame3", frame3);

        if(cv::waitKey(1) == 27)   // ESC
            break;
        else if (cv::waitKey(1) == 25){
            cv::imwrite(cv::format("../examples/line_images/frame_%d.png", count_frame), frame2);
        }

        if(save_flag == 1){
            cv::imwrite(cv::format("../data/make_line/frames_%d.png", count_frame), frame);
            // cv::imwrite(cv::format("../examples/line_frame3/frame3_%d.png", count_frame), frame3);
            save_flag = 0;
        }
    }
    cap.release();
    cv::destroyAllWindows();
    // std::cout <<"total frame: " << count_frame << std::endl;

    return 0;
}