#ifndef LANE_DETECTION__LANEMANAGER_HPP
#define LANE_DETECTION__LANEMANAGER_HPP

// system header
#include <opencv2/core/mat.hpp>
#include <queue>
#include <iostream>

// third party header
#include <sensor_msgs/Image.h>
#include <tuple>

// user defined header
#include "LaneDetection/Common.hpp"
#include "LaneDetection/ImageProcessor.hpp"
#include "LaneDetection/LaneDetector.hpp"
#include "LaneDetection/PIDController.hpp"
#include "LaneDetection/XycarController.hpp"

namespace XyCar
{
class LaneManager
{
public:
    LaneManager(PREC p_gain, PREC i_gain, PREC d_gain);

    void run();

private:
    ros::NodeHandle node_handler_;
    ros::Subscriber subscriber_;

    ImageProcessor image_processor_;
    LaneDetector detector_;
    PIDController pid_controller_;
    XycarController xycar_controller;

    std::queue <cv::Mat> current_images_;

    void image_callback(const sensor_msgs::Image& message);
};
} // XyCar

#endif // LANE_DETECTION__LANEMANAGER_HPP