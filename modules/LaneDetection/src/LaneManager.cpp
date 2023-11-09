#include "LaneDetection/LaneManager.hpp"
#include "LaneDetection/PIDController.hpp"

namespace XyCar
{
    LaneManager::LaneManager(PREC p_gain, PREC i_gain, PREC d_gain) : pid_controller_(p_gain, i_gain, d_gain)
    {
        subscriber_ = node_handler_.subscribe("/usb_cam/image_raw/", 1, image_callback);
    }

    void LaneManager::image_callback(const sensor_msgs::Image& message)
    {
        cv::Mat image = cv::Mat(message.height, message.width, CV_8UC1, const_cast<uint8_t*>(&message.data[0]), message.step);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        current_images_.push(image_processor_.process(image));
    }

    void LaneManager::run()
    {
        if (current_images_.empty())
            return;

        cv::Mat current_image = current_images_.front();
        current_images_.pop();

        //detect rpos, lpos, and flag of stop
        std::tuple<int32_t, int32_t, bool> output_detector;
        output_detector = detector_.findPos(current_image);
        int32_t left_pos = std::get<0>(output_detector);
        int32_t right_pos = std::get<1>(output_detector);
        bool is_stop = std::get<2>(output_detector);

        int32_t error = k_frame_width / 2 - static_cast<int32_t>((right_pos + left_pos) / 2);

        PREC angle = pid_controller_.computeAngle(error);

        xycar_controller.control(angle);
    }

}
