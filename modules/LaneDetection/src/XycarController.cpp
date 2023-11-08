#include "LaneDetection/XycarController.hpp"

namespace XyCar
{
    XycarController::XycarController()
    {
        publisher_ = node_handler_.advertise<xycar_msgs::xycar_motor>("xycar_motor", 1);
    }

    void XycarController::control(XyCar::PREC angle)
    {
        publisher_.publish(make_motor_message(angle, decide_speed(angle)));
    }

    PREC XycarController::decide_speed(PREC angle)
    {
        // when xycar turn left
        if (angle < -10 && speed_ > k_min_speed_)
            speed_ -= k_step_speed_;
            // when xycar turn right
        else if (angle > 10 && speed_ > k_min_speed_)
            speed_ -= k_step_speed_;
            // when xycar go straight
        else if ((angle >= -10 && angle <= 10) && speed_ < k_max_speed_)
            speed_ += k_step_speed_;

        return speed_;
    }

    xycar_msgs::xycar_motor make_motor_message(PREC angle, PREC speed)
    {
        xycar_msgs::xycar_motor motor_message;
        motor_message.angle = std::round(angle);
        motor_message.speed = std::round(speed);

        return motor_message;
    }
}
