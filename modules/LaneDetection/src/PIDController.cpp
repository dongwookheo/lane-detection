#include "LaneDetection/PIDController.hpp"

namespace XyCar
{
    PREC getControlOutput(int32_t error)
    {
        PREC cast_rrror = static_cast<PREC>(error);
        differential_error_ = cast_rrror - proportional_error_;
        proportional_error_ = cast_rrror;
        integral_error_ += cast_rrror;

        return proportional_gain_ * proportional_error_ + integral_gain_ * integral_error_ + differential_gain_ * differential_error_;
    }
}