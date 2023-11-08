#ifndef LANE_DETECTION__PIDCONTROLLER_HPP
#define LANE_DETECTION__PIDCONTROLLER_HPP

// user defined header
#include "LaneDetection/Common.hpp"

namespace Xycar{
/**
 * @details PID Controller Class
 */
class PIDController
{
public:
    PIDController(PREC p_gain, PREC i_gain, PREC d_gain);

    PREC getControlOutput(int32_t error);

private:
    PREC proportional_gain_;
    PREC integral_gain_;
    PREC differential_gain_;

    PREC proportional_error_ = 0.0;
    PREC integral_error_ = 0.0;
    PREC differential_error_ = 0.0;
};
}

#endif //LANE_DETECTION__PIDCONTROLLER_HPP
