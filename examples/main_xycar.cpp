// system header
#include <cstdint>

// user defined header
#include "LaneDetection/LaneManager.hpp"

int32_t main()
{
    ros::init();
    LaneManager laneManager;

    laneManager.run();
    ros::spin();

    return 0;
}
