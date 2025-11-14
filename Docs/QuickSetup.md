PX4 - `cd ~/Documents/PX4-Autopilot/ && make px4_sitl gz_x500` OR NO GUI `cd ~/Documents/PX4-Autopilot/ && HEADLESS=1 make px4_sitl gz_x500`
MavProxy - `cd ~/Documents/MAVProxy/ && mavproxy.py --master=udp:127.0.0.1:14550`

Drone Position Interpolation Demo - `colcon build --packages-select drone_api && source install/setup.bash && ros2 run drone_api test_demo`
Test Stream GUI Logger - `colcon build --packages-select drone_api && source install/setup.bash && ros2 run drone_api test_stream_logging`