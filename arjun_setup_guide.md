Run Instructions For Drone Sim:
PX4 - `cd ~/Documents/PX4-Autopilot/ && make px4_sitl gz_x500`
MavProxy - `cd ~/Documents/MAVProxy/ && mavproxy.py --master=udp:127.0.0.1:14550`
Both - `( cd ~/Documents/MAVProxy && mavproxy.py --master=udp:127.0.0.1:14550 >/dev/null 2>&1 ) & BG=$!; trap 'kill "$BG" 2>/dev/null' INT TERM EXIT; cd ~/Documents/PX4-Autopilot && make px4_sitl gz_x500`

Instructions For Drone Software Rebuild:
`colcon build --packages-select modular1 && source install/setup.bash && ros2 run modular1 service`

1. run with the project name as [PROJECT] and space separated deps as [DEPS] ex. rclpy mavsdk `ros2 pkg create --build-type ament_python --license Apache-2.0 [PROJECT] --dependencies [DEPS]`
2. Add python code to the ./[PROJECT]/ directory alongside __init__.py
3. Then in ./setup.py add `"[service_name] = [PROJECT].[file_name_no_py]:main"`
4. Also source python venv at home `source ~/ros2_venv/bin/activate`
5. Then run `rosdep install -i --from-path src --rosdistro kilted -y`
6. Then build with `colcon build --packages-select [PROJECT]`
7. And source with `source install/setup.bash`
8. And run `ros2 run [PROJECT] [service_name]`



<depend>python3-numpy</depend>
<depend>python3-matplotlib</depend>
<depend>rclpy</depend>
<depend>python3-scipy</depend>

ros2 pkg create --build-type ament_python --license Apache-2.0 modular1 --dependencies python3-numpy python3-matplotlib python3-scipy rclpy

for sys:
mavsdk
pymavlink
pyrealsense2