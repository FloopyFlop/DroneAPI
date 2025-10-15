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