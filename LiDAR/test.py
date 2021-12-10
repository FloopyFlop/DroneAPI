import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import numpy as np
from multiprocessing import Process, Value, Array


class lidar()

    def __init__(
        self,
        point_cloud_size: int = 100,
    ) -> None:

        self.collect_data = True
        self.pipeline = None
        self.pc = None
        self.decimate = None
        self.point_cloud = Array('i', np.zeros((self.point_cloud_size,)))
        self.point_cloud_size = point_cloud_size

        self.initialize_pipeline()

        update = Process(target=self.update, args=(self.point_cloud))
        update.start()

    def initialize_pipeline(
        self,
    ) -> None:

        self.pipeline = rs.pipeline()
        config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        config.enable_stream(rs.stream.depth, rs.format.z16, 30)
        #config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)

        # Get stream profile and camera intrinsics
        profile = self.pipeline.get_active_profile()

        # Set laser power, gain, and confidence
        sensor_dep = profile.get_device().first_depth_sensor()
        sensor_dep.set_option(rs.option.laser_power, 50)
        sensor_dep.set_option(rs.option.receiver_gain, 18)
        sensor_dep.set_option(rs.option.confidence_threshold, 3)

        # Get camera intrinsics
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        # Processing blocks
        self.pc = rs.pointcloud()
        self.decimate = rs.decimation_filter()
        self.decimate.set_option(rs.option.filter_magnitude, 2 ** 1)

    def update_pc(
        self,
        point_cloud: Array,
    ) -> None:
        #start_time = time.time()
        frames = pipeline.wait_for_frames()
        #end_time = time.time()
        #print(end_time-start_time)

        depth_frame = frames.get_depth_frame()

        depth_frame = self.decimate.process(depth_frame)

        # Grab new intrinsics (may be changed by decimation)
        depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        depth_image = np.asanyarray(depth_frame.get_data())

        points = self.pc.calculate(depth_frame)
        v = points.get_vertices()
        point_cloud = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz

    def update(
        self,
        point_cloud: Array,
    ) -> None:

        '''
        always update the point cloud with most recent lidar date
        '''

        while self.collect_data:
            self.update_pc(point_cloud)


'''
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(verts[:,0],verts[:,1],verts[:,2])
plt.show()
'''