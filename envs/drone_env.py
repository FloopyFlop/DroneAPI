import sys
sys.path.insert(1, 'c:/Users/Cameron Mehlman/Documents/magpie hardware')

import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Any, Dict, Type, Optional, Union

from trajectory_planning.path_planner import pathPlanner

import asyncio
from mavsdk import System
from mavsdk.offboard import (OffboardError, PositionNedYaw)

class magpieEnv():

    def __init__(
            self,
            path_planner: Type[pathPlanner],
            goal: list[float],
            point_cloud_size: int = 5000,     
    ) -> None:
        
        self.path_planner = path_planner

        self.goal = goal
        self.point_cloud_size = point_cloud_size

        self.initial_state = None
        self.global_state = None
        self.global_path_state = None
        self.global_path = None
        self.rel_path = None

        self.drone = System()

    

    def run(self) -> None:
        '''
        1. get state
        2. if position w/in tol of path[0]:
             pop(path,0)
             if path is None:
               take new LiDAR data
               fill point cloud
               generate new path
             else:
               take new LiDAR data
               fill point cloud
               check for safety
           else:
             return path[0] to pixhawk
        '''

    def reset(
            self,
    ) -> None:
        self.update_state()
        self.path_planner.set_goal_state(goal_state=self.goal)
        self.global_path_state = self.state
        self.initial_state = self.state

    def fill_point_cloud(
            self,
            point_cloud: list[list[float]],
    ) -> None:
        self.global_path_state = self.state

        if len(point_cloud) > self.point_cloud_size:
            self.path_planner.update_point_cloud(point_cloud=random.sample(point_cloud, self.point_cloud_size))
        else:
            self.path_planner.update_point_cloud(point_cloud=point_cloud)

    def generate_relative_path(
            self,
            rel_state: list[float],
            rel_goal: list[float],
    ) -> list[list[float]]:
        
        self.path_planner.set_goal_state(rel_goal)
        
        path = self.path_planner.compute_desired_path(
            state = rel_state,
            goal = rel_goal
        )

        return path
    
    def generate_global_path(
            self,
            global_goal: list[float],
    ) -> list[float]:
        
        global_state_change = self.state - self.global_path_state
        self.rel_path = self.generate_relative_path(rel_state=global_state_change,rel_goal=global_goal-self.global_path_state)

        self.global_path = self.rel_path + self.state

        return self.global_path
    
    def check_path_safety(self) -> bool:
        safe = self.path_planner.check_goal_safety(goals=self.rel_path,state=self.state)

    '''
    SUPPORT FUNCTIONS TO COMMUNICATE BETWEEN FLIGHT CONTROLLER AND DRONE ENV
    '''
    '''
    def update_state(self) -> None:
        '''
        get current positional/velocity state from pixhawk controller
        eurler angles would be good too if possible
        '''
        self.state = np.zeros(12,)

    def update_pixHawk(
            self,
            input: list[float],
    ) -> None:
        '''
        send positional/velocity input to pixHawk
        '''
        pass

    async def initialize_connection(
            self,
            port: str = '/dev/ttyTHS1',
            baud_rate: int = 57600,
    ) -> None:
        
        await self.drone.connect(system_address="port", baud=baud_rate)

        status_text_task = asyncio.ensure_future(print_status_text(self.drone))

        print("Waiting for connection...")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("Connection successful")
                break

    async def print_status_text(self):
        try:
            async for status_text in self.drone.telemetry.status_text():
                print(f"Status: {status_text.type}: {status_text.text}")
        except asyncio.CancelledError:
            return

    async def check_connection(
            self,
    ) -> bool:
        
        print("Waiting for global position estimate...")
        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print("Global position estimate OK")
                break
        
        print("Arming")
        await self.drone.action.arm()

        return True
    '''

    async def print_status_text():
        drone = System()
        await drone.connect(system_address="udp://:14540")

        print("Waiting for drone to connect...")
        async for state in drone.core.connection_state():
            if state.is_connected:
                print(f"-- Connected to drone!")
                break

        async for status_text in drone.telemetry.status_text():
            print("Statustext:", status_text)

        async for battery in drone.telemetry.battery():
            print(f"Battery: {battery.remaining_percent}")
            
    async def perform_connection(drone):
        await drone.connect(system_address="udp://:14540")

    async def perform_calibrations(drone):
        print("-- Starting gyroscope calibration")
        async for progress_data in drone.calibration.calibrate_gyro():
            print(progress_data)
        print("-- Gyroscope calibration finished")

        print("-- Starting accelerometer calibration")
        async for progress_data in drone.calibration.calibrate_accelerometer():
            print(progress_data)
        print("-- Accelerometer calibration finished")

        print("-- Starting magnetometer calibration")
        async for progress_data in drone.calibration.calibrate_magnetometer():
            print(progress_data)
        print("-- Magnetometer calibration finished")

        print("-- Starting board level horizon calibration")
        async for progress_data in drone.calibration.calibrate_level_horizon():
            print(progress_data)
        print("-- Board level calibration finished")
    
    async def print_gps_info(drone):
        async for gps_info in drone.telemetry.gps_info():
            print(f"GPS info: {gps_info}")


    async def print_in_air(drone):
        async for in_air in drone.telemetry.in_air():
            print(f"In air: {in_air}")
    
    async def print_position(drone):
        async for position in drone.telemetry.position():
            print(position)

    async def set_new_position(drone,NED):
        print("-- Arming")
        await drone.action.arm()

        print("-- Setting initial setpoint")
        await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))

        print("-- Starting offboard")
        try:
            await drone.offboard.start()
        except OffboardError as error:
            print(f"Starting offboard mode failed \
                    with error code: {error._result.result}")
            print("-- Disarming")
            await drone.action.disarm()
            return

        print("-- Go 0m North, 0m East, -5m Down \
                within local coordinate system")
        await drone.offboard.set_position_ned(
                PositionNedYaw(NED[1], NED[2], NED[3], NED[4]))
        await asyncio.sleep(10)

        print("-- Stopping offboard")
        try:
            await drone.offboard.stop()
        except OffboardError as error:
            print(f"Stopping offboard mode failed \
                    with error code: {error._result.result}")

    async def takeoff(drone):
                    # Execute the maneuvers
        print("-- Arming")
        await drone.action.arm()

        print("-- Taking off")
        await drone.action.set_takeoff_altitude(10.0)
        await drone.action.takeoff()

        await asyncio.sleep(10)

    async def land(drone):
        print("-- Landing")
        await drone.action.land()


if __name__ == "__main__": 
    #await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))
    pass



    
        

    
