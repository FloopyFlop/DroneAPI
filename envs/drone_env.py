import sys
sys.path.insert(1, '/home/magpie1/magpieEnv')

import numpy as np
from copy import deepcopy
import random
import time
import matplotlib.pyplot as plt
from typing import Any, Dict, Type, Optional, Union, List

from trajectory_planning.path_planner import pathPlanner
from LiDAR.lidar import lidar

import asyncio
from mavsdk import System
from mavsdk.offboard import (PositionNedYaw, VelocityNedYaw, OffboardError)
import serial
from pymavlink import mavutil

class magpieEnv():

    def __init__(
            self,
            path_planner: Type[pathPlanner],
            goal: List[float],
            point_cloud_size: int = 5000,     
            max_point_cloud_size: int = 500,
            points_min: int = 1, #minimum number of point cloud points per bin to register as obstacle
            position_tracking: bool = False,
            wait_time: int = 7
    ) -> None:
        
        self.path_planner = path_planner
        self.position_tracking = position_tracking

        self.goal = goal
        self.point_cloud = None
        self.point_cloud_size = point_cloud_size
        self.max_point_cloud_size = max_point_cloud_size
        self.points_min = points_min
        self.mission_end = False
        self.path = None
        self.wait_time = wait_time

        self.lidar = None
        self.drone = System()

        self.state = np.zeros((12,))
        self.state_offset = None
        self.time_step = 0


    '''
    SUPPORT FUNCTIONS TO COMMUNICATE BETWEEN FLIGHT CONTROLLER AND DRONE ENV
    '''

    def check_path_safety(self) -> bool:
        safe = self.path_planner.check_goal_safety(goals=self.rel_path,state=self.state)

    async def run_mission(
        self,
        angle: float = 0.0,
        tolerance: float = 0.5,
        initial_altitude: float = 3
    ) -> None:
        print('--setting up lidar pipeline')
        self.initialize_lidar()

        print('--preparing for takeoff')
        await self.mission_start(alt=initial_altitude)

        while not self.mission_end:
            await self.update_state()
            self.load_point_cloud()
            if self.path is None:
                self.generate_path()
            else:
                safe = self.check_safety()
                position_reached = await self.position_reached(wait_time=self.wait_time)#goal=next_point+self.state_offset[0:3])
                if not safe:
                    print('--recompute path')
                    self.generate_path()
                    next_point = self.xyz_to_NED(self.path[0][0:4] + self.state_offset[0:4])
                    next_point[3] = angle
                    await self.set_new_position(NED=next_point)
                    self.path = self.path[1:]
                if position_reached and safe:
                    if len(self.path) == 1:
                        if np.linalg.norm(self.path[0][0:3] - self.goal[0:3]) < tolerance:
                            self.mission_end=True
                        self.generate_path()
                    next_point = self.xyz_to_NED(self.path[0][0:4] + self.state_offset[0:4])
                    next_point[3] = angle
                    await self.set_new_position(NED=next_point)
                    self.path = self.path[1:]
                else:
                    pass  
            self.time_step += 1

        await self.mission_end()

    def check_safety(
        self,
    ):
        safe = self.path_planner.check_goal_safety(goals=self.path,state=self.state[0:3])
        return safe

    async def position_reached(
        self,
        goal: Optional[List[float]] = None,
        tolerance: float = 0.5,
        wait_time: int = 7,
    ) -> bool:
        if self.position_tracking:
            if np.linalg.norm(goal[0:3]-self.state[0:3]) < tolerance:
                return True
            else:
                return False
        else:
            await asyncio.sleep(wait_time)
            return True

    def generate_path(
        self,
    ) -> None:        
        path = self.path_planner.compute_desired_path(
            state = self.state-self.state_offset,
            point_cloud = self.point_cloud,
            points_min = self.points_min
        )

        self.path = path

    def load_point_cloud(
        self,
    ):
        
        #point_cloud = self.get_lidar_data()
        point_cloud = self.lidar.update_pc()

        if len(point_cloud) > self.max_point_cloud_size:
            idx = np.random.randint(len(point_cloud),size=self.max_point_cloud_size)
            self.point_cloud = point_cloud[idx,:]
        else:
            self.point_cloud = point_cloud 
        
        if len(self.point_cloud) == 0:
            self.point_cloud = None


        self.path_planner.update_point_cloud(point_cloud=self.point_cloud, points_min=self.points_min)

    def initialize_lidar(
        self,
        point_cloud_size: int = 1000,
    ):
        self.lidar = lidar(point_cloud_size=point_cloud_size)
        #self.lidar.start_update()

    def terminate_lidar(
        self,
    ):
        self.lidar.stop_update()

    def get_lidar_data(
        self,
    ):
        return self.lidar.get_point_cloud()

    def xyz_to_NED(
        self,
        xyz: List[float],
    ) -> List[float]:
        '''
        x -> left right
        y -> up down
        z -> forward backward
        '''
        NED = [0,0,0,0]

        NED[3] = 0
        NED[0] = xyz[2]
        NED[1] = xyz[0]
        NED[2] = -1*xyz[1]

        return NED

    async def compute_offset(
        self,
    ) -> List[float]:
        '''
        pixhawk NED initial position is not 0, compute initial offset in xyz
        '''
        offset = await self.get_position()

        self.state_offset = np.zeros((len(self.state),))
        self.state_offset[0] = offset[1]
        self.state_offset[1] = -1*offset[2]
        self.state_offset[2] = offset[0]
        

    async def update_state(self) -> None:
        '''
        get NED coordinates from pixhawk, and update state in xyz
        '''
        pos = await self.get_position()
        self.state[0] = pos[1]
        self.state[1] = -1*pos[2]
        self.state[2] = pos[0]

    async def get_battery_percentage(self):
        async for battery in self.drone.telemetry.battery():
            return battery.remaining_percent
    
    async def get_gps_info(self):
        async for gps_info in self.drone.telemetry.gps_info():
            return(gps_info)
    
    async def get_position(self):
        async for data in self.drone.telemetry.position_velocity_ned():
            return([data.position.north_m,data.position.east_m,data.position.down_m])

    async def set_new_position(
        self,
        NED: List[float],
        vel: Optional[List[float]] = None,
    ) -> None :
        print('--relative coordinates:', self.state[0:3]-self.state_offset[0:3])
        ned_coord = await self.get_position()
        print('--NED coordinates:', ned_coord)
        print("-- Setting new position")
        if vel is None:
            await self.drone.offboard.set_position_ned(PositionNedYaw(NED[0], NED[1], NED[2], NED[3]))
        else:
            await self.drone.offboard.set_position_velocity_ned(PositionNedYaw(NED[0], NED[1], NED[2], NED[3]), VelocityNedYaw(vel[0], vel[1], vel[2], vel[3]))


    async def turn_on_offboard(self):
        print("-- Starting offboard")
        try:
            await self.drone.offboard.start()
        except OffboardError as error:
            print(f"Starting offboard mode failed \
                    with error code: {error._result.result}")
            print("-- Disarming")
            await self.drone.action.disarm()
            return
            
    async def turn_off_offboard(self):
        print("-- Stopping offboard")
        try:
            await self.drone.offboard.stop()
        except OffboardError as error:
            print(f"Stopping offboard mode failed \
                    with error code: {error._result.result}")

    async def takeoff(
        self,
        altitude: float,
    ):
        print("-- Taking off")
        await self.drone.action.set_takeoff_altitude(altitude)
        await self.drone.action.takeoff()

    async def land(self):
        print("-- Landing")
        await self.drone.action.land()

    async def mission_start(
        self,
        alt: float = 2,
        angle: float = 0,
        port: str = 'serial:///dev/ttyTHS1:57600',
    ):

        await self.drone.connect(system_address=port)

        print("Waiting for connection...")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print(f"-- connection successful")
                break
        
        print("Waiting for drone to have a global position estimate...")
        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print("-- Global position estimate OK")
                break

        await self.turn_off_offboard()

        await self.compute_offset()
        rel_start_point = np.array([0, alt, 0, 0])
        start_point = self.xyz_to_NED(rel_start_point + self.state_offset[0:4])
        start_point[3] = angle

        print("-- Arming")
        await self.drone.action.arm()

        await self.update_state()
        print('--relative coordinates:', self.state[0:3]-self.state_offset[0:3])
        ned_coord = await self.get_position()
        print('--NED coordinates:', ned_coord)

        await self.set_new_position(NED=start_point, vel=[0.0,0.0,-0.2,0.0])
        print('--taking off')
        await self.turn_on_offboard()

        #await self.takeoff(altitude=alt)

        await asyncio.sleep(10)

        #await self.turn_off_offboard()

        #await asyncio.sleep(30)

        await self.update_state()
        print('--relative coordinates:', self.state[0:3]-self.state_offset[0:3])
        ned_coord = await self.get_position()
        print('--NED coordinates:', ned_coord)


    async def mission_finish(
        self,
    ):
        print("-- Landing")
        await self.land()

        await asyncio.sleep(15)

        await self.turn_off_offboard()

    async def demo1(
        self,
        port: str = 'serial:///dev/ttyTHS1:57600',
    ):

        points = np.array([ [0.0, 2.0, 1.0, 0], 
                            [1.0, 2.0, 2.0, 0], 
                            [0.0, 2.0, 3.0, 0], 
                            [1.0, 2.0, 4.0, 0] ])


        await self.mission_start(alt=2.0)

        
        for p in points:
            next_point = self.xyz_to_NED(p + self.state_offset[0:4])
            next_point[3] = 0
            await self.set_new_position(NED=next_point)
            await asyncio.sleep(8)
            await self.update_state()
            print('--relative coordinates:', self.state[0:3]-self.state_offset[0:3])
            #ned_coord = await self.get_position()
            #print('--NED coordinates:', ned_coord)
        
        
        await self.mission_finish()       

    async def demo2(
        self,
        port: str = 'serial:///dev/ttyTHS1:57600',
    ):
        
        print('--setting up lidar pipeline')
        self.initialize_lidar()

        await self.mission_start(alt=3.0,angle=0)
        '''

        points = np.array([ [0.0, 3.0, 0.0, 0], 
                            [0.0, 4.0, 0.0, 0], 
                            [0.0, 5.0, 0.0, 0], 
                            [0.0, 2.0, 0.0, 0] ])

        
        for p in points:
            next_point = self.xyz_to_NED(p + self.state_offset[0:4])
            next_point[3] = 0
            await self.set_new_position(NED=next_point)
            await asyncio.sleep(8)
            await self.update_state()
            print('--relative coordinates:', self.state[0:3]-self.state_offset[0:3])
        
        '''
        
        await self.update_state()
        print('--generate a path')
        self.generate_path()
        print(self.path)

        
        for i in range(5):
            
            ready = input('start lidar test')
            self.load_point_cloud()
            
            await self.update_state()
            self.load_point_cloud()
            safe = self.check_safety()
            print('local coord:', self.state[0:3]-self.state_offset[0:3])
            print('safety:', safe)

            name = 'pc_set_' + str(i) + '.npy'
            np.save(name, self.point_cloud)
            #data = np.load('data.npy')
            

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            #ax.plot(env.path[:,0], env.path[:,1], env.path[:,2])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.scatter(self.point_cloud[:,0],self.point_cloud[:,1],self.point_cloud[:,2])

            plt.savefig('img'+str(i)+'.png')
        
            
        
        
        await self.mission_finish()
        

        '''
        self.initialize_lidar()
        self.state = np.array([0,3,0,0,0,0,0,0,0,0,0,0])
        self.state_offset = np.array([0,0,0,0,0,0,0,0,0,0,0,0])

        print('--generate a path')
        self.generate_path()

        self.load_point_cloud()
        safe = self.check_safety()
        print(safe)
        '''
        


if __name__ == "__main__": 

    goal = [0,3,30,0,0,0,0,0,0,0,0,0]

    traj = pathPlanner(
        goal_state=goal,
        path_planning_algorithm='VFH',
        interpolation_method='spline',
        kwargs={'radius':10,
                'iterations':1,
                'layers':8, 
                'angle_sections':10,
                'distance_tolerance': 0.2,
                'probability_tolerance': 0.05,
                'min_obstacle_distance': 2.0,
                },
        n = 10,
    )
    
    
    env = magpieEnv(
        path_planner=traj,
        goal=goal,
        point_cloud_size = 1000,
        points_min = 10,
        wait_time = 1,
    )

    #env.mission_test()

    #asyncio.run(env.demo2())

    asyncio.run(env.run_mission(initial_altitude=goal[1]))

    
    #env.initialize_lidar()
    #env.load_point_cloud()
    '''
    env.state = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
    env.state_offset = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
    env.generate_path()

    

    for point in env.path:
        print(env.xyz_to_NED(point[0:4] + env.state_offset[0:4]))
    '''
    
    env.initialize_lidar()
    env.load_point_cloud()

    env.state = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
    env.state_offset = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
    env.generate_path()

    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot(env.path[:,0], env.path[:,1], env.path[:,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(env.point_cloud[:,0],env.point_cloud[:,1],env.point_cloud[:,2])

    plt.savefig('img.png')
    
    
    
    

    

