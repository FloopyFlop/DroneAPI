import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Any, Dict, Type, Optional, Union

from trajectory_planning.path_planner import pathPlanner

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

        self.global_state = None
        self.global_path_state = None
        self.global_path = None
        self.rel_path = None

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

    
        



    
        

    
