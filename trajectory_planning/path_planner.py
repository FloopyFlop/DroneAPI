import sys
sys.path.insert(1, 'c:/Users/Cameron Mehlman/Documents/Magpie')

import numpy as np
from typing import Any, Dict, Type, Optional, List, Tuple
import copy
import time
from scipy import interpolate

from trajectory_planning.base_path_planner import basePathPlanner


class pathPlanner(basePathPlanner):

    def __init__(
            self,
            path_planning_algorithm: str, #VFH
            kwargs: Dict[str, Any],
            goal_state: Optional[List[float]] = None,
            max_distance: float = 0.5,
            interpolation_method: str = 'linear',
            avg_speed: float = 0.5,
            n: int = 50,
    ):
        self.goal = goal_state
        self.avg_speed = avg_speed
        self.max_distance = max_distance
        self.interpolator = getattr(self,interpolation_method+'_interpolator')
        self.interpolation_method = interpolation_method
        self.n = n #for spline interpolation, determines how many points to compute

        super().__init__(
            path_planning_algorithm=path_planning_algorithm,
            kwargs=kwargs,
        )

    def set_goal_state(
            self,
            goal_state: List[float],
    ) -> None:
        self.goal = goal_state

    def update_point_cloud(
            self,
            point_cloud: List[List[float]],
            points_min: int = 1,
    ) -> None:
        if point_cloud is None:
            self.algorithm.reset_map()
        else:
            self.algorithm.input_points(points=point_cloud, points_min=points_min)

    def check_goal_safety(
            self,
            goals: List[List[float]],
            state: List[float] = None,
    ) -> bool:
        for goal in goals:
            goal = goal[0:3]
            safe = self.algorithm.check_goal_safety(goal - state)
            if not safe:
                return False
        return True


    def compute_desired_path(
            self,
            state: List[float],
            point_cloud: Optional[List[List[float]]] = None,
            points_min: int = 1,
    ) -> List[float]:

        state_offset = np.zeros((state.size,))
        state_offset[0:3] = state[0:3]
        current_state = state

        if point_cloud is None:
            next_location = self.goal[0:3] - state[0:3]
            if np.linalg.norm(next_location) < 0.25*self.algorithm.radius:
                next_location = [current_state[0:3] - state[0:3], next_location]
            else:
                next_location = [current_state[0:3] - state[0:3], next_location/np.linalg.norm(next_location)*0.75*self.algorithm.radius]
        else:
            goal = self.goal-state_offset
            if np.linalg.norm(goal[0:3]) < self.algorithm.get_layer_size():
                next_location = [current_state - state_offset, goal]
            else:
                #t0 = time.time()
                next_location = self.algorithm.compute_next_point(points=point_cloud, goal=goal, points_min=points_min)
                #print(time.time() - t0)
                
        if self.interpolation_method == 'linear':
            path = np.array([])

            for i in range(len(next_location)):

                vel_vector = next_location[i][:] - current_state[0:3]
                vel_vector = vel_vector/np.linalg.norm(vel_vector) * self.avg_speed
        
                next_state = copy.deepcopy(state_offset)
                next_state[0:3] += next_location[i][:]
                next_state[3:6] = vel_vector
                new_path = self.interpolator([current_state,next_state])
                if i==0:
                    path = new_path
                else:
                    path = np.concatenate((path,new_path),axis=0)
                current_state = next_state
        elif self.interpolation_method == 'spline':
            path = self.spline_interpolator(next_location+state[0:len(next_location[0])],pts=self.n)
            if len(path) == 0:
                return []
            else:
                filler = np.zeros((len(path),len(state)-3))
                path = np.concatenate((path,filler),axis=1)
        return path

    
    def linear_interpolator(
            self,
            trajectory: List[List[float]],
    ) -> List[float]:
        
        start = np.array(trajectory[0][0:3])
        end = np.array(trajectory[-1][0:3])

        n = int(np.linalg.norm(start-end)//self.max_distance + 1)

        if n <= 1:
            return np.array(trajectory)
        else:
            return np.linspace(trajectory[0],trajectory[-1],n)
    
    def spline_interpolator(
            self,
            trajectory: List[List[float]],
            pts: int = 50
    ) -> List[float]:

        k = 2
        if k >= len(trajectory):
            k = len(trajectory)-1

        tck, u = interpolate.splprep([trajectory[:,0],trajectory[:,1], trajectory[:,2]],k=k, s=2)
        u_fine = np.linspace(0,1,pts)
        x, y, z = interpolate.splev(u_fine, tck)
        points = []

        for i,px in enumerate(x):
            points.append([px,y[i],z[i]])
            
        return points


    


