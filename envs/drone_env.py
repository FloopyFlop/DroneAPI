import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Type, Optional, Union, List
import copy
from copy import deepcopy
from scipy import interpolate
import asyncio
from mavsdk import System
from mavsdk.offboard import (PositionNedYaw, VelocityNedYaw, OffboardError)
import pyrealsense2 as rs
from multiprocessing import Process, Value, Array

def gaussian_prob(
            x: List[float],
            mu: List[float],
            std: List[float],
) -> List[float]:
      '''
      given an array of means, standard deviations, and values x
      calculate the gaussian probability of values x
      '''
      prob = prob = np.zeros((len(x),))
      if np.linalg.norm(std) == 0 and np.linalg.norm(mu-x) > 0:
            return prob
      else: 
            for i in range(len(x)):
                  if std[i] == 0:
                        continue
                  else:
                        exponent = np.around(-0.5*np.square(np.divide((x[i]-mu[i]),std[i])), decimals=8)
                        prob[i] = np.divide(1,std[i]*np.sqrt(np.pi*2))*np.exp(exponent)
      return prob

class polarHistogram3D():

    def __init__(
            self,
            radius: float = 1,
            layers: int = 1,
            angle_sections: float = 36,
            probability_tolerance: float = 0.05,
            distance_tolerance: float = 0.2,
    ):
        self.points = None
        self.radius = radius
        self.layers = layers + 1 #always check 1 layer ahead
        #self.max_bin_arc = 2*np.pi*radius/angle_sections
        self.layer_depth = self.radius/self.layers
        self.probability_tol = probability_tolerance
        self.distance_tol = distance_tolerance

        self.sections = angle_sections
        self.range = 2*np.pi/self.sections
        self.histogram3D = np.zeros((self.sections,self.sections,self.layers,7)) #initialize the histogram
        self.refrerence_histogram3D = np.zeros((self.sections,self.sections,3)) #center points of each bin on unit ball
        self.initialize_reference_histogram3D()
        self.histogram_calc_storage = None

    def convert_cartesian_to_polar(
            self,
            point: List[float],
    ):
        theta1 = np.arctan2(point[1],point[0]) #angle between +x-axis and point vector
        theta2 = np.arctan2(point[2],point[0]) #angle between xy-plane and point vector (azimuth)

        #make sure angle is '+'
        if theta1 < 0:
            theta1 = 2*np.pi + theta1
        if theta2 < 0:
            theta2 = 2*np.pi + theta2

        dist = np.linalg.norm(point)

        return theta1,theta2,dist
    
    def convert_polar_to_bin(
            self,
            point: List[float],
    ) -> Tuple[int,int,int]:
        theta = int(point[0]//self.range)
        phi = int(point[1]//self.range)
        layer = int(point[2]//self.layer_depth)

        if theta == self.sections:
            theta -= 1
        if phi == self.sections:
            phi -= 1

        return theta, phi, layer
    
    def convert_cartesian_to_bin(
            self,
            point: List[float],
    ) -> Tuple[int,int,int]:
        theta1 = np.arctan2(point[1],point[0]) #angle between +x-axis and point vector
        theta2 = np.arctan2(point[2],point[0]) #angle between xy-plane and point vector (azimuth)

        #make sure angle is '+'
        if theta1 < 0:
            theta1 = 2*np.pi + theta1
        if theta2 < 0:
            theta2 = 2*np.pi + theta2

        dist = np.linalg.norm(point)

        theta = int(theta1//self.range)
        phi = int(theta2//self.range)
        layer = int(dist//self.layer_depth)

        return theta, phi, layer

    def get_reference_point_from_bin(
            self,
            bin: List[int],
            layer: int = 0,
    ) -> List[float]:

        return self.refrerence_histogram3D[int(bin[0])][int(bin[1])] * (self.layer_depth * (0.5+layer))
    
    def get_target_point_from_bin(
            self,
            bin: List[int],
            goal: List[float],
            layer: int = 0,
    ) -> Tuple[List[float], bool]:
        '''
        check if goal is inside chosen bin, 
        if goal is inside bin and within range -> return goal
        if goal is inside bin and within range -> return goal vector w/ appropriate distance from center
        '''

        theta1,theta2,dist = self.convert_cartesian_to_polar(goal)
        if int(theta1//self.range) == int(bin[0]) and int(theta2//self.range) == int(bin[1]):
            if np.linalg.norm(goal) < (self.layer_depth * (0.5+layer)):
                return goal, True
            else:
                return goal/np.linalg.norm(goal) * (self.layer_depth * (0.5+layer)), False
        else:
            return self.refrerence_histogram3D[int(bin[0])][int(bin[1])] * (self.layer_depth * (0.5+layer)), False
    
    def get_bin_from_index(
            self,
            bin: List[int],
    ) -> List[float]:
        
        return self.histogram3D[int(bin[0])][int(bin[1])]
    
    def reset_histogram(self) -> None:
        self.histogram3D[:] = 0

    def input_points(
            self, 
            points: List[List[float]],
            points_min: int = 1,
    ) -> None:
        #t0 = time.time()
        self.points = points
        self.histogram3D[:] = 0
        self.histogram_calc_storage = np.zeros((self.sections,self.sections,self.layers,3))
        a = 0

        for point in points:
            theta1,theta2,dist = self.convert_cartesian_to_polar(point)

            if dist > self.radius:
                next
            else:
                layer = int(dist//self.layer_depth)

                #bin_state = self.histogram3D[int(theta1//self.range)][int(theta2//self.range)][layer]
                bin_center = self.get_reference_point_from_bin(bin=[int(theta1//self.range),int(theta2//self.range)],layer=layer)
                self.histogram3D[int(theta1//self.range)][int(theta2//self.range)][layer][0:3] += point
                self.histogram3D[int(theta1//self.range)][int(theta2//self.range)][layer][3:6] += np.square(point) 
                self.histogram3D[int(theta1//self.range)][int(theta2//self.range)][layer][6] += 1
                self.histogram_calc_storage[int(theta1//self.range)][int(theta2//self.range)][layer] += point

                
                '''
                #only save the closest point to center in each bin
                if dist < self.histogram3D[int(theta1//self.range)][int(theta2//self.range)][layer][3] or self.histogram3D[int(theta1//self.range)][int(theta2//self.range)][layer][3] == 0:
                    self.histogram3D[int(theta1//self.range)][int(theta2//self.range)][layer] = [point[0],point[1],point[2],dist]
                '''
        '''
        Calculate the center of point cloud within each bin (average of x,y,z)
        and the standard dev. of the cloud within each bin (in x,y,z) to calculate
        a gaussian probability field of the chance there is an obstacle 
        '''
        for i,section1 in enumerate(self.histogram3D):
            for j,section2 in enumerate(section1):
                for k,layer in enumerate(section2):
                    if layer[6] == 0:
                        continue
                    elif layer[6] < points_min:
                        layer[:] = 0
                    else:
                        layer[0:3] /= layer[6]
                        layer[3:6] += np.multiply(-self.histogram_calc_storage[i][j][k]*2,layer[0:3]) + np.multiply(layer[6],np.square(layer[0:3]))
                        layer[3:6] /= layer[6]
                        layer[3:6] = np.sqrt(layer[3:6])
        #print(time.time() - t0)

    def initialize_reference_histogram3D(self) -> None:
        '''
        create a polar historgram that contains the xyz 
        coordinates of the centerpoint of each bin w distance of 1
        '''
        for i in range(self.sections):
            for j in range(self.sections):
                theta1 = i*self.range + self.range/2
                theta2 = j*self.range + self.range/2
                
                x = np.cos(theta2)*np.cos(theta1)
                y = np.cos(theta2)*np.sin(theta1)
                z = np.sin(theta2)

                self.refrerence_histogram3D[i][j] = [x,y,z]

    def sort_candidate_bins(
            self,
            point: List[float],
            layer: int = 0,
            previous: List[int] = None,
            previous2: List[int] = None,
    ) -> List[List[float]]:

        sorted_bins = []
        #theta1,theta2,layerp = self.convert_cartesian_to_polar(point)
        for i in range(self.sections):
            for j in range(self.sections):
                if (self.histogram3D[i][j][layer][0:3] == [0,0,0]).all():
                    if previous is None:
                        angle = np.arccos(np.dot(point[0:3],self.refrerence_histogram3D[i][j]) / (np.linalg.norm(point[0:3])*np.linalg.norm(self.refrerence_histogram3D[i][j])))
                        cost = angle
                    else:
                        previous_point, filler = self.get_target_point_from_bin(bin=previous,goal=point[0:3],layer=layer-1)
                        current_point, filler = self.get_target_point_from_bin(bin=[i,j],goal=point[0:3],layer=layer)
                        angle1 = np.arccos(np.clip(np.dot(point[0:3]-previous_point,current_point-previous_point) / (np.linalg.norm(point[0:3]-previous_point)*np.linalg.norm(current_point-previous_point)),-1,1))
                        cost = angle1
                        #if i == int(theta1) and j == int(theta2):
                        #    print(cost)
                        if previous2 is not None:
                            previous_point2, filler = self.get_target_point_from_bin(bin=previous2,goal=point[0:3],layer=layer-2)
                            angle2 = np.arccos(np.clip(np.dot(previous_point-previous_point2,current_point-previous_point) / (np.linalg.norm(previous_point-previous_point2)*np.linalg.norm(current_point-previous_point)),-1,1))
                            cost += angle2 * 0.2
                    sorted_bins.append([cost,i,j,layer])

        sorted_bins = np.array(sorted_bins)

        if sorted_bins.size == 0:
            return []
        else:
            return sorted_bins[sorted_bins[:, 0].argsort()]
        
    def check_obstacle_bins(
           self,
            point: List[float],
            bin: List[int],
            distance: float,
            layer: int = 0,
    ) -> bool:
        '''
        check obstacle bins in order of closest to furthest from point 
        if obstacle found but is further from distance, it is assumed
        all other obstacles will also be further from distance
        i.e. this WILL cause inaccurate min_distance if self.sections is low

        designed to be FAST, the slower version is sort_obstacle_bins()

        if * is the bin with the point of interest 
        first, we check:
        
                    + + +
                    + * +
                    + + +
        then, we check: 
                  
                  + + + + + 
                  +       +
                  +   *   +
                  +       +
                  + + + + +

        and so on...
        '''


        if all(b == 0 for b in self.histogram3D[:,:,layer,:].flatten()):
            return True
        else:
            theta = list(range(self.sections))
            phi = list(range(self.sections))

            theta.sort(key=lambda x: min(abs(bin[0]-x),abs(bin[0]-(x-self.sections))))
            phi.sort(key=lambda x: min(abs(bin[1]-x),abs(bin[1]-(x-self.sections))))
            theta.pop(0)
            phi.pop(0)

            row_flip = False
            column_flip = False
            last_pass = False

            #check the center bin first
            if (self.histogram3D[bin[0]][bin[1]][layer][0:3] != [0,0,0]).any():
                dist = np.linalg.norm(self.histogram3D[bin[0]][bin[1]][layer][0:3]-point)
                if dist < distance:
                    return False

            iter = int(np.ceil((self.sections-1)/2))
            for k in range(iter):
                if last_pass:
                    return True
                if k < iter-1:
                    start = k*2
                    end = k*2 + 2
                else:
                    if self.sections%2 == 1:
                        start = k*2
                        end = k*2 + 2
                    else:
                        start = k*2
                        end = k*2 + 1
                #check columns of square
                low = min(phi[start],phi[end-1])
                high = max(phi[start],phi[end-1])
                if low==high:
                    column = list(range(0,self.sections))
                else:
                    if column_flip:
                        column = list(range(0,low+1))+list(range(high,self.sections))
                    else:
                        column = list(range(phi[start],phi[end-1]+1))
                    if low == 0 or high == self.sections-1:
                        column_flip = True
                for i in theta[start:end]:
                    for j in column:
                        if (self.histogram3D[i][j][layer][0:3] != [0,0,0]).any():
                            dist = np.linalg.norm(self.histogram3D[i][j][layer][0:3]-point)
                            if dist < distance:
                                return False
                            else:
                                last_pass = True
                #check columns of square
                low = min(theta[start],theta[end-1])
                high = max(theta[start],theta[end-1])
                if low==high:
                    row = list(range(0,self.sections))
                else:
                    if row_flip:
                        row = list(range(0,low))+list(range(high+1,self.sections))
                    else:
                        row = list(range(low+1,high))
                    if low == 0 or high == self.sections-1:
                        row_flip = True
                for j in phi[start:end]:
                    for i in row:
                        if (self.histogram3D[i][j][layer][0:3] != [0,0,0]).any():
                            dist = np.linalg.norm(self.histogram3D[i][j][layer][0:3]-point)
                            if dist < distance:
                                return False 
                            else:
                                last_pass = True
                            
            return True

    
    def sort_obstacle_bins(
            self,
            point: List[float],
            bin: List[int],
            distance: float,
            layer: int = 0,
    ) -> List[List[float]]:

        sorted_bins = []

        for i in range(self.sections):
            for j in range(self.sections):
                if (self.histogram3D[i][j][layer][0:3] != [0,0,0]).any():
                    dist = np.linalg.norm(self.histogram3D[i][j][layer][0:3]-point)
                    sorted_bins.append([dist,i,j,layer])
        
        sorted_bins = np.array(sorted_bins)
        if sorted_bins.size == 0:
            return []
        else:
            return sorted_bins#[sorted_bins[:, 0].argsort()]
        
    def check_point_safety(
            self,
            min_distance: float,
            point: List[float],           
    ) -> bool:
        #t0 = time.time()

        theta1,theta2,dist = self.convert_cartesian_to_polar(point)
        b1,b2,l = self.convert_polar_to_bin([theta1,theta2,dist])
        bin = [b1,b2,l]
        fs = 0.9 #factor of safety, point must be more than 10% closer to min_distance to change route

        if dist > self.radius:
            return True
        else:
            layer = int(dist//self.layer_depth)

        layers = range(self.layers)

        obstacle_bins = self.sort_obstacle_bins(point=point,bin=bin,distance=min_distance,layer=layers[0])

        for i in layers[1:]:
            temp_obstacle_bins = self.sort_obstacle_bins(point=point,bin=bin,distance=min_distance, layer=i)
            if len(obstacle_bins) == 0:
                obstacle_bins = temp_obstacle_bins
            else:
                if(len(temp_obstacle_bins)) == 0:
                    continue
                else:
                    obstacle_bins = np.vstack((obstacle_bins,temp_obstacle_bins))

        if len(obstacle_bins) > 0:
            obstacle_bins = obstacle_bins[obstacle_bins[:, 0].argsort()]

        for bad_bin in obstacle_bins:
            obstacle = self.histogram3D[int(bad_bin[1])][int(bad_bin[2])][int(bad_bin[3])][0:3]
            obstacle_std = self.histogram3D[int(bad_bin[1])][int(bad_bin[2])][int(bad_bin[3])][3:6]
            obstacle_probability = gaussian_prob(mu=obstacle, std=obstacle_std, x=point-obstacle)
            #if gaussian dist. is not in the same dimensions as goal (std dev of x,y, or z is0),
            #there is no probability of there being an obstacle, else, it is distance from 
            #center of the ellipsoide
            zeros = np.where(obstacle_probability == 0)[0]
            if zeros.size != 0:
                for zero in zeros:
                    if abs(point[zero]-obstacle[zero]) > 0:
                        fobstacle_probability = 0
                        break
            else:
                fobstacle_probability = min(obstacle_probability)
            if fobstacle_probability > self.probability_tol or np.linalg.norm(point-obstacle) < min_distance*fs:
                return False  
        #print(time.time() - t0)
        return True

    def confirm_candidate_distance(
            self,
            min_distance: float,
            bin: List[int],
            goal: List[float],
            layer: int = 0,
            past_bin: Optional[List[int]] = None,
    ) -> bool:
        '''
        Checks all obstacle bins and confirms that no obstacle
        is closer than min_distance to the centerline of the
        candidate bin.
        '''

        center_point, filler = self.get_target_point_from_bin(bin=bin,goal=goal[0:3],layer=layer)
        theta1,theta2,dist = self.convert_cartesian_to_polar(center_point)

        if dist < min_distance:
            layer0 = 0
        else:
            layer0 = int((dist-min_distance)//self.layer_depth)

        if (dist + min_distance) > self.radius:
            layerN = self.layers
        else:
            layerN = int(np.ceil((dist+min_distance)/self.layer_depth))
            
        layers = range(layer0,layerN)

        b1,b2,l = self.convert_polar_to_bin([theta1,theta2,dist])

        for i in layers:
            safe = self.check_obstacle_bins(point=center_point,bin=[b1,b2,l],distance=min_distance, layer=i)
            if not safe:
                return False
        '''
        for bad_bin in obstacle_bins:
            #Calculate probability of an obstacle being at the new goal
            if bad_bin[0] < min_distance:
                return False
            else:
                obstacle = self.histogram3D[int(bad_bin[1])][int(bad_bin[2])][int(bad_bin[3])][0:3]
                obstacle_std = self.histogram3D[int(bad_bin[1])][int(bad_bin[2])][int(bad_bin[3])][3:6]
                obstacle_probability = gaussian_prob(mu=obstacle, std=obstacle_std, x=center_point-obstacle)
                #if gaussian dist. is not in the same dimensions as goal (std dev of x,y, or z is0),
                #there is no probability of there being an obstacle, else, it is distance from 
                #center of the ellipsoide
                zeros = np.where(obstacle_probability == 0)[0]
                if zeros.size != 0:
                    for zero in zeros:
                        if abs(center_point[zero]-obstacle[zero]) > 0:
                            fobstacle_probability = 0
                            break
                else:
                    fobstacle_probability = min(obstacle_probability)
                    
                if fobstacle_probability > self.probability_tol:
                    return False    
        '''
        '''
        Check if path from previous chose ben to current bin intersects a bin with obstacles
        ONLY necissary when running algorithm w/ more than 1 layer
        '''
        #if past_bin is not None:
        #    past_bin_center, filler = self.get_target_point_from_bin(bin=past_bin,goal=goal[0:3],layer=layer-1)
        #    n = int(np.linalg.norm(past_bin_center-center_point)//self.distance_tol + 1)
        #    check_positions = np.linspace(past_bin_center,center_point,n)
        #    for position in check_positions:
        #        theta1,theta2,dist = self.convert_cartesian_to_polar(position)
        #        layer = int(dist//self.layer_depth)
        #        if (self.histogram3D[int(theta1//self.range)][int(theta2//self.range)][layer][0:3] != [0,0,0]).any():
        #            return False
        return True
    

        #bin_min_distance = min_distance + self.max_bin_arc
        '''
        obstacle_bins = self.sort_obstacle_bins(point=self.refrerence_histogram3D[int(bin[0])][int(bin[1])], layer=layer)
        center_line = self.refrerence_histogram3D[int(bin[0])][int(bin[1])] - off_set

        for bad_bin in obstacle_bins:
            if bad_bin[0] > np.pi/2:
                break
            else:
                obstacle = self.histogram3D[int(bad_bin[1])][int(bad_bin[2])][layer][0:3] - off_set
                distance = np.linalg.norm(np.cross(center_line,-obstacle))/np.linalg.norm(center_line)
                if distance < min_distance:
                    return False
                
        return True
        '''

class basePathPlanner():

    def __init__(
            self,
            path_planning_algorithm: str,
            kwargs: Dict[str, Any],
    ):
        self.algorithm = getattr(self,path_planning_algorithm)(**kwargs)

    class VFH():
        '''
        Vector Field Histogram Method in #D
        Original VFH publication:
        https://ieeexplore.ieee.org/document/846405
        '''

        def __init__(
                self,
                radius: float = 1,
                layers: int = 1,
                iterations: int = 1,
                angle_sections: float = 8,
                min_obstacle_distance: float = 1,
                probability_tolerance: float = 0.05,
                distance_tolerance: float = 0.2,
        ):

            self.histogram = polarHistogram3D(radius=radius, 
                                              layers=layers, 
                                              angle_sections=angle_sections,
                                              probability_tolerance=probability_tolerance,
                                              distance_tolerance=distance_tolerance,
                                              )
            self.min_distance = min_obstacle_distance
            self.iterations = iterations
            self.layers = layers
            self.radius = radius

        def input_points(
            self, 
            points: List[List[float]],
            points_min: int = 1,
        ) -> None:
            self.histogram.input_points(points=points, points_min=points_min)

        def reset_map(self) -> None:
            self.histogram.reset_histogram()

        def get_layer_size(self) -> float:
            return self.histogram.layer_depth

        def compute_next_point(
                self,
                points: List[List[float]],
                goal: List[float],
                points_min: int = 1,
        ) -> List[float]:
            
            
            off_set = [0,0,0]
            computed_points = [off_set]
            filler = np.zeros((goal.size-3,))

            past_bin = None
            past_bin2 = None
            done = False

            #t0 = time.time()
            for i in range(self.iterations):
                self.histogram.input_points(points=np.array(points)-off_set, points_min=points_min)
                #t0 = time.time()
                for j in range(self.layers):
                    #t0 = time.time()
                    candidates = self.histogram.sort_candidate_bins(
                                                                point=np.array(goal)-np.concatenate((off_set,filler)),
                                                                layer=j, 
                                                                previous=past_bin,
                                                                previous2=past_bin2,
                                                                )   
                    for i,candidate in enumerate(candidates):
                        if self.histogram.confirm_candidate_distance(min_distance=self.min_distance,
                                                                    bin=[candidate[1],candidate[2]],
                                                                    layer=j,
                                                                    past_bin=past_bin,
                                                                    goal=np.array(goal)-np.concatenate((off_set,filler)),
                                                                    ):
                            if self.layers > 1:
                                if j >= 1:
                                    past_bin2 = past_bin
                                past_bin = [int(candidate[1]),int(candidate[2])]
                            target, done = self.histogram.get_target_point_from_bin(bin=[candidate[1],candidate[2]],goal=goal[0:3],layer=j)
                            computed_points.append(target+off_set)
                            break
                    if done:
                        break
                    #print(time.time() - t0,'layer')
                #print(time.time() - t0)
                if self.iterations > 1:
                    off_set = computed_points[-1]
            #print(time.time() - t0)
            return np.array(computed_points)
        
        def check_goal_safety(
                self,
                goal: List[float],
        ) -> bool:
            safe = self.histogram.check_point_safety(min_distance=self.min_distance, point=goal)
            return safe

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

class lidar():

    def __init__(
        self,
        point_cloud_size: int = 100,
    ) -> None:

        self.collect_data = True
        self.pipeline = None
        self.pc = None
        self.decimate = None
        self.point_cloud_size = point_cloud_size
        #self.point_cloud = Array('d', np.zeros((self.point_cloud_size)))
        self.point_cloud = np.zeros((self.point_cloud_size,3))
        self.point_cloud_size = point_cloud_size

        self.initialize_pipeline()
        self.update_process = None

    def initialize_pipeline(
        self,
    ) -> None:

        self.pipeline = rs.pipeline()
        config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
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
        #sensor_dep.set_option(rs.option.laser_power, 50)
        #sensor_dep.set_option(rs.option.receiver_gain, 18)
        #sensor_dep.set_option(rs.option.confidence_threshold, 3)

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
    ) -> None:

        #start_time = time.time()
        frames = self.pipeline.wait_for_frames()
        
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
        point_cloud_temp = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        
        #for i in range(len(point_cloud)):
        #    point_cloud[i] = point_cloud_temp[i]
        for i in reversed(range(len(point_cloud_temp))):
            if np.linalg.norm(point_cloud_temp[i]) < 0.1:
                point_cloud_temp = np.delete(point_cloud_temp,i,0)
        
        #adjust data if LiDAR is upside-down
        point_cloud_temp[:,0:1] *= -1

        return point_cloud_temp
        

    def start_update(
        self,
    ) -> None:

        self.update_process = Process(target=self.update, args=(self.point_cloud,)) 
        self.update_process.start()

    def stop_update(
        self,
    ) -> None:
        self.update_process.terminate()
        self.update_process.join()

    def update(
        self,
        point_cloud: Array
    ) -> None:

        '''
        always update the point cloud with most recent lidar date
        '''

        while self.collect_data:
            pc = self.update_pc()
            self.point_cloud = pc

    def get_point_cloud(
        self,
    ) -> None:
        return deepcopy(self.point_cloud[:])

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
    
    
    
    

    

