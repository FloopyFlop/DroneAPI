import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, Union, List, Tuple
import copy
from scipy import interpolate
import asyncio

from mavsdk import System
from mavsdk.offboard import (PositionNedYaw, VelocityNedYaw, OffboardError)


# =========================
# Utilities
# =========================

def _to_np(a: Union[List[float], np.ndarray]) -> np.ndarray:
    return a if isinstance(a, np.ndarray) else np.array(a, dtype=float)


def gaussian_prob(x: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Vectorized Gaussian PDF per-dimension.
    Returns per-dimension probabilities (not joint).
    """
    x = _to_np(x)
    mu = _to_np(mu)
    std = _to_np(std)
    prob = np.zeros_like(x, dtype=float)

    # If std is zero in any dim, that dim contributes 0 unless x==mu there.
    nz = std != 0
    if not np.any(nz):
        return prob
    z = (x[nz] - mu[nz]) / std[nz]
    exponent = -0.5 * np.square(z)
    prob[nz] = (1.0 / (std[nz] * np.sqrt(2.0 * np.pi))) * np.exp(exponent)
    return prob


# =========================
# VFH in 3D (your original, tidied)
# =========================

class polarHistogram3D:
    def __init__(
        self,
        radius: float = 1.0,
        layers: int = 1,
        angle_sections: int = 36,
        probability_tolerance: float = 0.05,
        distance_tolerance: float = 0.2,
    ):
        self.points = None
        self.radius = float(radius)
        self.layers = int(layers) + 1  # always check 1 layer ahead
        self.layer_depth = self.radius / self.layers
        self.probability_tol = float(probability_tolerance)
        self.distance_tol = float(distance_tolerance)

        self.sections = int(angle_sections)
        self.range = 2 * np.pi / self.sections
        self.histogram3D = np.zeros((self.sections, self.sections, self.layers, 7))
        self.reference_histogram3D = np.zeros((self.sections, self.sections, 3))
        self._initialize_reference_histogram3D()
        self.histogram_calc_storage = None

    @staticmethod
    def _cart_to_polar(point: np.ndarray) -> Tuple[float, float, float]:
        # NOTE: keeping your angle definition for compatibility
        theta1 = np.arctan2(point[1], point[0])  # angle vs +x in xy
        theta2 = np.arctan2(point[2], point[0])  # "azimuth" (custom)
        if theta1 < 0:
            theta1 += 2 * np.pi
        if theta2 < 0:
            theta2 += 2 * np.pi
        dist = float(np.linalg.norm(point))
        return theta1, theta2, dist

    def convert_polar_to_bin(self, polar: List[float]) -> Tuple[int, int, int]:
        theta = int(polar[0] // self.range)
        phi = int(polar[1] // self.range)
        layer = int(polar[2] // self.layer_depth)
        if theta == self.sections:
            theta -= 1
        if phi == self.sections:
            phi -= 1
        return theta, phi, layer

    def convert_cartesian_to_bin(self, point: np.ndarray) -> Tuple[int, int, int]:
        theta1, theta2, dist = self._cart_to_polar(point)
        theta = int(theta1 // self.range)
        phi = int(theta2 // self.range)
        layer = int(dist // self.layer_depth)
        return theta, phi, layer

    def get_reference_point_from_bin(self, bin_idx: List[int], layer: int = 0) -> np.ndarray:
        return self.reference_histogram3D[int(bin_idx[0]), int(bin_idx[1])] * (self.layer_depth * (0.5 + layer))

    def get_target_point_from_bin(
        self,
        bin_idx: List[int],
        goal: np.ndarray,
        layer: int = 0,
    ) -> Tuple[np.ndarray, bool]:
        theta1, theta2, dist = self._cart_to_polar(goal)
        if int(theta1 // self.range) == int(bin_idx[0]) and int(theta2 // self.range) == int(bin_idx[1]):
            if np.linalg.norm(goal) < (self.layer_depth * (0.5 + layer)):
                return goal, True
            return goal / np.linalg.norm(goal) * (self.layer_depth * (0.5 + layer)), False
        return self.reference_histogram3D[int(bin_idx[0]), int(bin_idx[1])] * (self.layer_depth * (0.5 + layer)), False

    def reset_histogram(self) -> None:
        self.histogram3D[:] = 0

    def input_points(self, points: np.ndarray, points_min: int = 1) -> None:
        self.points = points
        self.histogram3D[:] = 0
        self.histogram_calc_storage = np.zeros((self.sections, self.sections, self.layers, 3))

        for point in points:
            theta1, theta2, dist = self._cart_to_polar(point)
            if dist > self.radius:
                continue  # BUGFIX: 'next' -> 'continue'
            layer = int(dist // self.layer_depth)
            self.histogram3D[int(theta1 // self.range), int(theta2 // self.range), layer, 0:3] += point
            self.histogram3D[int(theta1 // self.range), int(theta2 // self.range), layer, 3:6] += np.square(point)
            self.histogram3D[int(theta1 // self.range), int(theta2 // self.range), layer, 6] += 1
            self.histogram_calc_storage[int(theta1 // self.range), int(theta2 // self.range), layer] += point

        # finalize mean + std per bin
        for i in range(self.sections):
            for j in range(self.sections):
                for k in range(self.layers):
                    layer = self.histogram3D[i, j, k]
                    if layer[6] == 0:
                        continue
                    if layer[6] < points_min:
                        layer[:] = 0
                        continue
                    # mean
                    layer[0:3] /= layer[6]
                    # std
                    layer[3:6] += (
                        -2 * self.histogram_calc_storage[i, j, k] * layer[0:3]
                        + layer[6] * np.square(layer[0:3])
                    )
                    layer[3:6] /= layer[6]
                    layer[3:6] = np.sqrt(layer[3:6])

    def _initialize_reference_histogram3D(self) -> None:
        for i in range(self.sections):
            for j in range(self.sections):
                theta1 = i * self.range + self.range / 2
                theta2 = j * self.range + self.range / 2
                x = np.cos(theta2) * np.cos(theta1)
                y = np.cos(theta2) * np.sin(theta1)
                z = np.sin(theta2)
                self.reference_histogram3D[i, j] = [x, y, z]

    def sort_candidate_bins(
        self,
        point: np.ndarray,
        layer: int = 0,
        previous: Optional[List[int]] = None,
        previous2: Optional[List[int]] = None,
    ) -> np.ndarray:
        sorted_bins = []
        for i in range(self.sections):
            for j in range(self.sections):
                if (self.histogram3D[i, j, layer, 0:3] == [0, 0, 0]).all():
                    if previous is None:
                        angle = np.arccos(
                            np.clip(
                                np.dot(point[0:3], self.reference_histogram3D[i, j])
                                / (np.linalg.norm(point[0:3]) * np.linalg.norm(self.reference_histogram3D[i, j])),
                                -1, 1,
                            )
                        )
                        cost = angle
                    else:
                        prev_pt, _ = self.get_target_point_from_bin(previous, goal=point[0:3], layer=max(0, layer - 1))
                        cur_pt, _ = self.get_target_point_from_bin([i, j], goal=point[0:3], layer=layer)
                        angle1 = np.arccos(
                            np.clip(
                                np.dot(point[0:3] - prev_pt, cur_pt - prev_pt)
                                / (np.linalg.norm(point[0:3] - prev_pt) * np.linalg.norm(cur_pt - prev_pt)),
                                -1, 1,
                            )
                        )
                        cost = angle1
                        if previous2 is not None and layer >= 2:
                            prev_pt2, _ = self.get_target_point_from_bin(previous2, goal=point[0:3], layer=layer - 2)
                            angle2 = np.arccos(
                                np.clip(
                                    np.dot(prev_pt - prev_pt2, cur_pt - prev_pt)
                                    / (np.linalg.norm(prev_pt - prev_pt2) * np.linalg.norm(cur_pt - prev_pt)),
                                    -1, 1,
                                )
                            )
                            cost += 0.2 * angle2
                    sorted_bins.append([cost, i, j, layer])

        sorted_bins = np.array(sorted_bins)
        if sorted_bins.size == 0:
            return np.array([])
        return sorted_bins[sorted_bins[:, 0].argsort()]

    def sort_obstacle_bins(
        self, point: np.ndarray, bin_idx: List[int], distance: float, layer: int = 0
    ) -> np.ndarray:
        sorted_bins = []
        for i in range(self.sections):
            for j in range(self.sections):
                if (self.histogram3D[i, j, layer, 0:3] != [0, 0, 0]).any():
                    dist = np.linalg.norm(self.histogram3D[i, j, layer, 0:3] - point)
                    sorted_bins.append([dist, i, j, layer])
        sorted_bins = np.array(sorted_bins)
        return sorted_bins if sorted_bins.size else np.array([])

    def check_obstacle_bins(self, point: np.ndarray, bin_idx: List[int], distance: float, layer: int = 0) -> bool:
        """
        Fast windowed check around the bin to ensure all obstacles are farther than 'distance'
        """
        if np.all(self.histogram3D[:, :, layer, :].flatten() == 0):
            return True

        theta_ord = list(range(self.sections))
        phi_ord = list(range(self.sections))

        theta_ord.sort(key=lambda x: min(abs(bin_idx[0] - x), abs(bin_idx[0] - (x - self.sections))))
        phi_ord.sort(key=lambda x: min(abs(bin_idx[1] - x), abs(bin_idx[1] - (x - self.sections))))
        theta_ord.pop(0)
        phi_ord.pop(0)

        row_flip = False
        column_flip = False
        last_pass = False

        # center first
        if (self.histogram3D[bin_idx[0], bin_idx[1], layer, 0:3] != [0, 0, 0]).any():
            dist = np.linalg.norm(self.histogram3D[bin_idx[0], bin_idx[1], layer, 0:3] - point)
            if dist < distance:
                return False

        iterations = int(np.ceil((self.sections - 1) / 2))
        for k in range(iterations):
            if last_pass:
                return True
            if k < iterations - 1:
                start = k * 2
                end = k * 2 + 2
            else:
                start = k * 2
                end = k * 2 + (2 if self.sections % 2 == 1 else 1)

            # vertical sweep
            low = min(phi_ord[start], phi_ord[end - 1])
            high = max(phi_ord[start], phi_ord[end - 1])
            if low == high:
                columns = list(range(0, self.sections))
            else:
                if column_flip:
                    columns = list(range(0, low + 1)) + list(range(high, self.sections))
                else:
                    columns = list(range(phi_ord[start], phi_ord[end - 1] + 1))
                if low == 0 or high == self.sections - 1:
                    column_flip = True

            for i in theta_ord[start:end]:
                for j in columns:
                    if (self.histogram3D[i, j, layer, 0:3] != [0, 0, 0]).any():
                        dist = np.linalg.norm(self.histogram3D[i, j, layer, 0:3] - point)
                        if dist < distance:
                            return False
                        last_pass = True

            # horizontal sweep
            low = min(theta_ord[start], theta_ord[end - 1])
            high = max(theta_ord[start], theta_ord[end - 1])
            if low == high:
                rows = list(range(0, self.sections))
            else:
                if row_flip:
                    rows = list(range(0, low)) + list(range(high + 1, self.sections))
                else:
                    rows = list(range(low + 1, high))
                if low == 0 or high == self.sections - 1:
                    row_flip = True

            for j in phi_ord[start:end]:
                for i in rows:
                    if (self.histogram3D[i, j, layer, 0:3] != [0, 0, 0]).any():
                        dist = np.linalg.norm(self.histogram3D[i, j, layer, 0:3] - point)
                        if dist < distance:
                            return False
                        last_pass = True

        return True

    def check_point_safety(self, min_distance: float, point: np.ndarray) -> bool:
        theta1, theta2, dist = self._cart_to_polar(point)
        b1, b2, l = self.convert_polar_to_bin([theta1, theta2, dist])

        if dist > self.radius:
            return True

        layers = range(self.layers)
        obstacle_bins = self.sort_obstacle_bins(point=point, bin_idx=[b1, b2, l], distance=min_distance, layer=layers[0])
        for i in layers[1:]:
            temp = self.sort_obstacle_bins(point=point, bin_idx=[b1, b2, l], distance=min_distance, layer=i)
            if temp.size:
                obstacle_bins = temp if not obstacle_bins.size else np.vstack((obstacle_bins, temp))

        if obstacle_bins.size:
            obstacle_bins = obstacle_bins[obstacle_bins[:, 0].argsort()]

        fs = 0.9  # safety factor
        for bad_bin in obstacle_bins:
            obstacle = self.histogram3D[int(bad_bin[1]), int(bad_bin[2]), int(bad_bin[3]), 0:3]
            obstacle_std = self.histogram3D[int(bad_bin[1]), int(bad_bin[2]), int(bad_bin[3]), 3:6]
            p = gaussian_prob(x=point - obstacle, mu=obstacle, std=obstacle_std)
            zeros = np.where(p == 0)[0]
            if zeros.size != 0:
                for zero in zeros:
                    if abs(point[zero] - obstacle[zero]) > 0:
                        fprob = 0
                        break
            else:
                fprob = float(np.min(p))
            if fprob > self.probability_tol or np.linalg.norm(point - obstacle) < min_distance * fs:
                return False
        return True

    def confirm_candidate_distance(
        self,
        min_distance: float,
        bin_idx: List[int],
        goal: np.ndarray,
        layer: int = 0,
        past_bin: Optional[List[int]] = None,
    ) -> bool:
        center_point, _ = self.get_target_point_from_bin(bin_idx, goal=goal[0:3], layer=layer)
        theta1, theta2, dist = self._cart_to_polar(center_point)

        layer0 = 0 if dist < min_distance else int((dist - min_distance) // self.layer_depth)
        layerN = self.layers if (dist + min_distance) > self.radius else int(np.ceil((dist + min_distance) / self.layer_depth))

        b1, b2, l = self.convert_polar_to_bin([theta1, theta2, dist])

        for i in range(layer0, layerN):
            safe = self.check_obstacle_bins(point=center_point, bin_idx=[b1, b2, l], distance=min_distance, layer=i)
            if not safe:
                return False
        return True


class basePathPlanner:
    def __init__(self, path_planning_algorithm: str, kwargs: Dict[str, Any]):
        self.algorithm = getattr(self, path_planning_algorithm)(**kwargs)

    class VFH:
        def __init__(
            self,
            radius: float = 1,
            layers: int = 1,
            iterations: int = 1,
            angle_sections: int = 8,
            min_obstacle_distance: float = 1,
            probability_tolerance: float = 0.05,
            distance_tolerance: float = 0.2,
        ):
            self.histogram = polarHistogram3D(
                radius=radius,
                layers=layers,
                angle_sections=angle_sections,
                probability_tolerance=probability_tolerance,
                distance_tolerance=distance_tolerance,
            )
            self.min_distance = float(min_obstacle_distance)
            self.iterations = int(iterations)
            self.layers = int(layers)
            self.radius = float(radius)

        def input_points(self, points: np.ndarray, points_min: int = 1) -> None:
            self.histogram.input_points(points=points, points_min=points_min)

        def reset_map(self) -> None:
            self.histogram.reset_histogram()

        def get_layer_size(self) -> float:
            return self.histogram.layer_depth

        def compute_next_point(
            self, points: np.ndarray, goal: np.ndarray, points_min: int = 1
        ) -> np.ndarray:
            off_set = np.zeros(3)
            computed_points = [off_set.copy()]
            filler = np.zeros(max(0, goal.size - 3))

            past_bin = None
            past_bin2 = None
            done = False

            for _ in range(self.iterations):
                self.histogram.input_points(points=points - off_set, points_min=points_min)
                for j in range(self.layers):
                    candidates = self.histogram.sort_candidate_bins(
                        point=goal - np.concatenate((off_set, filler)),
                        layer=j,
                        previous=past_bin,
                        previous2=past_bin2,
                    )
                    for candidate in candidates:
                        if self.histogram.confirm_candidate_distance(
                            min_distance=self.min_distance,
                            bin_idx=[int(candidate[1]), int(candidate[2])],
                            layer=j,
                            past_bin=past_bin,
                            goal=goal - np.concatenate((off_set, filler)),
                        ):
                            if self.layers > 1 and j >= 1:
                                past_bin2 = past_bin
                            past_bin = [int(candidate[1]), int(candidate[2])]
                            target, done = self.histogram.get_target_point_from_bin(
                                bin_idx=[int(candidate[1]), int(candidate[2])], goal=goal[0:3], layer=j
                            )
                            computed_points.append(target + off_set)
                            break
                    if done:
                        break
                if self.iterations > 1:
                    off_set = computed_points[-1]
            return np.array(computed_points)

        def check_goal_safety(self, goal: np.ndarray) -> bool:
            return self.histogram.check_point_safety(min_distance=self.min_distance, point=goal)


class pathPlanner(basePathPlanner):
    def __init__(
        self,
        path_planning_algorithm: str,  # "VFH"
        kwargs: Dict[str, Any],
        goal_state: Optional[np.ndarray] = None,
        max_distance: float = 0.5,
        interpolation_method: str = "linear",
        avg_speed: float = 0.5,
        n: int = 50,
    ):
        self.goal = _to_np(goal_state) if goal_state is not None else None
        self.avg_speed = float(avg_speed)
        self.max_distance = float(max_distance)
        self.interpolator = getattr(self, interpolation_method + "_interpolator")
        self.interpolation_method = interpolation_method
        self.n = int(n)

        super().__init__(path_planning_algorithm=path_planning_algorithm, kwargs=kwargs)

    def set_goal_state(self, goal_state: Union[List[float], np.ndarray]) -> None:
        self.goal = _to_np(goal_state)

    def update_point_cloud(self, point_cloud: Optional[np.ndarray], points_min: int = 1) -> None:
        if point_cloud is None:
            self.algorithm.reset_map()
        else:
            self.algorithm.input_points(points=point_cloud, points_min=points_min)

    def check_goal_safety(self, goals: np.ndarray, state: Optional[np.ndarray] = None) -> bool:
        state = np.zeros(3) if state is None else _to_np(state)
        for goal in goals:
            g = _to_np(goal)[0:3]
            if not self.algorithm.check_goal_safety(g - state):
                return False
        return True

    def compute_desired_path(
        self,
        state: np.ndarray,
        point_cloud: Optional[np.ndarray] = None,
        points_min: int = 1,
    ) -> np.ndarray:
        state = _to_np(state)
        state_offset = np.zeros_like(state)
        state_offset[0:3] = state[0:3]
        current_state = state.copy()

        if point_cloud is None:
            next_location = self.goal[0:3] - state[0:3]
            if np.linalg.norm(next_location) < 0.25 * self.algorithm.radius:
                next_location = [current_state[0:3] - state[0:3], next_location]
            else:
                next_location = [
                    current_state[0:3] - state[0:3],
                    next_location / np.linalg.norm(next_location) * 0.75 * self.algorithm.radius,
                ]
        else:
            goal = self.goal - state_offset
            if np.linalg.norm(goal[0:3]) < self.algorithm.get_layer_size():
                next_location = [current_state - state_offset, goal]
            else:
                next_location = self.algorithm.compute_next_point(points=point_cloud, goal=goal, points_min=points_min)

        if self.interpolation_method == "linear":
            path = np.empty((0, len(state)))
            for i in range(len(next_location)):
                vel_vector = next_location[i][:] - current_state[0:3]
                if np.linalg.norm(vel_vector) == 0:
                    vel_vector = np.zeros(3)
                else:
                    vel_vector = vel_vector / np.linalg.norm(vel_vector) * self.avg_speed

                next_state = copy.deepcopy(state_offset)
                next_state[0:3] += next_location[i][:]
                next_state[3:6] = vel_vector
                new_path = self.linear_interpolator([current_state, next_state])
                path = new_path if i == 0 else np.concatenate((path, new_path), axis=0)
                current_state = next_state
        else:
            path_xyz = self.spline_interpolator(np.array(next_location) + state[0: len(next_location[0])], pts=self.n)
            if len(path_xyz) == 0:
                return np.empty((0, len(state)))
            filler = np.zeros((len(path_xyz), len(state) - 3))
            path = np.concatenate((np.array(path_xyz), filler), axis=1)
        return path

    def linear_interpolator(self, trajectory: List[np.ndarray]) -> np.ndarray:
        start = _to_np(trajectory[0][0:3])
        end = _to_np(trajectory[-1][0:3])
        n = int(np.linalg.norm(start - end) // self.max_distance + 1)
        if n <= 1:
            return np.array(trajectory)
        return np.linspace(trajectory[0], trajectory[-1], n)

    def spline_interpolator(self, trajectory: np.ndarray, pts: int = 50) -> List[List[float]]:
        k = 2
        if k >= len(trajectory):
            k = len(trajectory) - 1
        tck, u = interpolate.splprep([trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]], k=k, s=2)
        u_fine = np.linspace(0, 1, pts)
        x, y, z = interpolate.splev(u_fine, tck)
        return [[float(xi), float(yi), float(zi)] for xi, yi, zi in zip(x, y, z)]


# =========================
# Env layer (NED/XYZ, MAVSDK bindings)
# =========================

class MagpieEnv:
    """
    Low-level environment that talks to MAVSDK and exposes simple XYZ <-> NED conversions.
    """

    def __init__(
        self,
        planner: pathPlanner,
        goal: Union[List[float], np.ndarray],
        points_min: int = 1,
        wait_time: int = 7,
        position_tracking: bool = False,
    ):
        self.planner = planner
        self.goal = _to_np(goal)
        self.points_min = int(points_min)
        self.wait_time = int(wait_time)
        self.position_tracking = bool(position_tracking)

        self.point_cloud: Optional[np.ndarray] = None
        self.mission_running = False

        self.path: Optional[np.ndarray] = None
        self.lidar = None  # placeholder for sensors
        self.drone = System()

        self.state = np.zeros(12)
        self.state_offset = None  # xyz offset from initial NED origin
        self.time_step = 0

    def xyz_to_NED(self, xyz: Union[List[float], np.ndarray]) -> List[float]:
        xyz = _to_np(xyz)
        # x -> east, y -> up, z -> north  (custom mapping retained)
        NED = [0.0, 0.0, 0.0, 0.0]
        NED[0] = float(xyz[2])        # north
        NED[1] = float(xyz[0])        # east
        NED[2] = float(-xyz[1])       # down
        return NED

    async def get_position(self) -> List[float]:
        async for data in self.drone.telemetry.position_velocity_ned():
            return [data.position.north_m, data.position.east_m, data.position.down_m]

    async def update_state(self) -> None:
        pos = await self.get_position()
        # convert to xyz for planner
        self.state[0] = pos[1]
        self.state[1] = -pos[2]
        self.state[2] = pos[0]

    async def compute_offset(self) -> None:
        pos = await self.get_position()
        self.state_offset = np.zeros_like(self.state)
        self.state_offset[0] = pos[1]
        self.state_offset[1] = -pos[2]
        self.state_offset[2] = pos[0]

    async def set_new_position(self, NED: List[float], vel: Optional[List[float]] = None) -> None:
        print('--relative coordinates (xyz):', self.state[0:3] - self.state_offset[0:3])
        ned_now = await self.get_position()
        print('--NED now:', ned_now)
        print("-- Setting new position")
        if vel is None:
            await self.drone.offboard.set_position_ned(PositionNedYaw(NED[0], NED[1], NED[2], NED[3]))
        else:
            await self.drone.offboard.set_position_velocity_ned(
                PositionNedYaw(NED[0], NED[1], NED[2], NED[3]),
                VelocityNedYaw(vel[0], vel[1], vel[2], vel[3])
            )

    async def turn_on_offboard(self) -> None:
        print("-- Starting offboard")
        try:
            await self.drone.offboard.start()
        except OffboardError as error:
            print(f"Starting offboard failed with: {error._result.result}")
            print("-- Disarming")
            await self.drone.action.disarm()

    async def turn_off_offboard(self) -> None:
        print("-- Stopping offboard")
        try:
            await self.drone.offboard.stop()
        except OffboardError as error:
            print(f"Stopping offboard failed with: {error._result.result}")

    async def arm(self) -> None:
        print("-- Arming")
        await self.drone.action.arm()

    async def disarm(self) -> None:
        print("-- Disarming")
        await self.drone.action.disarm()

    async def takeoff_to_altitude(self, altitude: float, yaw: float = 0.0) -> None:
        """
        Offboard takeoff: set a position target then start offboard.
        """
        await self.compute_offset()
        rel_start_point = np.array([0.0, altitude, 0.0, 0.0])  # xyz,yaw
        start_point = self.xyz_to_NED(rel_start_point + self.state_offset[0:4])
        start_point[3] = yaw

        await self.update_state()
        await self.set_new_position(NED=start_point, vel=[0.0, 0.0, -0.2, 0.0])
        print('-- taking off')
        await self.turn_on_offboard()
        await asyncio.sleep(3.0)
        await self.update_state()

    async def simple_land(self) -> None:
        print("-- Landing")
        await self.drone.action.land()

    def set_point_cloud(self, cloud_xyz: Optional[np.ndarray]) -> None:
        self.point_cloud = cloud_xyz

    def plan_path(self) -> None:
        """
        Compute and cache a path using current planner, state, and point cloud.
        """
        assert self.state_offset is not None, "Call begin_mission() first to compute offsets."
        path = self.planner.compute_desired_path(
            state=self.state - self.state_offset,
            point_cloud=self.point_cloud,
            points_min=self.planner.algorithm.min_distance if hasattr(self.planner.algorithm, "min_distance") else 1
        )
        self.path = path

    def path_is_safe(self) -> bool:
        return self.planner.check_goal_safety(goals=self.path, state=self.state[0:3]) if self.path is not None else True


# =========================
# Public Facade: DroneAPI
# =========================

class DroneAPI:
    """
    Minimal, high-level API you can import and call with just a few functions.
    """

    def __init__(
        self,
        system_address: str = "udp://:14540",
        vfh_params: Optional[Dict[str, Any]] = None,
        interpolation: str = "spline",
        spline_points: int = 20,
        goal: Union[List[float], np.ndarray] = (0, 3, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ):
        self.system_address = system_address

        vfh_defaults = dict(
            radius=10.0,
            iterations=1,
            layers=8,
            angle_sections=10,
            distance_tolerance=0.2,
            probability_tolerance=0.05,
            min_obstacle_distance=2.0,
        )
        if vfh_params:
            vfh_defaults.update(vfh_params)

        self.planner = pathPlanner(
            goal_state=_to_np(goal),
            path_planning_algorithm="VFH",
            interpolation_method=interpolation,
            kwargs=vfh_defaults,
            n=spline_points,
        )
        self.env = MagpieEnv(planner=self.planner, goal=_to_np(goal), wait_time=1, position_tracking=False)

    # -------- lifecycle

    async def begin_mission(self, initial_altitude: float = 2.0, yaw: float = 0.0) -> None:
        """
        Connect, wait for healthy estimates, arm, and take off to initial_altitude in offboard.
        """
        await self.env.drone.connect(system_address=self.system_address)

        print("Waiting for connection...")
        async for state in self.env.drone.core.connection_state():
            if state.is_connected:
                print("-- connection successful")
                break

        print("Waiting for global position / home position...")
        async for health in self.env.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print("-- Global position estimate OK")
                break

        # Make sure offboard is stopped (safe)
        await self.env.turn_off_offboard()

        await self.env.arm()
        await self.env.takeoff_to_altitude(altitude=initial_altitude, yaw=yaw)

    async def end_mission(self) -> None:
        """
        Land and stop offboard safely.
        """
        await self.env.simple_land()
        await asyncio.sleep(10)  # let it land fully
        await self.env.turn_off_offboard()

    async def shutdown(self) -> None:
        """
        Emergency-safe: ensure offboard off and disarm.
        """
        await self.env.turn_off_offboard()
        await self.env.disarm()

    # -------- configuration

    def set_point_cloud(self, cloud_xyz: Optional[np.ndarray]) -> None:
        self.env.set_point_cloud(cloud_xyz)
        self.planner.update_point_cloud(cloud_xyz, points_min=10 if cloud_xyz is not None else 1)

    def set_goal(self, goal_xyz12: Union[List[float], np.ndarray]) -> None:
        self.planner.set_goal_state(_to_np(goal_xyz12))
        self.env.goal = _to_np(goal_xyz12)

    def set_vfh_params(self, **kwargs) -> None:
        """
        Update VFH parameters on the fly (radius, layers, min_obstacle_distance, etc.).
        """
        for k, v in kwargs.items():
            if hasattr(self.planner.algorithm, k):
                setattr(self.planner.algorithm, k, v)

    # -------- movement primitives

    async def goto_xyz(self, x: float, y: float, z: float, yaw: float = 0.0, wait_sec: float = 5.0) -> None:
        """
        Move to a single xyz coord (meters, local frame) with yaw (deg).
        """
        assert self.env.state_offset is not None, "Call begin_mission() first."
        await self.env.update_state()
        target_local = np.array([x, y, z, yaw], dtype=float) + self.env.state_offset[0:4]
        ned = self.env.xyz_to_NED(target_local)
        ned[3] = yaw
        await self.env.set_new_position(ned)
        await asyncio.sleep(wait_sec)
        await self.env.update_state()

    async def follow_waypoints(self, waypoints: List[Tuple[float, float, float, float]], dwell_sec: float = 3.0) -> None:
        """
        Waypoints as list of (x, y, z, yaw). Minimal brains: issues setpoint, waits dwell_sec.
        """
        assert self.env.state_offset is not None, "Call begin_mission() first."
        for (x, y, z, yaw) in waypoints:
            await self.goto_xyz(x, y, z, yaw=yaw, wait_sec=dwell_sec)

    def plan_path_to(self, x: float, y: float, z: float) -> np.ndarray:
        """
        VFH plan from current to goal. Returns path (N x 12 state slots; xyz populated).
        """
        assert self.env.state_offset is not None, "Call begin_mission() first."
        goal = np.zeros_like(self.planner.goal)
        goal[:3] = [x, y, z]
        self.set_goal(goal)
        self.env.plan_path()
        return self.env.path if self.env.path is not None else np.empty((0, 12))

    async def goto_with_pathfinding(self, x: float, y: float, z: float, yaw: float = 0.0, dwell_sec: float = 1.0) -> None:
        """
        Plans a VFH path to (x,y,z) and follows it using offboard set-position setpoints.
        """
        path = self.plan_path_to(x, y, z)
        if path.size == 0:
            print("-- planner returned empty path; falling back to direct goto")
            await self.goto_xyz(x, y, z, yaw=yaw, wait_sec=dwell_sec)
            return

        for row in path:
            # row is 12-long state; first 3 are xyz
            local = row[0:4] + self.env.state_offset[0:4]
            ned = self.env.xyz_to_NED(local)
            ned[3] = yaw
            await self.env.set_new_position(ned)
            await asyncio.sleep(dwell_sec)
            await self.env.update_state()

    async def repeat_path(self, waypoints: List[Tuple[float, float, float, float]], times: int = 2, dwell_sec: float = 2.0) -> None:
        for _ in range(int(times)):
            await self.follow_waypoints(waypoints, dwell_sec=dwell_sec)


# =========================
# Demo main
# =========================

async def _demo():
    """
    Quick demo:
      - connect & takeoff to 3m
      - fly a couple waypoints
      - pathfind to (0,3,30)
      - land & shutdown
    """
    drone = DroneAPI(system_address="udp://:14540")

    # optional: load a synthetic point cloud for planning (commented)
    # cloud = np.random.uniform(low=[-5, 0.5, -5], high=[5, 5, 5], size=(1000, 3))
    # drone.set_point_cloud(cloud)

    await drone.begin_mission(initial_altitude=3.0, yaw=0.0)

    # simple moves
    await drone.goto_xyz(0, 3, 5, yaw=0)
    await drone.follow_waypoints([(0, 3, 7, 0), (1, 3, 7, 20), (1, 3, 9, 0)], dwell_sec=2.0)

    # plan & execute VFH path to a distant goal
    await drone.goto_with_pathfinding(0, 3, 30, yaw=0, dwell_sec=0.8)

    await drone.end_mission()
    await drone.shutdown()


def main():
    asyncio.run(_demo())


if __name__ == "__main__":
    # matplotlib is only used in your original demo for plotting; not needed for core API.
    main()