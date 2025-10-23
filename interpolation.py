# interpolation.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class InterpContext:
    """
    Inputs to an interpolation strategy, refreshed every control tick.
    - All positions are in LOCAL frame (offset removed).
    """
    dt: float
    max_speed: float
    threshold: float
    now_state_pos: np.ndarray           # shape (3,)  [x, y, z]
    now_state_vel: np.ndarray           # shape (3,)  [vx, vy, vz]
    now_state_yaw: float                # degrees
    target_pos: np.ndarray              # shape (3,)
    target_yaw: float                   # degrees
    future_waypoints: List[Tuple[float, float, float, float, float]]  # (x,y,z,yaw,threshold)
    use_velocity_command: bool = True


@dataclass
class InterpOutput:
    """
    Output produced by the strategy each tick.
    - Provide either/both a next position (local) and a velocity command [vx, vy, vz, yaw_rate].
    - Set done=True when the waypoint is satisfied (within threshold, etc.).
    """
    position_local: Optional[np.ndarray] = None
    yaw_deg: Optional[float] = None
    velocity_xyz_yaw: Optional[np.ndarray] = None
    done: bool = False


class BaseInterpolation:
    """
    Base class for interpolation strategies.
    Subclasses should override start() (optional) and step() (required).
    """

    def start(self, ctx: InterpContext) -> None:
        """Called once before the first step; can initialize internal state."""
        self._t = 0.0
        self._duration = None

    def step(self, ctx: InterpContext) -> InterpOutput:  # pragma: no cover
        raise NotImplementedError


# ------------------ Utility helpers ------------------

def _clamp_speed(v: np.ndarray, vmax: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-9:
        return np.zeros_like(v)
    if n > vmax:
        return v * (vmax / n)
    return v


def _shortest_yaw_delta_deg(a_deg: float, b_deg: float) -> float:
    """delta to go from a->b in degrees, in [-180, 180]."""
    da = (b_deg - a_deg + 180.0) % 360.0 - 180.0
    return float(da)


def _done_by_threshold(ctx: InterpContext) -> bool:
    pos_err = float(np.linalg.norm(ctx.target_pos - ctx.now_state_pos))
    yaw_err = abs(_shortest_yaw_delta_deg(ctx.now_state_yaw, ctx.target_yaw))
    return (pos_err <= ctx.threshold) and (yaw_err <= 2.0)  # 2° yaw tolerance


# ------------------ Precision (GOTO) ------------------

class Precision(BaseInterpolation):
    """
    Default "GOTO" behavior: hold the target position (optionally with gentle velocity guidance)
    and finish when inside `threshold`.
    """

    def step(self, ctx: InterpContext) -> InterpOutput:
        if _done_by_threshold(ctx):
            return InterpOutput(position_local=ctx.target_pos, yaw_deg=ctx.target_yaw, done=True)

        vel_cmd = None
        if ctx.use_velocity_command:
            to_target = ctx.target_pos - ctx.now_state_pos
            desired_v = _clamp_speed(to_target / max(ctx.dt, 1e-3), ctx.max_speed)
            yaw_rate = _shortest_yaw_delta_deg(ctx.now_state_yaw, ctx.target_yaw) / max(ctx.dt, 1e-3)
            vel_cmd = np.array([desired_v[0], desired_v[1], desired_v[2], yaw_rate], dtype=float)

        return InterpOutput(position_local=ctx.target_pos, yaw_deg=ctx.target_yaw, velocity_xyz_yaw=vel_cmd, done=False)


# ------------------ Linear ------------------

class Linear(BaseInterpolation):
    """
    Linear position interpolation over time, clamped by max_speed.
    """

    def start(self, ctx: InterpContext) -> None:
        super().start(ctx)
        dist = float(np.linalg.norm(ctx.target_pos - ctx.now_state_pos))
        self._duration = max(dist / max(ctx.max_speed, 1e-6), ctx.dt)
        self._start = ctx.now_state_pos.copy()
        self._goal = ctx.target_pos.copy()
        self._start_yaw = float(ctx.now_state_yaw)
        self._goal_yaw = float(ctx.target_yaw)

    def step(self, ctx: InterpContext) -> InterpOutput:
        if _done_by_threshold(ctx):
            return InterpOutput(position_local=ctx.target_pos, yaw_deg=ctx.target_yaw, done=True)

        self._t += ctx.dt
        alpha = float(np.clip(self._t / max(self._duration, 1e-6), 0.0, 1.0))
        pos = self._start + alpha * (self._goal - self._start)
        yaw = self._start_yaw + alpha * _shortest_yaw_delta_deg(self._start_yaw, self._goal_yaw)

        vel_cmd = None
        if ctx.use_velocity_command:
            desired_v = (self._goal - ctx.now_state_pos) / max(ctx.dt, 1e-3)
            desired_v = _clamp_speed(desired_v, ctx.max_speed)
            yaw_rate = _shortest_yaw_delta_deg(ctx.now_state_yaw, self._goal_yaw) / max(ctx.dt, 1e-3)
            vel_cmd = np.array([desired_v[0], desired_v[1], desired_v[2], yaw_rate], dtype=float)

        return InterpOutput(position_local=pos, yaw_deg=float(yaw), velocity_xyz_yaw=vel_cmd, done=False)


# ------------------ Cubic ------------------

class Cubic(BaseInterpolation):
    """
    Cubic ease-in/ease-out time-scaling along the straight line.
    s(a) = 3a^2 - 2a^3, a in [0,1]
    """

    def start(self, ctx: InterpContext) -> None:
        super().start(ctx)
        dist = float(np.linalg.norm(ctx.target_pos - ctx.now_state_pos))
        self._duration = max(dist / max(0.75 * ctx.max_speed, 1e-6), 2 * ctx.dt)  # slightly slower for smoothness
        self._start = ctx.now_state_pos.copy()
        self._goal = ctx.target_pos.copy()
        self._start_yaw = float(ctx.now_state_yaw)
        self._goal_yaw = float(ctx.target_yaw)

    @staticmethod
    def _cubic(a: float) -> float:
        return 3 * a * a - 2 * a * a * a

    def step(self, ctx: InterpContext) -> InterpOutput:
        if _done_by_threshold(ctx):
            return InterpOutput(position_local=ctx.target_pos, yaw_deg=ctx.target_yaw, done=True)

        self._t += ctx.dt
        a = float(np.clip(self._t / max(self._duration, 1e-6), 0.0, 1.0))
        s = self._cubic(a)

        pos = self._start + s * (self._goal - self._start)
        yaw = self._start_yaw + s * _shortest_yaw_delta_deg(self._start_yaw, self._goal_yaw)

        vel_cmd = None
        if ctx.use_velocity_command:
            # numerical derivative of s for a velocity hint
            a_next = float(np.clip((self._t + ctx.dt) / max(self._duration, 1e-6), 0.0, 1.0))
            s_next = self._cubic(a_next)
            ds = (s_next - s) / max(ctx.dt, 1e-6)
            desired_v = ds * (self._goal - self._start)
            desired_v = _clamp_speed(desired_v, ctx.max_speed)
            yaw_rate = _shortest_yaw_delta_deg(ctx.now_state_yaw, self._goal_yaw) / max(ctx.dt, 1e-3)
            vel_cmd = np.array([desired_v[0], desired_v[1], desired_v[2], yaw_rate], dtype=float)

        return InterpOutput(position_local=pos, yaw_deg=float(yaw), velocity_xyz_yaw=vel_cmd, done=False)


# ------------------ MinJerk ------------------

class MinJerk(BaseInterpolation):
    """
    Minimum-jerk (quintic) time-scaling along the straight line.
    s(a) = 10a^3 - 15a^4 + 6a^5
    """

    def start(self, ctx: InterpContext) -> None:
        super().start(ctx)
        dist = float(np.linalg.norm(ctx.target_pos - ctx.now_state_pos))
        self._duration = max(dist / max(0.7 * ctx.max_speed, 1e-6), 2 * ctx.dt)  # smoothest; a bit slower
        self._start = ctx.now_state_pos.copy()
        self._goal = ctx.target_pos.copy()
        self._start_yaw = float(ctx.now_state_yaw)
        self._goal_yaw = float(ctx.target_yaw)

    @staticmethod
    def _minjerk(a: float) -> float:
        return 10 * a**3 - 15 * a**4 + 6 * a**5

    def step(self, ctx: InterpContext) -> InterpOutput:
        if _done_by_threshold(ctx):
            return InterpOutput(position_local=ctx.target_pos, yaw_deg=ctx.target_yaw, done=True)

        self._t += ctx.dt
        a = float(np.clip(self._t / max(self._duration, 1e-6), 0.0, 1.0))
        s = self._minjerk(a)

        pos = self._start + s * (self._goal - self._start)
        yaw = self._start_yaw + s * _shortest_yaw_delta_deg(self._start_yaw, self._goal_yaw)

        vel_cmd = None
        if ctx.use_velocity_command:
            a_next = float(np.clip((self._t + ctx.dt) / max(self._duration, 1e-6), 0.0, 1.0))
            s_next = self._minjerk(a_next)
            ds = (s_next - s) / max(ctx.dt, 1e-6)
            desired_v = ds * (self._goal - self._start)
            desired_v = _clamp_speed(desired_v, ctx.max_speed)
            yaw_rate = _shortest_yaw_delta_deg(ctx.now_state_yaw, self._goal_yaw) / max(ctx.dt, 1e-3)
            vel_cmd = np.array([desired_v[0], desired_v[1], desired_v[2], yaw_rate], dtype=float)

        return InterpOutput(position_local=pos, yaw_deg=float(yaw), velocity_xyz_yaw=vel_cmd, done=False)
