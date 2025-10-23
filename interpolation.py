# interpolation.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class InterpContext:
    """
    Inputs to an interpolation strategy, refreshed every control tick.
    All vectors are LOCAL frame (offset removed).
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
    Output per tick.
      - position_local (optional): anchor position setpoint in LOCAL frame.
      - velocity_xyz_yaw (optional): [vx, vy, vz, yaw_rate] in LOCAL frame (deg/s).
      - yaw_deg (optional): absolute yaw to hold if not providing yaw rate.
      - done: True when waypoint satisfied (<= threshold).
    """
    position_local: Optional[np.ndarray] = None
    yaw_deg: Optional[float] = None
    velocity_xyz_yaw: Optional[np.ndarray] = None
    done: bool = False


class BaseInterpolation:
    def start(self, ctx: InterpContext) -> None:
        self._t = 0.0

    def step(self, ctx: InterpContext) -> InterpOutput:  # pragma: no cover
        raise NotImplementedError


# ------------------ Helpers ------------------

def _clamp(v: np.ndarray, vmax: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-9:
        return np.zeros_like(v)
    if n > vmax:
        return v * (vmax / n)
    return v


def _shortest_yaw_delta_deg(a_deg: float, b_deg: float) -> float:
    return float((b_deg - a_deg + 180.0) % 360.0 - 180.0)


def _done(ctx: InterpContext, extra_speed_tol: float = None) -> bool:
    pos_err = float(np.linalg.norm(ctx.target_pos - ctx.now_state_pos))
    if pos_err > ctx.threshold:
        return False
    if extra_speed_tol is not None:
        if float(np.linalg.norm(ctx.now_state_vel)) > extra_speed_tol:
            return False
    return True


# ======================================================================
# Baseline methods you already had (kept): Linear, Cubic
# ======================================================================

class Linear(BaseInterpolation):
    """Linear position interpolation with velocity clamped by max_speed."""
    def start(self, ctx: InterpContext) -> None:
        super().start(ctx)
        dist = float(np.linalg.norm(ctx.target_pos - ctx.now_state_pos))
        self._duration = max(dist / max(ctx.max_speed, 1e-6), ctx.dt)
        self._p0 = ctx.now_state_pos.copy()
        self._p1 = ctx.target_pos.copy()
        self._yaw0 = float(ctx.now_state_yaw)
        self._yaw1 = float(ctx.target_yaw)

    def step(self, ctx: InterpContext) -> InterpOutput:
        if _done(ctx):
            return InterpOutput(position_local=ctx.target_pos, yaw_deg=ctx.target_yaw, done=True)

        self._t += ctx.dt
        a = float(np.clip(self._t / max(self._duration, 1e-6), 0.0, 1.0))
        pos = self._p0 + a * (self._p1 - self._p0)
        yaw = self._yaw0 + a * _shortest_yaw_delta_deg(self._yaw0, self._yaw1)

        vel_cmd = None
        if ctx.use_velocity_command:
            desired_v = (self._p1 - ctx.now_state_pos) / max(ctx.dt, 1e-3)
            desired_v = _clamp(desired_v, ctx.max_speed)
            yaw_rate = _shortest_yaw_delta_deg(ctx.now_state_yaw, self._yaw1) / max(ctx.dt, 1e-3)
            vel_cmd = np.array([desired_v[0], desired_v[1], desired_v[2], yaw_rate], dtype=float)

        return InterpOutput(position_local=pos, yaw_deg=float(yaw), velocity_xyz_yaw=vel_cmd, done=False)


class Cubic(BaseInterpolation):
    """Cubic ease-in/out (time-scaling) along straight line."""
    @staticmethod
    def _cubic(a: float) -> float:
        return 3 * a * a - 2 * a * a * a

    def start(self, ctx: InterpContext) -> None:
        super().start(ctx)
        dist = float(np.linalg.norm(ctx.target_pos - ctx.now_state_pos))
        self._duration = max(dist / max(0.75 * ctx.max_speed, 1e-6), 2 * ctx.dt)
        self._p0 = ctx.now_state_pos.copy()
        self._p1 = ctx.target_pos.copy()
        self._yaw0 = float(ctx.now_state_yaw)
        self._yaw1 = float(ctx.target_yaw)

    def step(self, ctx: InterpContext) -> InterpOutput:
        if _done(ctx):
            return InterpOutput(position_local=ctx.target_pos, yaw_deg=ctx.target_yaw, done=True)

        self._t += ctx.dt
        a = float(np.clip(self._t / max(self._duration, 1e-6), 0.0, 1.0))
        s = self._cubic(a)

        pos = self._p0 + s * (self._p1 - self._p0)
        yaw = self._yaw0 + s * _shortest_yaw_delta_deg(self._yaw0, self._yaw1)

        vel_cmd = None
        if ctx.use_velocity_command:
            # derivative of s via lookahead
            a2 = float(np.clip((self._t + ctx.dt) / max(self._duration, 1e-6), 0.0, 1.0))
            s2 = self._cubic(a2)
            ds = (s2 - s) / max(ctx.dt, 1e-6)
            desired_v = ds * (self._p1 - self._p0)
            desired_v = _clamp(desired_v, ctx.max_speed)
            yaw_rate = _shortest_yaw_delta_deg(ctx.now_state_yaw, self._yaw1) / max(ctx.dt, 1e-3)
            vel_cmd = np.array([desired_v[0], desired_v[1], desired_v[2], yaw_rate], dtype=float)

        return InterpOutput(position_local=pos, yaw_deg=float(yaw), velocity_xyz_yaw=vel_cmd, done=False)


# ======================================================================
# New strategies (5+)
# ======================================================================

class TrapezoidalVelocity(BaseInterpolation):
    """
    Velocity-controlled straight-line motion using a trapezoidal speed profile.
    Parameters (tuned in-code): max_accel, end_speed.
    """
    def start(self, ctx: InterpContext) -> None:
        super().start(ctx)
        self._dir = ctx.target_pos - ctx.now_state_pos
        self._dist = float(np.linalg.norm(self._dir))
        self._dir = self._dir / (self._dist + 1e-9)
        self._max_accel = max(0.5 * ctx.max_speed, 0.3)   # m/s^2
        self._end_speed = 0.05                            # stop near target

        # Peak speed (may be less than ctx.max_speed if short segment)
        t_to_vmax = ctx.max_speed / self._max_accel
        d_accel = 0.5 * self._max_accel * t_to_vmax**2
        if 2 * d_accel > self._dist:
            # triangular profile
            self._v_peak = (2 * self._max_accel * self._dist)**0.5
        else:
            self._v_peak = ctx.max_speed

    def step(self, ctx: InterpContext) -> InterpOutput:
        err_vec = ctx.target_pos - ctx.now_state_pos
        d = float(np.linalg.norm(err_vec))
        if d <= ctx.threshold and np.linalg.norm(ctx.now_state_vel) <= max(self._end_speed, 0.05):
            return InterpOutput(position_local=ctx.target_pos, yaw_deg=ctx.target_yaw, done=True)

        # Decide target speed based on remaining distance (simple braking distance)
        v = float(np.dot(ctx.now_state_vel, self._dir))
        braking_d = max((v**2 - self._end_speed**2) / (2 * self._max_accel), 0.0)
        want_speed = self._v_peak if d > braking_d else max(self._end_speed, 0.0)

        # Accelerate / decelerate toward want_speed along the line
        dv = np.clip(want_speed - v, -self._max_accel * ctx.dt, self._max_accel * ctx.dt)
        v_next = v + dv
        vel_vec = v_next * self._dir
        vel_vec = _clamp(vel_vec, ctx.max_speed)

        yaw_rate = _shortest_yaw_delta_deg(ctx.now_state_yaw, ctx.target_yaw) / max(ctx.dt, 1e-3)
        return InterpOutput(position_local=ctx.target_pos, yaw_deg=None,
                            velocity_xyz_yaw=np.array([vel_vec[0], vel_vec[1], vel_vec[2], yaw_rate], dtype=float),
                            done=False)


class PDVelocity(BaseInterpolation):
    """
    Pure velocity control using PD on position error: v = Kp*e - Kd*v
    Good for tight stop-on-target, fully velocity-driven.
    """
    def start(self, ctx: InterpContext) -> None:
        super().start(ctx)
        self._kp = 1.2         # proportional (1/s)
        self._kd = 0.6         # damping
        self._stop_speed = 0.06

    def step(self, ctx: InterpContext) -> InterpOutput:
        e = ctx.target_pos - ctx.now_state_pos
        if np.linalg.norm(e) <= ctx.threshold and np.linalg.norm(ctx.now_state_vel) <= self._stop_speed:
            return InterpOutput(position_local=ctx.target_pos, yaw_deg=ctx.target_yaw, done=True)

        v = self._kp * e - self._kd * ctx.now_state_vel
        v = _clamp(v, ctx.max_speed)
        yaw_rate = _shortest_yaw_delta_deg(ctx.now_state_yaw, ctx.target_yaw) / max(ctx.dt, 1e-3)
        return InterpOutput(position_local=ctx.target_pos, velocity_xyz_yaw=np.array([v[0], v[1], v[2], yaw_rate], dtype=float))


class Bezier(BaseInterpolation):
    """
    Quadratic Bézier curve from start->control->target.
    Control point is inferred from the heading to the next waypoint if available,
    otherwise uses a mild offset in the start->target direction.
    """
    def start(self, ctx: InterpContext) -> None:
        super().start(ctx)
        self._p0 = ctx.now_state_pos.copy()
        self._p2 = ctx.target_pos.copy()

        if ctx.future_waypoints:
            nx, ny, nz, *_ = ctx.future_waypoints[0]
            next_p = np.array([nx, ny, nz], dtype=float)
            # control aimed between p0->p2 and p2->next segment for smoothing
            d0 = self._p2 - self._p0
            d1 = next_p - self._p2
            c_dir = _clamp(d0, np.inf) + 0.5 * _clamp(d1, np.inf)
            self._p1 = self._p2 - 0.33 * c_dir  # pull control a bit before target
        else:
            d = self._p2 - self._p0
            self._p1 = self._p0 + 0.5 * d

        # estimate duration based on arc length approx
        chord = np.linalg.norm(self._p2 - self._p0)
        self._duration = max(chord / max(0.7 * ctx.max_speed, 1e-6), 2 * ctx.dt)

    @staticmethod
    def _bezier(p0, p1, p2, t):
        u = 1.0 - t
        return u*u*p0 + 2*u*t*p1 + t*t*p2

    @staticmethod
    def _bezier_deriv(p0, p1, p2, t):
        return 2*(1-t)*(p1-p0) + 2*t*(p2-p1)

    def step(self, ctx: InterpContext) -> InterpOutput:
        if _done(ctx):
            return InterpOutput(position_local=ctx.target_pos, yaw_deg=ctx.target_yaw, done=True)

        self._t += ctx.dt
        a = float(np.clip(self._t / max(self._duration, 1e-6), 0.0, 1.0))
        pos = self._bezier(self._p0, self._p1, self._p2, a)

        vel_cmd = None
        if ctx.use_velocity_command:
            deriv = self._bezier_deriv(self._p0, self._p1, self._p2, a)
            deriv = _clamp(deriv, ctx.max_speed)
            yaw_rate = _shortest_yaw_delta_deg(ctx.now_state_yaw, ctx.target_yaw) / max(ctx.dt, 1e-3)
            vel_cmd = np.array([deriv[0], deriv[1], deriv[2], yaw_rate], dtype=float)

        return InterpOutput(position_local=pos, yaw_deg=ctx.target_yaw, velocity_xyz_yaw=vel_cmd, done=False)


class LookAheadBlend(BaseInterpolation):
    """
    Blends direction between current segment and the next to round corners.
    Pure velocity guidance; speed tapers near the target.
    """
    def start(self, ctx: InterpContext) -> None:
        super().start(ctx)
        self._min_speed = 0.1

    def step(self, ctx: InterpContext) -> InterpOutput:
        to_target = ctx.target_pos - ctx.now_state_pos
        d = float(np.linalg.norm(to_target))
        if d <= ctx.threshold:
            return InterpOutput(position_local=ctx.target_pos, yaw_deg=ctx.target_yaw, done=True)

        dir0 = to_target / (d + 1e-9)

        if ctx.future_waypoints:
            nx, ny, nz, *_ = ctx.future_waypoints[0]
            next_vec = np.array([nx, ny, nz], dtype=float) - ctx.target_pos
            if np.linalg.norm(next_vec) > 1e-6:
                dir1 = next_vec / np.linalg.norm(next_vec)
            else:
                dir1 = dir0
            blend = 0.5 * dir0 + 0.5 * dir1  # simple average; could weight by d
            direction = blend / (np.linalg.norm(blend) + 1e-9)
        else:
            direction = dir0

        # Speed schedule: slow down near target
        speed = np.clip(0.2 + 0.8 * (d / (d + 1.0)), self._min_speed, ctx.max_speed)
        v = speed * direction

        yaw_rate = _shortest_yaw_delta_deg(ctx.now_state_yaw, ctx.target_yaw) / max(ctx.dt, 1e-3)
        return InterpOutput(position_local=ctx.target_pos,
                            velocity_xyz_yaw=np.array([v[0], v[1], v[2], yaw_rate], dtype=float),
                            done=False)


class SineEase(BaseInterpolation):
    """
    Sinusoidal ease along straight line: s = 0.5 - 0.5*cos(pi*a)
    Smooth start/stop, mid-speed boosted vs. cubic.
    """
    @staticmethod
    def _sine(a: float) -> float:
        return 0.5 - 0.5*np.cos(np.pi * a)

    def start(self, ctx: InterpContext) -> None:
        super().start(ctx)
        d = float(np.linalg.norm(ctx.target_pos - ctx.now_state_pos))
        self._duration = max(d / max(0.9 * ctx.max_speed, 1e-6), 2 * ctx.dt)
        self._p0 = ctx.now_state_pos.copy()
        self._p1 = ctx.target_pos.copy()
        self._yaw0 = float(ctx.now_state_yaw)
        self._yaw1 = float(ctx.target_yaw)

    def step(self, ctx: InterpContext) -> InterpOutput:
        if _done(ctx):
            return InterpOutput(position_local=ctx.target_pos, yaw_deg=ctx.target_yaw, done=True)

        self._t += ctx.dt
        a = float(np.clip(self._t / max(self._duration, 1e-6), 0.0, 1.0))
        s = self._sine(a)

        pos = self._p0 + s * (self._p1 - self._p0)
        yaw = self._yaw0 + s * _shortest_yaw_delta_deg(self._yaw0, self._yaw1)

        vel_cmd = None
        if ctx.use_velocity_command:
            # central difference on s
            a2 = float(np.clip((self._t + ctx.dt) / max(self._duration, 1e-6), 0.0, 1.0))
            s2 = self._sine(a2)
            ds = (s2 - s) / max(ctx.dt, 1e-6)
            desired_v = ds * (self._p1 - self._p0)
            desired_v = _clamp(desired_v, ctx.max_speed)
            yaw_rate = _shortest_yaw_delta_deg(ctx.now_state_yaw, self._yaw1) / max(ctx.dt, 1e-3)
            vel_cmd = np.array([desired_v[0], desired_v[1], desired_v[2], yaw_rate], dtype=float)

        return InterpOutput(position_local=pos, yaw_deg=float(yaw), velocity_xyz_yaw=vel_cmd, done=False)
