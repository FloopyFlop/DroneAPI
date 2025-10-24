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
      - done: True when waypoint satisfied.
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


def _stop_done(ctx: InterpContext, speed_tol: float = 0.07) -> bool:
    """For stop-at-waypoint strategies: inside threshold and slow enough."""
    pos_err = float(np.linalg.norm(ctx.target_pos - ctx.now_state_pos))
    if pos_err > ctx.threshold:
        return False
    if float(np.linalg.norm(ctx.now_state_vel)) > speed_tol:
        return False
    return True


def _make_gate(p0: np.ndarray, p1: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Plane gate at the target p1, orthogonal to segment direction d.
    Returns (dir d, p1, length).
    """
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return np.array([0.0, 1.0, 0.0], dtype=float), p1.copy(), 0.0  # arbitrary dir
    return d / L, p1.copy(), L


def _passed_gate(now_pos: np.ndarray, d_hat: np.ndarray, p1: np.ndarray, L: float, lateral_tol: float) -> bool:
    """
    Pass-through completion:
      1) Crossed the plane at target: dot(now - p1, d_hat) >= 0
      2) Not wildly off-path near crossing: lateral error bounded.
    """
    rel = now_pos - p1
    along = float(np.dot(rel, d_hat))
    if along < 0:
        return False
    # lateral error check
    lateral_vec = rel - along * d_hat
    lat = float(np.linalg.norm(lateral_vec))
    # allow larger lateral when L is big; always >= threshold scale
    max_lat = max(lateral_tol, 0.05 * max(L, 1.0))
    return lat <= max_lat


# ======================================================================
# 1) Linear (modified to allow pass-through if there's a next waypoint)
# ======================================================================

class Linear(BaseInterpolation):
    """Linear position interpolation with optional velocity guidance and pass-through."""
    def start(self, ctx: InterpContext) -> None:
        super().start(ctx)
        self._p0 = ctx.now_state_pos.copy()
        self._p1 = ctx.target_pos.copy()
        self._yaw0 = float(ctx.now_state_yaw)
        self._yaw1 = float(ctx.target_yaw)

        d = float(np.linalg.norm(self._p1 - self._p0))
        self._duration = max(d / max(ctx.max_speed, 1e-6), ctx.dt)

        # Build a pass-through gate only if there is a next waypoint
        self._has_next = len(ctx.future_waypoints) > 0
        self._d_hat, self._gate_p, self._seg_len = _make_gate(self._p0, self._p1)

    def step(self, ctx: InterpContext) -> InterpOutput:
        # Finish criterion
        if self._has_next:
            if _passed_gate(ctx.now_state_pos, self._d_hat, self._gate_p, self._seg_len, ctx.threshold):
                return InterpOutput(position_local=ctx.target_pos, yaw_deg=ctx.target_yaw, done=True)
        else:
            if _stop_done(ctx):  # last point -> stop there
                return InterpOutput(position_local=ctx.target_pos, yaw_deg=ctx.target_yaw, done=True)

        # Parametric linear motion for setpoint + velocity hint
        self._t += ctx.dt
        a = float(np.clip(self._t / max(self._duration, 1e-6), 0.0, 1.0))
        pos = self._p0 + a * (self._p1 - self._p0)
        yaw = self._yaw0 + a * _shortest_yaw_delta_deg(self._yaw0, self._yaw1)

        vel_cmd = None
        if ctx.use_velocity_command:
            # PD-ish along the remaining vector to help tracking without stopping
            to_goal = self._p1 - ctx.now_state_pos
            desired_v = _clamp(to_goal / max(ctx.dt, 1e-3), ctx.max_speed)
            yaw_rate = _shortest_yaw_delta_deg(ctx.now_state_yaw, self._yaw1) / max(ctx.dt, 1e-3)
            vel_cmd = np.array([desired_v[0], desired_v[1], desired_v[2], yaw_rate], dtype=float)

        return InterpOutput(position_local=pos, yaw_deg=float(yaw), velocity_xyz_yaw=vel_cmd, done=False)


# ======================================================================
# 2) Cubic (unchanged semantics: stop with smooth ease-in/out)
# ======================================================================

class Cubic(BaseInterpolation):
    """Cubic ease-in/out (time-scaling) along straight line; **stops** at waypoint."""
    @staticmethod
    def _cubic(a: float) -> float:
        return 3 * a * a - 2 * a * a * a

    def start(self, ctx: InterpContext) -> None:
        super().start(ctx)
        self._p0 = ctx.now_state_pos.copy()
        self._p1 = ctx.target_pos.copy()
        self._yaw0 = float(ctx.now_state_yaw)
        self._yaw1 = float(ctx.target_yaw)
        dist = float(np.linalg.norm(self._p1 - self._p0))
        self._duration = max(dist / max(0.75 * ctx.max_speed, 1e-6), 2 * ctx.dt)

    def step(self, ctx: InterpContext) -> InterpOutput:
        if _stop_done(ctx):  # stop-type completion
            return InterpOutput(position_local=ctx.target_pos, yaw_deg=ctx.target_yaw, done=True)

        self._t += ctx.dt
        a = float(np.clip(self._t / max(self._duration, 1e-6), 0.0, 1.0))
        s = self._cubic(a)

        pos = self._p0 + s * (self._p1 - self._p0)
        yaw = self._yaw0 + s * _shortest_yaw_delta_deg(self._yaw0, self._yaw1)

        vel_cmd = None
        if ctx.use_velocity_command:
            a2 = float(np.clip((self._t + ctx.dt) / max(self._duration, 1e-6), 0.0, 1.0))
            s2 = self._cubic(a2)
            ds = (s2 - s) / max(ctx.dt, 1e-6)
            desired_v = _clamp(ds * (self._p1 - self._p0), ctx.max_speed)
            yaw_rate = _shortest_yaw_delta_deg(ctx.now_state_yaw, self._yaw1) / max(ctx.dt, 1e-3)
            vel_cmd = np.array([desired_v[0], desired_v[1], desired_v[2], yaw_rate], dtype=float)

        return InterpOutput(position_local=pos, yaw_deg=float(yaw), velocity_xyz_yaw=vel_cmd, done=False)


# ======================================================================
# 3) Trapezoidal Velocity (STOP at each waypoint)
# ======================================================================

class TrapezoidalVelocity(BaseInterpolation):
    """
    Velocity-controlled straight-line motion using a trapezoidal speed profile.
    This strategy **stops** at each waypoint (linear accel, cruise, linear decel).
    """
    def start(self, ctx: InterpContext) -> None:
        super().start(ctx)
        self._p0 = ctx.now_state_pos.copy()
        self._p1 = ctx.target_pos.copy()
        self._d_hat, _, self._seg_len = _make_gate(self._p0, self._p1)

        # Tunables
        self._a_max = max(0.5 * ctx.max_speed, 0.3)   # m/s^2
        self._v_limit = ctx.max_speed
        self._v_end = 0.0

        # Pre-compute peak speed for the segment (triangular if too short)
        t_to_vmax = self._v_limit / self._a_max
        d_accel = 0.5 * self._a_max * t_to_vmax**2
        if 2 * d_accel > self._seg_len:
            self._v_peak = (2 * self._a_max * self._seg_len)**0.5
        else:
            self._v_peak = self._v_limit

    def step(self, ctx: InterpContext) -> InterpOutput:
        if _stop_done(ctx, speed_tol=0.05):
            return InterpOutput(position_local=ctx.target_pos, yaw_deg=ctx.target_yaw, done=True)

        # Signed progress along the line
        rel = ctx.now_state_pos - self._p0
        s = float(np.dot(rel, self._d_hat))               # along-track
        s = np.clip(s, 0.0, self._seg_len)
        remaining = self._seg_len - s

        # Current along-track speed (projection)
        v_along = float(np.dot(ctx.now_state_vel, self._d_hat))

        # Braking distance to v_end
        d_brake = max((v_along**2 - self._v_end**2) / (2 * self._a_max), 0.0)

        # Speed target
        v_target = self._v_peak if remaining > d_brake else self._v_end

        # First-order accel toward v_target
        dv = np.clip(v_target - v_along, -self._a_max * ctx.dt, self._a_max * ctx.dt)
        v_next = v_along + dv
        vel_vec = v_next * self._d_hat

        # Add small cross-track correction toward the straight line
        cross = ctx.now_state_pos - (self._p0 + s * self._d_hat)
        vel_vec += -0.8 * cross  # gentle centering

        vel_vec = _clamp(vel_vec, ctx.max_speed)
        yaw_rate = _shortest_yaw_delta_deg(ctx.now_state_yaw, ctx.target_yaw) / max(ctx.dt, 1e-3)
        return InterpOutput(position_local=ctx.target_pos,
                            velocity_xyz_yaw=np.array([vel_vec[0], vel_vec[1], vel_vec[2], yaw_rate], dtype=float),
                            done=False)


# ======================================================================
# 4) Bézier Curve (pass-through by default)
# ======================================================================

class Bezier(BaseInterpolation):
    """
    Quadratic Bézier curve from start->control->target.
    Control point inferred from direction to next waypoint for smooth turns.
    Pass-through completion via gate plane; will not stop unless last segment.
    """
    def start(self, ctx: InterpContext) -> None:
        super().start(ctx)
        self._p0 = ctx.now_state_pos.copy()
        self._p2 = ctx.target_pos.copy()
        self._has_next = len(ctx.future_waypoints) > 0

        if ctx.future_waypoints:
            nx, ny, nz, *_ = ctx.future_waypoints[0]
            next_p = np.array([nx, ny, nz], dtype=float)
            d0 = self._p2 - self._p0
            d1 = next_p - self._p2
            # “pull” control toward the downstream direction
            c_dir = _clamp(d0, np.inf) + 0.7 * _clamp(d1, np.inf)
            self._p1 = self._p2 - 0.33 * c_dir
        else:
            d = self._p2 - self._p0
            self._p1 = self._p0 + 0.5 * d

        self._d_hat, self._gate_p, self._seg_len = _make_gate(self._p0, self._p2)

        chord = np.linalg.norm(self._p2 - self._p0)
        self._duration = max(chord / max(0.8 * ctx.max_speed, 1e-6), 2 * ctx.dt)

    @staticmethod
    def _bezier(p0, p1, p2, t):
        u = 1.0 - t
        return u*u*p0 + 2*u*t*p1 + t*t*p2

    @staticmethod
    def _bezier_deriv(p0, p1, p2, t):
        return 2*(1-t)*(p1-p0) + 2*t*(p2-p1)

    def step(self, ctx: InterpContext) -> InterpOutput:
        # Completion
        if self._has_next:
            if _passed_gate(ctx.now_state_pos, self._d_hat, self._gate_p, self._seg_len, ctx.threshold):
                return InterpOutput(position_local=ctx.target_pos, yaw_deg=ctx.target_yaw, done=True)
        else:
            if _stop_done(ctx):
                return InterpOutput(position_local=ctx.target_pos, yaw_deg=ctx.target_yaw, done=True)

        # March along the curve at roughly constant parametric speed
        self._t += ctx.dt
        a = float(np.clip(self._t / max(self._duration, 1e-6), 0.0, 1.0))
        pos = self._bezier(self._p0, self._p1, self._p2, a)

        vel_cmd = None
        if ctx.use_velocity_command:
            deriv = self._bezier_deriv(self._p0, self._p1, self._p2, a)
            # Normalize to speed budget
            deriv = _clamp(deriv, ctx.max_speed)
            # Small attraction to curve to avoid drift
            to_curve = pos - ctx.now_state_pos
            vel = 0.65 * deriv + 0.35 * _clamp(to_curve, ctx.max_speed)
            vel = _clamp(vel, ctx.max_speed)
            yaw_rate = _shortest_yaw_delta_deg(ctx.now_state_yaw, ctx.target_yaw) / max(ctx.dt, 1e-3)
            vel_cmd = np.array([vel[0], vel[1], vel[2], yaw_rate], dtype=float)

        return InterpOutput(position_local=pos, yaw_deg=ctx.target_yaw, velocity_xyz_yaw=vel_cmd, done=False)


# ======================================================================
# 5) BarrelTowards (pass-through, maintains near-constant speed)
# ======================================================================

class BarrelTowards(BaseInterpolation):
    """
    Maintain near-constant speed heading toward target; slow only in a braking window.
    Uses gate-plane completion so it won't orbit. Good for “keep moving” feel.
    """
    def start(self, ctx: InterpContext) -> None:
        super().start(ctx)
        self._p0 = ctx.now_state_pos.copy()
        self._p1 = ctx.target_pos.copy()
        self._has_next = len(ctx.future_waypoints) > 0
        self._d_hat, self._gate_p, self._seg_len = _make_gate(self._p0, self._p1)

        # Tunables
        self._a_max = max(0.6 * ctx.max_speed, 0.4)
        self._v_cruise = 0.9 * ctx.max_speed
        self._v_end = 0.15 if self._has_next else 0.0  # if next segment, don’t fully stop

    def step(self, ctx: InterpContext) -> InterpOutput:
        # Completion
        if self._has_next:
            if _passed_gate(ctx.now_state_pos, self._d_hat, self._gate_p, self._seg_len, ctx.threshold):
                return InterpOutput(position_local=ctx.target_pos, yaw_deg=ctx.target_yaw, done=True)
        else:
            if _stop_done(ctx, speed_tol=max(self._v_end, 0.05)):
                return InterpOutput(position_local=ctx.target_pos, yaw_deg=ctx.target_yaw, done=True)

        # Along-track values
        rel0 = ctx.now_state_pos - self._p0
        s = float(np.dot(rel0, self._d_hat))
        s = np.clip(s, 0.0, self._seg_len)
        remaining = self._seg_len - s

        v_along = float(np.dot(ctx.now_state_vel, self._d_hat))

        # Decide speed via braking distance
        d_brake = max((v_along**2 - self._v_end**2) / (2 * self._a_max), 0.0)
        v_target = self._v_cruise if remaining > d_brake else self._v_end

        dv = np.clip(v_target - v_along, -self._a_max * ctx.dt, self._a_max * ctx.dt)
        v_next = v_along + dv
        vel_vec = v_next * self._d_hat

        # Cross-track correction (proportional) so we don't “orbit”
        cross = ctx.now_state_pos - (self._p0 + s * self._d_hat)
        vel_vec += -0.9 * cross

        vel_vec = _clamp(vel_vec, ctx.max_speed)
        yaw_rate = _shortest_yaw_delta_deg(ctx.now_state_yaw, ctx.target_yaw) / max(ctx.dt, 1e-3)
        return InterpOutput(position_local=self._p1,  # keep setpoint ahead
                            velocity_xyz_yaw=np.array([vel_vec[0], vel_vec[1], vel_vec[2], yaw_rate], dtype=float),
                            done=False)
