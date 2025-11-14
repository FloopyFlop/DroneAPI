# interpolation.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


# =========================
# Context & Output
# =========================

@dataclass
class InterpContext:
    dt: float
    max_speed: float
    threshold: float
    now_state_pos: np.ndarray           # (3,)
    now_state_vel: np.ndarray           # (3,)
    now_state_yaw: float                # deg
    target_pos: np.ndarray              # (3,)
    target_yaw: float                   # deg
    future_waypoints: List[Tuple[float, float, float, float, float]]  # (x,y,z,yaw,thr)
    use_velocity_command: bool = True


@dataclass
class InterpOutput:
    position_local: Optional[np.ndarray] = None
    yaw_deg: Optional[float] = None
    velocity_xyz_yaw: Optional[np.ndarray] = None
    done: bool = False


class BaseInterpolation:
    def start(self, ctx: InterpContext) -> None:
        self._t = 0.0
        self._last_pos = ctx.now_state_pos.copy()  # for robust completion

    def _update_last(self, ctx: InterpContext) -> None:
        self._last_pos = ctx.now_state_pos.copy()

    def step(self, ctx: InterpContext) -> InterpOutput:  # pragma: no cover
        raise NotImplementedError


# =========================
# Helpers
# =========================

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
    pos_err = float(np.linalg.norm(ctx.target_pos - ctx.now_state_pos))
    if pos_err > ctx.threshold:
        return False
    if float(np.linalg.norm(ctx.now_state_vel)) > speed_tol:
        return False
    return True


def _seg_dir_len(p0: np.ndarray, p1: np.ndarray) -> Tuple[np.ndarray, float]:
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=float), 0.0
    return d / L, L


def _project_param_u(p0: np.ndarray, p1: np.ndarray, p: np.ndarray) -> float:
    d = p1 - p0
    L2 = float(np.dot(d, d))
    if L2 < 1e-12:
        return 1.0
    return float(np.dot(p - p0, d) / L2)


def _point_to_segment_distance(p0: np.ndarray, p1: np.ndarray, p: np.ndarray) -> float:
    u = _project_param_u(p0, p1, p)
    u_clamped = float(np.clip(u, 0.0, 1.0))
    closest = p0 + u_clamped * (p1 - p0)
    return float(np.linalg.norm(p - closest))


def _plane_sign(p: np.ndarray, p_plane: np.ndarray, n_hat: np.ndarray) -> float:
    return float(np.dot(p - p_plane, n_hat))


def _dynamic_threshold(ctx: InterpContext) -> float:
    # speed-aware margin so fast motion doesn’t miss completion
    vmag = float(np.linalg.norm(ctx.now_state_vel))
    return float(ctx.threshold + 0.8 * vmag * ctx.dt)


def _movement_intersects_sphere(a: np.ndarray, b: np.ndarray, center: np.ndarray, radius: float) -> bool:
    """Does the movement segment a->b intersect the sphere around center?"""
    ab = b - a
    ab2 = float(np.dot(ab, ab))
    if ab2 < 1e-12:
        return float(np.linalg.norm(a - center)) <= radius
    t = float(np.clip(np.dot(center - a, ab) / ab2, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(closest - center)) <= radius


def _segment_crossed_endpoint(prev_pos: np.ndarray, now_pos: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> bool:
    d_hat, L = _seg_dir_len(p0, p1)
    if L <= 0.0:
        return True
    s_prev = _plane_sign(prev_pos, p1, d_hat)
    s_now  = _plane_sign(now_pos,  p1, d_hat)
    return (s_prev < 0.0) and (s_now >= 0.0)


def _progress_complete(prev_pos: np.ndarray, now_pos: np.ndarray,
                       p0: np.ndarray, p1: np.ndarray, tol: float) -> bool:
    """
    Robust pass-through completion:

      ✓ distance(now, goal) <= tol
      ✓ param progress u(now) >= 1 - eps and dist(now, segment) <= tol
      ✓ movement segment crosses end plane
      ✓ movement segment intersects goal sphere
    """
    # (A) already inside the sphere around the goal
    if float(np.linalg.norm(now_pos - p1)) <= tol:
        return True

    d_hat, L = _seg_dir_len(p0, p1)
    if L <= 0.0:
        return True

    # (B) param progress with relaxed tolerance
    u_now = _project_param_u(p0, p1, now_pos)
    if (u_now >= 1.0 - 1e-3) and (_point_to_segment_distance(p0, p1, now_pos) <= tol):
        return True

    # (C) crossed the plane through the goal, along segment direction
    if _segment_crossed_endpoint(prev_pos, now_pos, p0, p1):
        return True

    # (D) classic line-through-sphere
    if _movement_intersects_sphere(prev_pos, now_pos, p1, tol):
        return True

    return False

# =========================
# Interpolations
# =========================

class Linear(BaseInterpolation):
    """
    Linear setpoints with velocity hint. Pass-through when another waypoint exists.
    """
    def start(self, ctx: InterpContext) -> None:
        super().start(ctx)
        self._p0 = ctx.now_state_pos.copy()
        self._p1 = ctx.target_pos.copy()
        self._yaw0 = float(ctx.now_state_yaw)
        self._yaw1 = float(ctx.target_yaw)
        _, L = _seg_dir_len(self._p0, self._p1)
        self._has_next = len(ctx.future_waypoints) > 0
        self._duration = max(L / max(ctx.max_speed, 1e-6), ctx.dt)

    def step(self, ctx: InterpContext) -> InterpOutput:
        tol = _dynamic_threshold(ctx)
        if self._has_next:
            if _progress_complete(self._last_pos, ctx.now_state_pos, self._p0, self._p1, tol):
                self._update_last(ctx)
                return InterpOutput(position_local=self._p1, yaw_deg=ctx.target_yaw, done=True)
        else:
            if _stop_done(ctx):
                self._update_last(ctx)
                return InterpOutput(position_local=self._p1, yaw_deg=ctx.target_yaw, done=True)

        self._t += ctx.dt
        a = float(np.clip(self._t / max(self._duration, 1e-6), 0.0, 1.0))
        pos = self._p0 + a * (self._p1 - self._p0)
        yaw = self._yaw0 + a * _shortest_yaw_delta_deg(self._yaw0, self._yaw1)

        vel_cmd = None
        if ctx.use_velocity_command:
            to_goal = self._p1 - ctx.now_state_pos
            desired_v = _clamp(to_goal / max(ctx.dt, 1e-3), ctx.max_speed)
            yaw_rate = _shortest_yaw_delta_deg(ctx.now_state_yaw, self._yaw1) / max(ctx.dt, 1e-3)
            vel_cmd = np.array([desired_v[0], desired_v[1], desired_v[2], yaw_rate], dtype=float)

        self._update_last(ctx)
        return InterpOutput(position_local=pos, yaw_deg=float(yaw), velocity_xyz_yaw=vel_cmd, done=False)


class Cubic(BaseInterpolation):
    """Stop-at-goal with cubic ease-in/out."""
    @staticmethod
    def _cubic(a: float) -> float:
        return 3*a*a - 2*a*a*a

    def start(self, ctx: InterpContext) -> None:
        super().start(ctx)
        self._p0 = ctx.now_state_pos.copy()
        self._p1 = ctx.target_pos.copy()
        self._yaw0 = float(ctx.now_state_yaw)
        self._yaw1 = float(ctx.target_yaw)
        _, L = _seg_dir_len(self._p0, self._p1)
        self._duration = max(L / max(0.75 * ctx.max_speed, 1e-6), 2 * ctx.dt)

    def step(self, ctx: InterpContext) -> InterpOutput:
        if _stop_done(ctx):
            self._update_last(ctx)
            return InterpOutput(position_local=self._p1, yaw_deg=ctx.target_yaw, done=True)

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

        self._update_last(ctx)
        return InterpOutput(position_local=pos, yaw_deg=float(yaw), velocity_xyz_yaw=vel_cmd, done=False)


class TrapezoidalVelocity(BaseInterpolation):
    """Stop-at-goal accel–cruise–decel with cross-track damping."""
    def start(self, ctx: InterpContext) -> None:
        super().start(ctx)
        self._p0 = ctx.now_state_pos.copy()
        self._p1 = ctx.target_pos.copy()
        self._d_hat, self._L = _seg_dir_len(self._p0, self._p1)
        self._a_max = max(0.5 * ctx.max_speed, 0.3)
        self._v_limit = ctx.max_speed
        self._v_end = 0.0
        t_to_vmax = self._v_limit / self._a_max
        d_accel = 0.5 * self._a_max * t_to_vmax**2
        self._v_peak = (2 * self._a_max * self._L)**0.5 if 2*d_accel > self._L else self._v_limit

    def step(self, ctx: InterpContext) -> InterpOutput:
        if _stop_done(ctx, speed_tol=0.05):
            self._update_last(ctx)
            return InterpOutput(position_local=self._p1, yaw_deg=ctx.target_yaw, done=True)

        rel = ctx.now_state_pos - self._p0
        s = float(np.dot(rel, self._d_hat))
        s = np.clip(s, 0.0, self._L)
        remaining = self._L - s
        v_along = float(np.dot(ctx.now_state_vel, self._d_hat))

        d_brake = max((v_along**2 - self._v_end**2) / (2 * self._a_max), 0.0)
        v_target = self._v_peak if remaining > d_brake else self._v_end

        dv = np.clip(v_target - v_along, -self._a_max * ctx.dt, self._a_max * ctx.dt)
        v_next = v_along + dv
        vel_vec = v_next * self._d_hat

        on_line = self._p0 + s * self._d_hat
        cross = ctx.now_state_pos - on_line
        vel_vec += -0.8 * cross

        vel_vec = _clamp(vel_vec, ctx.max_speed)
        yaw_rate = _shortest_yaw_delta_deg(ctx.now_state_yaw, ctx.target_yaw) / max(ctx.dt, 1e-3)

        self._update_last(ctx)
        return InterpOutput(position_local=self._p1,
                            velocity_xyz_yaw=np.array([vel_vec[0], vel_vec[1], vel_vec[2], yaw_rate], dtype=float),
                            done=False)


class Bezier(BaseInterpolation):
    """Quadratic Bézier pass-through with robust completion."""
    def start(self, ctx: InterpContext) -> None:
        super().start(ctx)
        self._p0 = ctx.now_state_pos.copy()
        self._p2 = ctx.target_pos.copy()
        self._has_next = len(ctx.future_waypoints) > 0

        if self._has_next:
            nx, ny, nz, *_ = ctx.future_waypoints[0]
            next_p = np.array([nx, ny, nz], dtype=float)
            d0 = self._p2 - self._p0
            d1 = next_p - self._p2
            c_dir = _clamp(d0, np.inf) + 0.7 * _clamp(d1, np.inf)
            self._p1 = self._p2 - 0.33 * c_dir
        else:
            d = self._p2 - self._p0
            self._p1 = self._p0 + 0.5 * d

        _, L = _seg_dir_len(self._p0, self._p2)
        self._seg_p0 = self._p0
        self._seg_p1 = self._p2
        self._duration = max(L / max(0.8 * ctx.max_speed, 1e-6), 2 * ctx.dt)

    @staticmethod
    def _bz(p0, p1, p2, t):
        u = 1.0 - t
        return u*u*p0 + 2*u*t*p1 + t*t*p2

    @staticmethod
    def _bz_d(p0, p1, p2, t):
        return 2*(1-t)*(p1-p0) + 2*t*(p2-p1)

    def step(self, ctx: InterpContext) -> InterpOutput:
        tol = _dynamic_threshold(ctx)
        if self._has_next:
            if _progress_complete(self._last_pos, ctx.now_state_pos, self._seg_p0, self._seg_p1, tol):
                self._update_last(ctx)
                return InterpOutput(position_local=self._seg_p1, yaw_deg=ctx.target_yaw, done=True)
        else:
            if _stop_done(ctx):
                self._update_last(ctx)
                return InterpOutput(position_local=self._seg_p1, yaw_deg=ctx.target_yaw, done=True)

        self._t += ctx.dt
        a = float(np.clip(self._t / max(self._duration, 1e-6), 0.0, 1.0))
        pos = self._bz(self._p0, self._p1, self._p2, a)

        vel_cmd = None
        if ctx.use_velocity_command:
            deriv = self._bz_d(self._p0, self._p1, self._p2, a)
            deriv = _clamp(deriv, ctx.max_speed)
            to_curve = pos - ctx.now_state_pos
            vel = 0.65 * deriv + 0.35 * _clamp(to_curve, ctx.max_speed)
            vel = _clamp(vel, ctx.max_speed)
            yaw_rate = _shortest_yaw_delta_deg(ctx.now_state_yaw, ctx.target_yaw) / max(ctx.dt, 1e-3)
            vel_cmd = np.array([vel[0], vel[1], vel[2], yaw_rate], dtype=float)

        self._update_last(ctx)
        return InterpOutput(position_local=pos, yaw_deg=ctx.target_yaw, velocity_xyz_yaw=vel_cmd, done=False)


class BarrelTowards(BaseInterpolation):
    """Constant-speed bias with braking and cross-track damping; pass-through completion."""
    def start(self, ctx: InterpContext) -> None:
        super().start(ctx)
        self._p0 = ctx.now_state_pos.copy()
        self._p1 = ctx.target_pos.copy()
        self._has_next = len(ctx.future_waypoints) > 0
        self._d_hat, self._L = _seg_dir_len(self._p0, self._p1)
        self._a_max = max(0.6 * ctx.max_speed, 0.4)
        self._v_cruise = 0.9 * ctx.max_speed
        self._v_end = 0.15 if self._has_next else 0.0

    def step(self, ctx: InterpContext) -> InterpOutput:
        tol = _dynamic_threshold(ctx)
        if self._has_next:
            if _progress_complete(self._last_pos, ctx.now_state_pos, self._p0, self._p1, tol):
                self._update_last(ctx)
                return InterpOutput(position_local=self._p1, yaw_deg=ctx.target_yaw, done=True)
        else:
            if _stop_done(ctx, speed_tol=max(self._v_end, 0.05)):
                self._update_last(ctx)
                return InterpOutput(position_local=self._p1, yaw_deg=ctx.target_yaw, done=True)

        rel = ctx.now_state_pos - self._p0
        s = float(np.dot(rel, self._d_hat))
        s = np.clip(s, 0.0, self._L)
        remaining = self._L - s

        v_along = float(np.dot(ctx.now_state_vel, self._d_hat))
        d_brake = max((v_along**2 - self._v_end**2) / (2 * self._a_max), 0.0)
        v_target = self._v_cruise if remaining > d_brake else self._v_end

        dv = np.clip(v_target - v_along, -self._a_max * ctx.dt, self._a_max * ctx.dt)
        v_next = v_along + dv
        vel_vec = v_next * self._d_hat

        on_line = self._p0 + s * self._d_hat
        cross = ctx.now_state_pos - on_line
        # fade cross-track gain as we get close to end, to avoid orbit
        fade = float(remaining / (remaining + 1.0))
        vel_vec += -0.9 * fade * cross

        vel_vec = _clamp(vel_vec, ctx.max_speed)
        yaw_rate = _shortest_yaw_delta_deg(ctx.now_state_yaw, ctx.target_yaw) / max(ctx.dt, 1e-3)

        self._update_last(ctx)
        return InterpOutput(position_local=self._p1,
                            velocity_xyz_yaw=np.array([vel_vec[0], vel_vec[1], vel_vec[2], yaw_rate], dtype=float),
                            done=False)


# =========================
# NEW: L1/Pure-Pursuit Guidance (tuned endgame)
# =========================

class L1Guidance(BaseInterpolation):
    """
    Pure-pursuit style pass-through with robust completion and endgame tuning.
    """
    def start(self, ctx: InterpContext) -> None:
        super().start(ctx)
        self._p0 = ctx.now_state_pos.copy()
        self._p1 = ctx.target_pos.copy()
        self._has_next = len(ctx.future_waypoints) > 0
        self._d_hat, self._Lseg = _seg_dir_len(self._p0, self._p1)

        # Lookahead & gains
        self._L_min = max(0.6, 0.35 * self._Lseg)
        self._L_gain = 1.3
        self._kx_base = 1.0
        self._v_cruise = 0.9 * ctx.max_speed
        self._v_end = 0.15 if self._has_next else 0.0
        self._a_max = max(0.6 * ctx.max_speed, 0.4)
        self._along_bias = 0.2  # small push to cross the endpoint plane

    def _closest_u(self, p: np.ndarray) -> float:
        return _project_param_u(self._p0, self._p1, p)

    def _point_on_seg(self, u: float) -> np.ndarray:
        u = float(np.clip(u, 0.0, 1.0))
        return self._p0 + u * (self._p1 - self._p0)

    def step(self, ctx: InterpContext) -> InterpOutput:
        tol = _dynamic_threshold(ctx)
        if self._has_next:
            if _progress_complete(self._last_pos, ctx.now_state_pos, self._p0, self._p1, tol):
                self._update_last(ctx)
                return InterpOutput(position_local=self._p1, yaw_deg=ctx.target_yaw, done=True)
        else:
            if _stop_done(ctx, speed_tol=max(self._v_end, 0.05)):
                self._update_last(ctx)
                return InterpOutput(position_local=self._p1, yaw_deg=ctx.target_yaw, done=True)

        # Closest projection on segment
        u = self._closest_u(ctx.now_state_pos)
        u_clamped = float(np.clip(u, 0.0, 1.0))
        closest = self._point_on_seg(u_clamped)
        remaining = (1.0 - u_clamped) * self._Lseg

        # Lookahead shrinks near the end
        vmag = float(np.linalg.norm(ctx.now_state_vel))
        L_look = max(self._L_min, self._L_gain * max(vmag, 0.1))
        L_look = float(min(L_look, max(0.4, 0.8 * remaining)))
        carrot_u = float(np.clip(u_clamped + L_look / max(self._Lseg, 1e-6), 0.0, 1.0))
        carrot = self._point_on_seg(carrot_u)

        # Velocity toward carrot + endgame bias and cross-track fade
        to_carrot = carrot - ctx.now_state_pos
        if np.linalg.norm(to_carrot) < 1e-6:
            dir_vec = self._d_hat
        else:
            dir_vec = to_carrot / (np.linalg.norm(to_carrot) + 1e-9)

        v_along = float(np.dot(ctx.now_state_vel, self._d_hat))
        d_brake = max((v_along**2 - self._v_end**2) / (2 * self._a_max), 0.0)
        v_target = self._v_cruise if remaining > d_brake else self._v_end
        v_vec = v_target * dir_vec

        # cross-track correction fades near the end
        cross = ctx.now_state_pos - closest
        kx = self._kx_base * float(remaining / (remaining + 1.0))
        v_vec += -kx * cross

        # small along-track bias to push through the plane
        v_vec += self._along_bias * self._d_hat * float(remaining < 0.8 * self._Lseg)

        v_vec = _clamp(v_vec, ctx.max_speed)
        yaw_rate = _shortest_yaw_delta_deg(ctx.now_state_yaw, ctx.target_yaw) / max(ctx.dt, 1e-3)

        self._update_last(ctx)
        return InterpOutput(position_local=self._p1,
                            velocity_xyz_yaw=np.array([v_vec[0], v_vec[1], v_vec[2], yaw_rate], dtype=float),
                            done=False)

