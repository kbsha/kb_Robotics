#!/usr/bin/env python3
"""
============================================================
 AGENTIC INTELLIGENT MPC — JETRACER (SINGLE-FILE V1)
============================================================

One self-contained, runnable Python file implementing the full
architecture:

    Calibration
        -> 2D simulated world (road + obstacles + goal)
        -> Perception            (lane, obstacles, vehicle state)
        -> Local optimal planner (candidate trajectories, collision
                                   check, lowest-cost selection)
        -> Intelligent MPC       (sampling-horizon steering search
                                   + adaptive throttle)
        -> Safety layer          (emergency stop conditions)
        -> Vehicle actuation     (simulated JetRacer, or real
                                   JetRacer if MODE = "hardware")

DESIGN RULE (kept from the planning discussion):
    The planner and MPC never know whether `state` came from the
    simulator or from a real camera. Both paths funnel through the
    SAME get_state() -> plan() -> control() -> apply() pipeline.

MODE:
    "simulation" -> runs entirely on this computer (default, and
                    the only mode this script actually exercises,
                    since no physical JetRacer/camera is attached
                    here).
    "hardware"   -> same pipeline, but get_state() reads a real
                    camera + the local_planner/MPC output is sent
                    to a real JetRacer. The hooks are implemented
                    but will raise a clear error if the hardware
                    libraries (jetcam, jetracer, torch2trt, ...)
                    are not present, exactly like SAFE_MODE stops
                    would in the original script.

Run:
    python3 agentic_impc_jetracer.py

Output:
    - Live dashboard printed to console every step
    - agentic_impc_run.gif   (animated run, saved automatically)
    - agentic_impc_summary.png (final trajectory + telemetry)
============================================================
"""

import math
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe; swap to default backend for a live window
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation


# ============================================================
# 0. MODE / TOP-LEVEL SWITCH
# ============================================================

MODE = "simulation"        # "simulation" or "hardware"
SAFE_MODE = True            # hardware actuation is gated by this, same as V1 script


# ============================================================
# 1. CALIBRATION
# ============================================================

@dataclass
class VehicleCalibration:
    """Manual vehicle calibration (Section 1 of the architecture)."""

    length: float = 0.30          # m, bumper to bumper
    width: float = 0.18           # m
    wheelbase: float = 0.20       # m, L in the bicycle model
    camera_offset_x: float = 0.05  # m, forward of vehicle center
    camera_offset_y: float = 0.0
    camera_height: float = 0.08   # m
    camera_fov_deg: float = 90.0

    # Initial pose (world frame)
    init_x: float = 0.0
    init_y: float = 0.0
    init_yaw: float = 0.0         # rad
    init_v: float = 0.0           # m/s

    # Actuator limits
    max_steering_rad: float = 0.5     # ~28 deg, physical steering angle
    max_accel: float = 0.9            # m/s^2
    max_decel: float = -2.5           # m/s^2 (braking)
    max_speed: float = 1.2            # m/s
    min_speed: float = 0.0


@dataclass
class WorldCalibration:
    """2D simulated environment configuration (Section 2)."""

    road_length: float = 20.0
    road_half_width: float = 1.0     # m, lane edge to lane edge / 2
    dt: float = 0.05                 # s, control loop period (20 Hz)
    total_time: float = 16.0         # s
    target_speed: float = 0.75       # m/s, nominal cruise speed
    goal_x: float = 19.0

    obstacles: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        # (x, y, radius) in world frame
        (6.0, 0.35, 0.20),
        (11.5, -0.35, 0.20),
        (16.0, 0.30, 0.20),
    ])


VEH = VehicleCalibration()
WORLD = WorldCalibration()


# ============================================================
# 2. VEHICLE MODEL  (shared by simulator AND MPC rollout)
# ============================================================

def bicycle_step(x: float, y: float, yaw: float, v: float,
                  steering: float, accel: float, dt: float,
                  wheelbase: float) -> Tuple[float, float, float, float]:
    """Kinematic bicycle model, rear-axle reference point."""
    steering = float(np.clip(steering, -VEH.max_steering_rad, VEH.max_steering_rad))
    x_next = x + v * math.cos(yaw) * dt
    y_next = y + v * math.sin(yaw) * dt
    yaw_next = yaw + (v / wheelbase) * math.tan(steering) * dt
    v_next = float(np.clip(v + accel * dt, VEH.min_speed, VEH.max_speed))
    return x_next, y_next, yaw_next, v_next


# ============================================================
# 3. SIMULATED WORLD  (stands in for camera + physical car)
# ============================================================

class SimulatedWorld:
    """Simulation-mode 'ground truth' the simulated camera perceives."""

    def __init__(self, veh: VehicleCalibration, world: WorldCalibration):
        self.veh = veh
        self.world = world
        self.x = veh.init_x
        self.y = veh.init_y
        self.yaw = veh.init_yaw
        self.v = veh.init_v
        self.obstacles = list(world.obstacles)

    def apply_control(self, steering: float, accel: float, dt: float):
        self.x, self.y, self.yaw, self.v = bicycle_step(
            self.x, self.y, self.yaw, self.v, steering, accel, dt, self.veh.wheelbase
        )

    def ground_truth_pose(self):
        return self.x, self.y, self.yaw, self.v


# ============================================================
# 4. PERCEPTION  (camera-style: lane detection, obstacle detection,
#                 vehicle-state estimation)
# ============================================================

@dataclass
class VehicleState:
    x: float
    y: float
    yaw: float
    v: float
    lane_error: float        # lateral error vs. lane center (m)
    heading_error: float     # rad
    obstacles: List[Tuple[float, float, float]]  # (x, y, radius) in world frame


def lane_center_y(x: float) -> float:
    """Lane centerline. Straight road at y=0 (swap for a curve if desired)."""
    return 0.0


def simulated_perception(world: SimulatedWorld) -> VehicleState:
    """
    Stand-in for: USB camera -> lane detection -> YOLO obstacle
    detection -> vehicle localization, all fused into one state.

    In simulation mode this reads ground truth directly (as the
    planning discussion recommended: the simulator should produce
    exactly the same STATE STRUCTURE a real state estimator would,
    not something planner/MPC-specific).
    """
    x, y, yaw, v = world.ground_truth_pose()
    target_y = lane_center_y(x)
    lane_error = y - target_y
    heading_error = yaw  # road is straight along x, so yaw IS the heading error
    return VehicleState(
        x=x, y=y, yaw=yaw, v=v,
        lane_error=lane_error,
        heading_error=heading_error,
        obstacles=list(world.obstacles),
    )


def real_camera_perception() -> VehicleState:
    """
    Hardware-mode perception hook. Reads a real camera frame, runs
    lane detection + YOLO + vehicle localization, and returns a
    VehicleState with the SAME fields as simulated_perception().

    Not runnable in this environment (no camera / jetcam / YOLO
    weights attached), so it fails loudly instead of silently
    doing nothing — same philosophy as the emergency safety layer.
    """
    raise RuntimeError(
        "Hardware perception requested but no camera/jetcam/YOLO stack "
        "is available in this environment. Wire this function to your "
        "existing detect_lane()/detect_objects()/state estimator, "
        "keeping the same VehicleState fields, then set MODE='hardware'."
    )


def get_state(world: Optional[SimulatedWorld]) -> VehicleState:
    """Single entry point the planner/MPC call. Mode-agnostic."""
    if MODE == "simulation":
        assert world is not None
        return simulated_perception(world)
    elif MODE == "hardware":
        return real_camera_perception()
    else:
        raise ValueError(f"Unknown MODE: {MODE}")


# ============================================================
# 5. LOCAL OPTIMAL PATH PLANNER
# ============================================================

@dataclass
class PlannerResult:
    feasible: bool
    selected_offset: float                 # target lateral offset (m)
    label: str                              # e.g. "CENTER", "LEFT-PASS", "RIGHT-PASS"
    reference_path: List[Tuple[float, float]]  # (x, y) points ahead
    num_candidates: int
    closest_obstacle_dist: float


LOOKAHEAD = 3.0            # m, planning horizon distance
PATH_SAMPLES = 12
CANDIDATE_OFFSETS = np.linspace(-0.75, 0.75, 11)  # m, relative to lane center
OBSTACLE_SAFETY_MARGIN = 0.12  # m, added to obstacle radius + half vehicle width


def _distance_to_nearest_obstacle_ahead(x0: float,
                                         obstacles: List[Tuple[float, float, float]]) -> float:
    ahead = [ox - orad - x0 for (ox, oy, orad) in obstacles if ox + orad > x0]
    return min(ahead) if ahead else float("inf")


def _generate_candidate_path(x0: float, y0: float, yaw0: float,
                              target_offset: float,
                              obstacles: List[Tuple[float, float, float]]
                              ) -> List[Tuple[float, float]]:
    """
    Smooth blend from current pose to a target lateral offset.

    IMPORTANT: the blend must complete BEFORE reaching a nearby obstacle,
    not at the far end of the fixed lookahead — otherwise the candidate
    is still mid-swerve exactly where it needs to already be clear. So
    the transition distance is capped to (distance to nearest obstacle
    ahead - safety buffer) when an obstacle is inside the lookahead;
    beyond the transition, the path holds the target offset steady.
    """
    dist_ahead = _distance_to_nearest_obstacle_ahead(x0, obstacles)
    if dist_ahead < LOOKAHEAD * 1.5:
        transition = float(np.clip(dist_ahead - 0.35, 0.6, LOOKAHEAD))
    else:
        transition = LOOKAHEAD

    target_y = lane_center_y(x0 + LOOKAHEAD) + target_offset
    pts = []
    for i in range(1, PATH_SAMPLES + 1):
        x = x0 + LOOKAHEAD * (i / PATH_SAMPLES)
        d = x - x0
        s = float(np.clip(d / transition, 0.0, 1.0))
        blend = 3 * s ** 2 - 2 * s ** 3   # smoothstep, zero slope at both ends
        y = y0 + (target_y - y0) * blend
        pts.append((x, y))
    return pts


def _path_collides(path: List[Tuple[float, float]],
                    obstacles: List[Tuple[float, float, float]]) -> bool:
    half_vehicle = VEH.width / 2.0
    for (px, py) in path:
        for (ox, oy, orad) in obstacles:
            d = math.hypot(px - ox, py - oy)
            if d < orad + half_vehicle + OBSTACLE_SAFETY_MARGIN:
                return True
    return False


def _obstacle_clearance(x: float, y: float,
                         obstacles: List[Tuple[float, float, float]]) -> float:
    """Real clearance from the vehicle's ACTUAL current position to the
    nearest obstacle surface — used for reporting and the safety layer.
    (Deliberately independent of which candidate paths were rejected:
    a candidate swerving close to an obstacle and getting rejected must
    not by itself make the safety layer think the vehicle is in danger.)"""
    half_vehicle = VEH.width / 2.0
    if not obstacles:
        return float("inf")
    return min(math.hypot(x - ox, y - oy) - orad - half_vehicle
               for (ox, oy, orad) in obstacles)


def local_planner(state: VehicleState, prev_offset: float,
                   world_cal: WorldCalibration) -> PlannerResult:
    """
    Generates candidate trajectories, rejects colliding ones, and
    selects the lowest-cost safe trajectory (Section 4 of the
    architecture).
    """
    best = None
    best_cost = float("inf")
    closest_obs = _obstacle_clearance(state.x, state.y, state.obstacles)
    feasible_found = False
    max_offset = world_cal.road_half_width - VEH.width / 2.0 - 0.05

    for offset in CANDIDATE_OFFSETS:
        offset = float(np.clip(offset, -max_offset, max_offset))
        path = _generate_candidate_path(state.x, state.y, state.yaw, offset, state.obstacles)
        collision = _path_collides(path, state.obstacles)

        if collision:
            continue

        feasible_found = True
        w_lane, w_change = 1.0, 0.6
        cost = w_lane * offset ** 2 + w_change * (offset - prev_offset) ** 2

        if cost < best_cost:
            best_cost = cost
            best = (offset, path)

    if not feasible_found or best is None:
        # No collision-free candidate: planner reports infeasible so the
        # safety layer can force an emergency stop.
        fallback_path = _generate_candidate_path(state.x, state.y, state.yaw, prev_offset, state.obstacles)
        return PlannerResult(
            feasible=False, selected_offset=prev_offset, label="NO_SAFE_PATH",
            reference_path=fallback_path, num_candidates=len(CANDIDATE_OFFSETS),
            closest_obstacle_dist=closest_obs,
        )

    offset, path = best
    if offset < -0.15:
        label = "RIGHT-PASS"
    elif offset > 0.15:
        label = "LEFT-PASS"
    else:
        label = "CENTER"

    return PlannerResult(
        feasible=True, selected_offset=offset, label=label,
        reference_path=path, num_candidates=len(CANDIDATE_OFFSETS),
        closest_obstacle_dist=closest_obs,
    )


# ============================================================
# 6. INTELLIGENT MPC CONTROLLER
# ============================================================

@dataclass
class ControlResult:
    steering: float          # rad
    throttle_accel: float    # m/s^2 command
    steering_candidates_evaluated: int
    horizon: int
    cost: float


MPC_HORIZON = 8
MPC_DT = 0.1
STEERING_CANDIDATES = np.linspace(-VEH.max_steering_rad, VEH.max_steering_rad, 15)

W_TRACK = 6.0
W_HEADING = 1.5
W_STEER_MAG = 0.4
W_STEER_RATE = 1.2
W_OBSTACLE = 8.0


def _reference_y_at(x_query: float, planner: PlannerResult, state: VehicleState) -> float:
    """Interpolate the planner's reference path to get y at a given x."""
    xs = [state.x] + [p[0] for p in planner.reference_path]
    ys = [state.y] + [p[1] for p in planner.reference_path]
    return float(np.interp(x_query, xs, ys))


def mpc_controller(state: VehicleState, planner: PlannerResult,
                    prev_steering: float) -> ControlResult:
    """
    Sampling-horizon MPC: for each candidate constant steering, roll
    the vehicle model forward MPC_HORIZON steps and score the rollout
    against the planner's reference trajectory, heading alignment,
    steering effort/smoothness, and obstacle clearance. Selects the
    steering that minimizes total cost (Section 5).
    """
    best_steering = 0.0
    best_cost = float("inf")

    for steer in STEERING_CANDIDATES:
        x, y, yaw, v = state.x, state.y, state.yaw, state.v
        cost = 0.0
        for _ in range(MPC_HORIZON):
            x, y, yaw, v = bicycle_step(x, y, yaw, v, steer, 0.0, MPC_DT, VEH.wheelbase)
            ref_y = _reference_y_at(x, planner, state)
            cost += W_TRACK * (y - ref_y) ** 2
            cost += W_HEADING * (yaw ** 2)
            for (ox, oy, orad) in state.obstacles:
                d = math.hypot(x - ox, y - oy)
                clearance = d - (orad + VEH.width / 2.0)
                if clearance < 0.3:
                    cost += W_OBSTACLE * max(0.0, 0.3 - clearance) ** 2

        cost += W_STEER_MAG * steer ** 2
        cost += W_STEER_RATE * (steer - prev_steering) ** 2

        if cost < best_cost:
            best_cost = cost
            best_steering = steer

    # Adaptive throttle: slow down for sharp steering or close obstacles
    steer_severity = abs(best_steering) / VEH.max_steering_rad
    obstacle_factor = 1.0
    if planner.closest_obstacle_dist < 1.0:
        obstacle_factor = float(np.clip(planner.closest_obstacle_dist / 1.0, 0.25, 1.0))
    speed_target = WORLD.target_speed * (1.0 - 0.5 * steer_severity) * obstacle_factor
    speed_target = max(0.15, speed_target)

    speed_error = speed_target - state.v
    accel = float(np.clip(speed_error * 1.5, VEH.max_decel, VEH.max_accel))

    return ControlResult(
        steering=best_steering, throttle_accel=accel,
        steering_candidates_evaluated=len(STEERING_CANDIDATES),
        horizon=MPC_HORIZON, cost=best_cost,
    )


# ============================================================
# 7. EMERGENCY SAFETY LAYER
# ============================================================

CRITICAL_OBSTACLE_DIST = 0.03   # m clearance -> hard stop (very last resort;
                                 # the planner's own safety margin normally
                                 # keeps well clear of this)
COMM_TIMEOUT_S = 0.5            # hardware-mode watchdog (unused in sim)


@dataclass
class SafetyResult:
    override: bool
    reason: str


def safety_check(state: VehicleState, planner: PlannerResult) -> SafetyResult:
    # Invalid state (NaN/Inf from a bad estimate)
    for val in (state.x, state.y, state.yaw, state.v):
        if not math.isfinite(val):
            return SafetyResult(True, "Invalid state (non-finite value)")

    # No safe path found by the planner
    if not planner.feasible:
        return SafetyResult(True, "No collision-free trajectory available")

    # Obstacle too close, even along the selected/planned path
    if planner.closest_obstacle_dist < CRITICAL_OBSTACLE_DIST:
        return SafetyResult(True, "Obstacle inside critical clearance")

    return SafetyResult(False, "Nominal")


# ============================================================
# 8. ACTUATION  (simulation vs. hardware — same call signature)
# ============================================================

def apply_to_hardware(steering_rad: float, accel: float):
    """
    Hardware-mode actuation hook: map steering (rad) to the JetRacer's
    normalized [-1, 1] car.steering, map accel to car.throttle, and
    write to the real actuators. Not runnable here (no JetRacer
    attached) — mirrors the SAFE_MODE guard in the original script.
    """
    if not SAFE_MODE:
        raise RuntimeError(
            "Hardware actuation requested but SAFE_MODE is off and no "
            "JetRacer/NvidiaRacecar interface is attached in this "
            "environment. Wire this to car.steering / car.throttle."
        )
    # SAFE_MODE: intentionally do nothing physical.
    return


# ============================================================
# 9. MAIN SIMULATION LOOP
# ============================================================

@dataclass
class StepLog:
    t: float
    state: VehicleState
    planner: PlannerResult
    control: ControlResult
    safety: SafetyResult
    applied_steering: float
    applied_accel: float


def run_simulation() -> List[StepLog]:
    world = SimulatedWorld(VEH, WORLD) if MODE == "simulation" else None
    prev_offset = 0.0
    prev_steering = 0.0
    logs: List[StepLog] = []

    n_steps = int(WORLD.total_time / WORLD.dt)
    print("=" * 60)
    print("AGENTIC iMPC — JETRACER SIMULATION")
    print("=" * 60)
    print(f"MODE: {MODE}   SAFE_MODE: {SAFE_MODE}")
    print(f"Vehicle: length={VEH.length}m width={VEH.width}m wheelbase={VEH.wheelbase}m")
    print(f"Road half-width: {WORLD.road_half_width}m   Obstacles: {len(WORLD.obstacles)}")
    print("-" * 60)

    for step in range(n_steps):
        t = step * WORLD.dt

        # --- PERCEPTION / STATE ESTIMATION ---
        state = get_state(world)

        # --- LOCAL PLANNER ---
        planner = local_planner(state, prev_offset, WORLD)

        # --- MPC CONTROLLER ---
        control = mpc_controller(state, planner, prev_steering)

        # --- SAFETY LAYER (can override planner/MPC output) ---
        safety = safety_check(state, planner)
        if safety.override:
            applied_steering, applied_accel = 0.0, VEH.max_decel
        else:
            applied_steering, applied_accel = control.steering, control.throttle_accel

        # --- ACTUATION ---
        if MODE == "simulation":
            world.apply_control(applied_steering, applied_accel, WORLD.dt)
        else:
            apply_to_hardware(applied_steering, applied_accel)

        logs.append(StepLog(t, state, planner, control, safety,
                             applied_steering, applied_accel))

        if step % 10 == 0 or safety.override:
            print(
                f"t={t:5.2f}s  x={state.x:6.2f} y={state.y:+5.2f} v={state.v:4.2f}  "
                f"plan={planner.label:10s}  steer={applied_steering:+.3f} "
                f"accel={applied_accel:+.2f}  "
                f"{'*** SAFETY OVERRIDE: ' + safety.reason if safety.override else ''}"
            )

        prev_offset = planner.selected_offset
        prev_steering = applied_steering

        if state.x >= WORLD.goal_x:
            print("-" * 60)
            print(f"GOAL REACHED at t={t:.2f}s, x={state.x:.2f}m")
            break

    print("=" * 60)
    return logs


# ============================================================
# 10. VISUALIZATION (matches the live-dashboard mockup)
# ============================================================

def render_dashboard(logs: List[StepLog], gif_path: str, png_path: str):
    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1])
    ax_world = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])
    ax_text.axis("off")

    ax_world.set_xlim(-0.5, WORLD.road_length)
    ax_world.set_ylim(-WORLD.road_half_width - 0.3, WORLD.road_half_width + 0.3)
    ax_world.set_aspect("equal")
    ax_world.set_title("AGENTIC iMPC — SIMULATION")
    ax_world.axhline(WORLD.road_half_width, color="black", linewidth=2)
    ax_world.axhline(-WORLD.road_half_width, color="black", linewidth=2)
    ax_world.axhline(0, color="gray", linestyle="--", linewidth=1)

    for (ox, oy, orad) in WORLD.obstacles:
        ax_world.add_patch(patches.Circle((ox, oy), orad, color="firebrick", alpha=0.85))
    ax_world.axvline(WORLD.goal_x, color="green", linestyle=":", linewidth=2)
    ax_world.text(WORLD.goal_x, WORLD.road_half_width + 0.08, "GOAL",
                  color="green", ha="center", fontsize=9)

    path_line, = ax_world.plot([], [], color="steelblue", linewidth=2, label="driven path")
    ref_line, = ax_world.plot([], [], color="orange", linewidth=1.5, linestyle="--",
                               label="planned trajectory")
    vehicle_patch = patches.Rectangle((0, 0), VEH.length, VEH.width,
                                       color="dodgerblue", zorder=5)
    ax_world.add_patch(vehicle_patch)
    ax_world.legend(loc="upper right", fontsize=8)

    text_box = ax_text.text(0.02, 0.98, "", va="top", ha="left",
                             family="monospace", fontsize=9, transform=ax_text.transAxes)

    xs_hist, ys_hist = [], []

    def frame_update(i):
        log = logs[i]
        xs_hist.append(log.state.x)
        ys_hist.append(log.state.y)
        path_line.set_data(xs_hist, ys_hist)

        ref_x = [log.state.x] + [p[0] for p in log.planner.reference_path]
        ref_y = [log.state.y] + [p[1] for p in log.planner.reference_path]
        ref_line.set_data(ref_x, ref_y)

        # Vehicle rectangle, oriented by yaw, anchored at rear axle
        yaw = log.state.yaw
        cx, cy = log.state.x, log.state.y
        t = matplotlib.transforms.Affine2D().rotate_around(cx, cy, yaw) + ax_world.transData
        vehicle_patch.set_xy((cx, cy - VEH.width / 2))
        vehicle_patch.set_transform(t)

        safe_txt = "OVERRIDE: " + log.safety.reason if log.safety.override else "ON (nominal)"
        info = (
            f"STATE\n"
            f"  x        {log.state.x:6.2f} m\n"
            f"  y        {log.state.y:+6.2f} m\n"
            f"  yaw      {math.degrees(log.state.yaw):6.1f} deg\n"
            f"  v        {log.state.v:6.2f} m/s\n\n"
            f"PERCEPTION\n"
            f"  lane err {log.state.lane_error:+6.3f} m\n"
            f"  obstacle {log.planner.closest_obstacle_dist:6.2f} m\n\n"
            f"PLANNER\n"
            f"  candidates {log.planner.num_candidates:3d}\n"
            f"  selected   {log.planner.label}\n"
            f"  offset     {log.planner.selected_offset:+.3f} m\n\n"
            f"MPC\n"
            f"  steering  {log.applied_steering:+.3f} rad\n"
            f"  throttle  {log.applied_accel:+.2f} m/s^2\n"
            f"  horizon   {log.control.horizon}\n\n"
            f"MODE: {MODE.upper():10s} SAFE: {safe_txt}\n"
            f"t = {log.t:5.2f} s"
        )
        text_box.set_text(info)
        return path_line, ref_line, vehicle_patch, text_box

    # Downsample frames for a reasonably sized GIF
    frame_step = max(1, len(logs) // 120)
    frame_indices = list(range(0, len(logs), frame_step))

    anim = animation.FuncAnimation(
        fig, lambda i: frame_update(frame_indices[i]),
        frames=len(frame_indices), interval=80, blit=False,
    )

    try:
        anim.save(gif_path, writer=animation.PillowWriter(fps=12))
        print(f"Saved animation: {gif_path}")
    except Exception as e:
        print(f"[WARN] Could not save GIF ({e}); saving final frame only.")

    frame_update(len(logs) - 1)
    fig.savefig(png_path, dpi=150)
    print(f"Saved summary image: {png_path}")
    plt.close(fig)


# ============================================================
# 11. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    logs = run_simulation()
    if len(logs) == 0:
        print("No simulation steps were logged — nothing to render.")
        sys.exit(0)

    render_dashboard(
        logs,
        gif_path="agentic_impc_run.gif",
        png_path="agentic_impc_summary.png",
    )
