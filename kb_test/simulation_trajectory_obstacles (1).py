"""
============================================================
SIMULATION -- reference trajectory + obstacle avoidance, no camera
============================================================
A standalone 2D world to develop and sanity-check the MPC before any
camera or hardware is involved. Uses the exact same mpc_planner.py
that drive_brain_mpc.py uses, so behavior you tune here transfers.

WORLD
-----
- A reference path (waypoints) the car should track.
- Static circular obstacles scattered near/around the path.
- The car only "sees" (i.e. the MPC's collision cost only includes)
  obstacles inside a forward-facing sensor window: a max range and a
  half-angle field of view. This is the simulation's equivalent of
  your camera's manually-calibrated perception window -- change
  SENSOR_RANGE / SENSOR_HALF_FOV_DEG to see how a narrower "vision"
  makes avoidance more reactive/fragile, exactly like shrinking the
  camera ROI would.

CONTROLS (while the OpenCV window is focused)
----------------------------------------------
  q          quit
  r          reset to the start of the path
  [ / ]      shrink / grow the sensor window (range)
  , / .      shrink / grow the sensor field of view

If no display is available (headless container, CI, etc.) it still
runs and prints progress/metrics instead of drawing -- set SHOW=False
below or it auto-detects via the DISPLAY env var.
============================================================
"""

import os
import time

import cv2
import numpy as np

from mpc_planner import KinematicBicycle, SamplingMPC, nearest_ref_point_cost


# ============================================================
# CONFIG
# ============================================================

SHOW = bool(os.environ.get("DISPLAY"))   # auto-detects; set True/False to force

CANVAS_W, CANVAS_H = 900, 700
PX_PER_M = 60.0            # world (meters) <-> canvas (pixels)

# --- Reference trajectory: an S-curve lane, in meters ---
def make_reference_path():
    xs = np.linspace(0.5, 12.5, 60)
    ys = 3.5 + 1.4 * np.sin(xs / 2.3)
    return list(zip(xs, ys))

# --- Obstacles: (x, y, radius) in meters. A mix of on-path and off-path. ---
OBSTACLES = [
    (3.0, 4.3, 0.35),
    (5.4, 3.0, 0.30),
    (7.8, 4.7, 0.40),
    (9.6, 2.6, 0.30),
    (2.0, 2.2, 0.25),   # off to the side, never actually blocks the path
    (11.0, 4.6, 0.35),
]

# --- Vehicle ---
VEHICLE_RADIUS_M = 0.28
WHEELBASE_M = 0.30
MAX_STEER_RAD = 0.5
MAX_SPEED_MPS = 1.2
MIN_SPEED_MPS = 0.0

# --- Perception window (this sim's equivalent of the camera ROI) ---
SENSOR_RANGE_M = 3.5
SENSOR_HALF_FOV_DEG = 55.0

# --- MPC ---
HORIZON = 10
DT = 0.1
NUM_SAMPLES = 250
COLLISION_PENALTY = 8000.0
TRACK_WEIGHT = 1.2
CONTROL_WEIGHT = 0.03
PROGRESS_WEIGHT = 2.5    # rewards net +x progress along the path, not just staying near it

STEP_HZ = 20


# ============================================================
# PERCEPTION: which obstacles are inside the sensor window right now
# ============================================================

def visible_obstacles(state, obstacles, sensor_range, half_fov_rad):
    x, y, theta, _ = state
    visible = []
    for (ox, oy, r) in obstacles:
        dx, dy = ox - x, oy - y
        dist = np.hypot(dx, dy)
        if dist > sensor_range + r:
            continue
        bearing = np.arctan2(dy, dx) - theta
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
        if abs(bearing) > half_fov_rad:
            continue
        visible.append((ox, oy, r))
    return visible


def make_cost_fn(ref_points, visible_obs, vehicle_radius):
    track_cost = nearest_ref_point_cost(ref_points)
    obs = np.array([[o[0], o[1], o[2]] for o in visible_obs]) if visible_obs else None

    def cost_fn(traj, seqs):
        N, T1, _ = traj.shape
        costs = np.zeros(N)
        xy = traj[:, 1:, 0:2]  # (N, T, 2), skip t=0 (current state)
        costs += TRACK_WEIGHT * np.sum(track_cost(xy), axis=1)

        if obs is not None:
            for (ox, oy, orad) in obs:
                d = np.hypot(xy[:, :, 0] - ox, xy[:, :, 1] - oy)  # (N, T)
                hit = d < (orad + vehicle_radius)
                costs += np.sum(np.where(hit, COLLISION_PENALTY, 0.0), axis=1)

        costs += CONTROL_WEIGHT * np.sum(seqs ** 2, axis=(1, 2))
        costs -= PROGRESS_WEIGHT * xy[:, -1, 0]   # reward net +x progress (path runs left-to-right)
        return costs

    return cost_fn


# ============================================================
# DRAWING
# ============================================================

def world_to_canvas(x, y):
    return int(x * PX_PER_M) + 40, CANVAS_H - (int(y * PX_PER_M) + 40)


def draw_frame(state, path, obstacles, visible_obs, planned_traj, sensor_range, half_fov_rad, trail):
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 245, dtype=np.uint8)

    # reference path
    pts = [world_to_canvas(x, y) for (x, y) in path]
    for i in range(len(pts) - 1):
        cv2.line(canvas, pts[i], pts[i + 1], (200, 160, 60), 2)

    # trail
    for i in range(1, len(trail)):
        cv2.line(canvas, world_to_canvas(*trail[i - 1][:2]), world_to_canvas(*trail[i][:2]), (180, 180, 180), 1)

    # obstacles
    visible_set = {(o[0], o[1]) for o in visible_obs}
    for (ox, oy, orad) in obstacles:
        c = world_to_canvas(ox, oy)
        color = (0, 0, 220) if (ox, oy) in visible_set else (170, 170, 170)
        cv2.circle(canvas, c, max(2, int(orad * PX_PER_M)), color, -1)

    x, y, theta, v = state

    # sensor window (perception window equivalent)
    fov_pts = [world_to_canvas(x, y)]
    for a in np.linspace(-half_fov_rad, half_fov_rad, 20):
        fx = x + sensor_range * np.cos(theta + a)
        fy = y + sensor_range * np.sin(theta + a)
        fov_pts.append(world_to_canvas(fx, fy))
    fov_pts.append(world_to_canvas(x, y))
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [np.array(fov_pts, dtype=np.int32)], (255, 240, 200))
    canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)

    # planned trajectory
    if planned_traj is not None:
        ppts = [world_to_canvas(px, py) for (px, py, *_r) in planned_traj]
        for i in range(len(ppts) - 1):
            cv2.line(canvas, ppts[i], ppts[i + 1], (0, 160, 0), 2)

    # vehicle
    vc = world_to_canvas(x, y)
    tip = world_to_canvas(x + 0.35 * np.cos(theta), y + 0.35 * np.sin(theta))
    cv2.circle(canvas, vc, max(3, int(VEHICLE_RADIUS_M * PX_PER_M)), (40, 40, 40), -1)
    cv2.line(canvas, vc, tip, (0, 220, 255), 3)

    cv2.putText(canvas, f"v={v:.2f} m/s  sensor_range={sensor_range:.1f}m  fov=+/-{np.degrees(half_fov_rad):.0f}deg",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
    cv2.putText(canvas, "q quit | r reset | [ ] range | , . fov",
                (12, CANVAS_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
    return canvas


# ============================================================
# MAIN
# ============================================================

def main():
    path = make_reference_path()
    model = KinematicBicycle(wheelbase=WHEELBASE_M, max_steer=MAX_STEER_RAD,
                              max_speed=MAX_SPEED_MPS, min_speed=MIN_SPEED_MPS)
    mpc = SamplingMPC(model, horizon=HORIZON, dt=DT, num_samples=NUM_SAMPLES)

    sensor_range = SENSOR_RANGE_M
    half_fov_rad = np.radians(SENSOR_HALF_FOV_DEG)

    def reset_state():
        x0, y0 = path[0]
        return np.array([x0, y0, 0.0, 0.0])

    state = reset_state()
    trail = [tuple(state[:2]) + (0,)]
    step = 0
    period = 1.0 / STEP_HZ
    goal = np.array(path[-1])

    print("[SIM] Running. q=quit r=reset [ ]=sensor range , .=sensor FOV")

    try:
        while True:
            t0 = time.time()

            vis = visible_obstacles(state, OBSTACLES, sensor_range, half_fov_rad)
            cost_fn = make_cost_fn(path, vis, VEHICLE_RADIUS_M)
            control, planned_traj, cost = mpc.plan(state, cost_fn)

            if cost >= COLLISION_PENALTY:
                control = np.array([0.0, -MAX_SPEED_MPS])  # brake hard, don't guess
                mpc.reset_warm_start()

            # advance one real step using the model directly (not a rollout copy)
            x, y, theta, v = state
            steer, accel = np.clip(control[0], -model.max_steer, model.max_steer), control[1]
            v = float(np.clip(v + accel * DT, model.min_speed, model.max_speed))
            x = x + v * np.cos(theta) * DT
            y = y + v * np.sin(theta) * DT
            theta = theta + (v / model.L) * np.tan(steer) * DT
            state = np.array([x, y, theta, v])
            trail.append((x, y, step))
            step += 1

            dist_to_goal = np.hypot(x - goal[0], y - goal[1])
            min_clearance = min((np.hypot(x - ox, y - oy) - orad for (ox, oy, orad) in OBSTACLES), default=999)

            if step % 20 == 0:
                print(f"\r[SIM] step={step:5d} v={v:.2f} dist_to_goal={dist_to_goal:.2f} "
                      f"min_clearance={min_clearance:.2f} visible_obs={len(vis)}   ", end="", flush=True)

            if dist_to_goal < 0.3:
                print("\n[SIM] Reached goal. Resetting.")
                state = reset_state()
                trail = [tuple(state[:2]) + (0,)]
                mpc.reset_warm_start()

            if SHOW:
                frame = draw_frame(state, path, OBSTACLES, vis, planned_traj, sensor_range, half_fov_rad, trail)
                cv2.imshow("Trajectory + Obstacle Avoidance (MPC sim)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    state = reset_state()
                    trail = [tuple(state[:2]) + (0,)]
                    mpc.reset_warm_start()
                elif key == ord("["):
                    sensor_range = max(0.5, sensor_range - 0.25)
                elif key == ord("]"):
                    sensor_range = min(10.0, sensor_range + 0.25)
                elif key == ord(","):
                    half_fov_rad = max(np.radians(10), half_fov_rad - np.radians(5))
                elif key == ord("."):
                    half_fov_rad = min(np.radians(90), half_fov_rad + np.radians(5))
            else:
                if step > 4000:
                    print("\n[SIM] Headless run limit reached, stopping.")
                    break

            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        if SHOW:
            cv2.destroyAllWindows()
        print("\n[SIM] Done.")


if __name__ == "__main__":
    main()
