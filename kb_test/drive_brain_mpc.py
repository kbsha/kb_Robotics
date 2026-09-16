"""
============================================================
DRIVE BRAIN -- MPC edition
============================================================
Two changes from the plain-corridor version, per what you asked for:

1. THE PERCEPTION WINDOW IS THE ENTIRE VISUAL INPUT, NOT AN ADVISORY
   CROP. You calibrate it manually with the on-screen sliders (same
   idea as the old Tkinter ROI sliders, now as OpenCV trackbars so it
   works headless-friendly and over SSH+X11 the same way). Everything
   downstream -- edge/obstacle detection, the reference path, the MPC
   cost function -- only ever receives the cropped window. Nothing
   outside it is read, computed on, or can influence a decision. If
   you calibrate the window too small to see an obstacle, the system
   will not avoid that obstacle -- that's the actual meaning of "the
   perception window IS what the model considers", not a metaphor.

2. OPTIMAL DRIVING = MPC, NOT A PROPORTIONAL CONTROLLER. Every frame:
   build an obstacle distance-map from everything inside the window,
   build a short reference path down the drivable corridor, then let
   SamplingMPC roll out and score candidate trajectories and apply the
   best one's first control. See mpc_planner.py for the planner itself.

Actuator modes are unchanged from drive_brain.py: MODE = "sim" (now,
on your computer) / "local_car" / "remote_car" (once the JetRacer
exists). See that file's docstring for the brain/muscle split
rationale -- it's identical here.
============================================================
"""

import atexit
import json
import os
import signal
import socket
import sys
import threading
import time

import cv2
import numpy as np

from mpc_planner import KinematicBicycle, SamplingMPC, nearest_ref_point_cost

try:
    from jetracer.nvidia_racecar import NvidiaRacecar
    JETRACER_AVAILABLE = True
except Exception:
    JETRACER_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

MODE = "sim"   # "sim" | "local_car" | "remote_car"

CAMERA_TYPE = "auto"
CAMERA_INDEX = 0
FALLBACK_VIDEO = "test_data/road2.mp4"
CAP_WIDTH, CAP_HEIGHT = 320, 240
CAP_FPS = 30
TARGET_LOOP_HZ = 12          # MPC is heavier per-frame than the P-controller was

# --- Vehicle calibration (pixels in the capture frame) ---
VEHICLE_WIDTH_PX = 80
VEHICLE_LENGTH_PX = 120      # used as the bicycle model's wheelbase

# --- Perception window -- manually calibrated via trackbars below.
#     These are just the STARTING values; drag the sliders to set
#     the real window once you can see the video. ---
INITIAL_ROI = (10, int(CAP_HEIGHT * 0.30), CAP_WIDTH - 20, int(CAP_HEIGHT * 0.65))

# --- Safety ---
SAFETY_MARGIN = 1.25            # required clearance = vehicle_width * this
MAX_THROTTLE = 0.15
MAX_STEER_RAD = 0.5             # ~28.6 deg, matches typical RC servo throw
MAX_LOOP_STALL_S = 0.5
MAX_FRAME_AGE_S = 0.5
STARTUP_GRACE_S = 3.0

# --- MPC tuning ---
MPC_HORIZON = 8
MPC_DT = 1.0 / TARGET_LOOP_HZ
MPC_SAMPLES = 150
MPC_COLLISION_PENALTY = 8000.0
MPC_TRACK_WEIGHT = 0.03
MPC_CONTROL_WEIGHT = 0.05
MPC_PROGRESS_WEIGHT = 0.08       # rewards actually moving forward, not just staying safe
MPC_MAX_SIM_SPEED = 55.0        # pixels/sec the model is allowed to *plan* at
                                  # (mapped down to MAX_THROTTLE for the real motor)

# --- Remote mode networking ---
ROBOT_HOST = "192.168.1.50"
ROBOT_PORT = 5555


# ============================================================
# ACTUATOR ABSTRACTION (same contract as drive_brain.py)
# ============================================================

class Actuator:
    def apply(self, steering, throttle):
        raise NotImplementedError

    def stop(self, latch=False):
        raise NotImplementedError

    def clear_latch(self):
        pass

    @property
    def latched(self):
        return False


class SimActuator(Actuator):
    def __init__(self):
        self.last = (0.0, 0.0)
        self._latch = False
        print("[ACTUATOR] SIM -- no commands sent anywhere")

    def apply(self, steering, throttle):
        if self._latch:
            steering, throttle = 0.0, 0.0
        self.last = (steering, throttle)

    def stop(self, latch=False):
        self.last = (0.0, 0.0)
        if latch:
            self._latch = True

    def clear_latch(self):
        self._latch = False

    @property
    def latched(self):
        return self._latch


class LocalCarActuator(Actuator):
    def __init__(self):
        if not JETRACER_AVAILABLE:
            raise RuntimeError("MODE='local_car' needs the jetracer library installed here.")
        self._car = NvidiaRacecar()
        self._car.steering, self._car.throttle = 0.0, 0.0
        self._lock = threading.Lock()
        self._latch = False
        self.last = (0.0, 0.0)
        atexit.register(self.stop, True)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, lambda s, f: (self.stop(True), sys.exit(0)))
            except Exception:
                pass
        print("[ACTUATOR] LOCAL_CAR -- driving real motors in-process")

    def apply(self, steering, throttle):
        with self._lock:
            if self._latch:
                steering, throttle = 0.0, 0.0
            self.last = (steering, throttle)
            try:
                self._car.steering = float(steering)
                self._car.throttle = float(throttle)
            except Exception as e:
                print("[ACTUATOR] write failed, forcing stop:", e)
                self.stop(latch=True)

    def stop(self, latch=False):
        with self._lock:
            self.last = (0.0, 0.0)
            if latch:
                self._latch = True
            try:
                self._car.throttle, self._car.steering = 0.0, 0.0
            except Exception:
                pass

    def clear_latch(self):
        with self._lock:
            self._latch = False

    @property
    def latched(self):
        with self._lock:
            return self._latch


class RemoteCarActuator(Actuator):
    def __init__(self, host, port):
        self.host, self.port = host, port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0
        self.last = (0.0, 0.0)
        self._latch = False
        atexit.register(self.stop, True)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, lambda s, f: (self.stop(True), sys.exit(0)))
            except Exception:
                pass
        print(f"[ACTUATOR] REMOTE_CAR -- UDP to {host}:{port}")

    def _send(self, steering, throttle, e_stop=False):
        self.seq += 1
        payload = json.dumps({"seq": self.seq, "t": time.time(),
                               "steering": float(steering), "throttle": float(throttle),
                               "e_stop": bool(e_stop)}).encode("utf-8")
        try:
            self.sock.sendto(payload, (self.host, self.port))
        except Exception as e:
            print("[ACTUATOR] UDP send failed:", e)

    def apply(self, steering, throttle):
        if self._latch:
            steering, throttle = 0.0, 0.0
        self.last = (steering, throttle)
        self._send(steering, throttle)

    def stop(self, latch=False):
        self.last = (0.0, 0.0)
        if latch:
            self._latch = True
        self._send(0.0, 0.0, e_stop=latch)

    def clear_latch(self):
        self._latch = False

    @property
    def latched(self):
        return self._latch


def build_actuator():
    if MODE == "sim":
        return SimActuator()
    if MODE == "local_car":
        return LocalCarActuator()
    if MODE == "remote_car":
        return RemoteCarActuator(ROBOT_HOST, ROBOT_PORT)
    raise ValueError(f"Unknown MODE: {MODE}")


# ============================================================
# CAMERA
# ============================================================

def _csi_gst_pipeline(width, height, fps):
    return (
        f"nvarguscamerasrc ! video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={fps}/1, format=NV12 ! nvvidconv flip-method=0 ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1"
    )


def open_camera():
    mode = CAMERA_TYPE
    if mode == "auto":
        cap = cv2.VideoCapture(_csi_gst_pipeline(CAP_WIDTH, CAP_HEIGHT, CAP_FPS), cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap, "csi"
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
            return cap, "usb"
        cap = cv2.VideoCapture(FALLBACK_VIDEO)
        if cap.isOpened():
            return cap, "video_file"
        raise RuntimeError("No camera source could be opened")
    if mode == "csi":
        cap = cv2.VideoCapture(_csi_gst_pipeline(CAP_WIDTH, CAP_HEIGHT, CAP_FPS), cv2.CAP_GSTREAMER)
    elif mode == "usb":
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
    elif mode == "video_file":
        cap = cv2.VideoCapture(FALLBACK_VIDEO)
    else:
        raise ValueError(f"Unknown CAMERA_TYPE: {mode}")
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera in mode '{mode}'")
    return cap, mode


# ============================================================
# PERCEPTION WINDOW -- manually calibrated, hard boundary of "vision"
# ============================================================

class PerceptionWindow:
    """Owns the ROI trackbars and crops frames. Everything downstream
    of `.crop()` never sees the rest of the frame again -- that's the
    enforcement mechanism, not just a naming convention."""

    def __init__(self, cap_w, cap_h, initial, interactive):
        self.cap_w, self.cap_h = cap_w, cap_h
        self.interactive = interactive
        x, y, w, h = initial
        if interactive:
            cv2.namedWindow("Perception Window Calibration", cv2.WINDOW_NORMAL)
            cv2.createTrackbar("X", "Perception Window Calibration", x, cap_w, lambda v: None)
            cv2.createTrackbar("Y", "Perception Window Calibration", y, cap_h, lambda v: None)
            cv2.createTrackbar("W", "Perception Window Calibration", w, cap_w, lambda v: None)
            cv2.createTrackbar("H", "Perception Window Calibration", h, cap_h, lambda v: None)
        else:
            self._fixed = (x, y, w, h)

    def get_roi(self):
        if not self.interactive:
            return self._fixed
        x = cv2.getTrackbarPos("X", "Perception Window Calibration")
        y = cv2.getTrackbarPos("Y", "Perception Window Calibration")
        w = max(20, cv2.getTrackbarPos("W", "Perception Window Calibration"))
        h = max(20, cv2.getTrackbarPos("H", "Perception Window Calibration"))
        x = min(x, self.cap_w - 20)
        y = min(y, self.cap_h - 20)
        w = min(w, self.cap_w - x)
        h = min(h, self.cap_h - y)
        return x, y, w, h

    def crop(self, frame):
        x, y, w, h = self.get_roi()
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


# ============================================================
# OBSTACLE MAP + REFERENCE PATH (built ONLY from the cropped window)
# ============================================================

def auto_canny(gray, sigma=0.33):
    v = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)


def obstacle_distance_map(window_bgr):
    """Distance (px) from every pixel to the nearest detected obstacle edge,
    computed only within the perception window. Larger = more clearance."""
    gray = cv2.cvtColor(window_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = auto_canny(gray)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    free = 255 - edges
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    return dist, edges


def build_reference_points(edges, vehicle_width_px, current_x_ref, margin):
    """Coarse drivable-corridor centerline, near-to-far, in (lateral_x,
    forward_s) pixel coordinates -- forward_s = 0 at the bottom of the
    window (the car), growing toward the top (ahead)."""
    h, w = edges.shape
    row_fracs = [0.95, 0.85, 0.72, 0.58, 0.45]
    points = []

    def free_segments(row):
        occ = row > 0
        segs, start = [], None
        for i, is_occ in enumerate(occ):
            if not is_occ and start is None:
                start = i
            elif is_occ and start is not None:
                segs.append((start, i)); start = None
        if start is not None:
            segs.append((start, len(occ)))
        return segs

    ref = np.clip(current_x_ref, 0, w - 1)
    for frac in row_fracs:
        y = int(np.clip(frac * (h - 1), 0, h - 1))
        segs = free_segments(edges[y, :])
        if not segs:
            continue
        best, best_dist = None, None
        for (s, e) in segs:
            dist = 0 if s <= ref <= e else min(abs(ref - s), abs(ref - e))
            width_px = e - s
            if best is None or dist < best_dist or (dist == best_dist and width_px > (best[1] - best[0])):
                best, best_dist = (s, e), dist
        s, e = best
        if (e - s) < vehicle_width_px * margin:
            continue  # too narrow here to be worth steering toward
        cx = (s + e) / 2.0
        forward_s = (h - 1) - y
        points.append((cx, forward_s))
    return points


# ============================================================
# MPC COST FUNCTION -- collision (window-bounded) + reference tracking
# ============================================================

def make_cost_fn(dist_map, ref_points, vehicle_radius):
    H_map, W_map = dist_map.shape
    track_cost = nearest_ref_point_cost(ref_points)

    def cost_fn(traj, seqs):
        # traj: (N, T+1, 4) with cols x(lateral), y(forward_s), theta, v
        N, T1, _ = traj.shape
        costs = np.zeros(N)
        for t in range(1, T1):
            x = traj[:, t, 0]
            s = traj[:, t, 1]
            img_x = np.clip(x, 0, W_map - 1).astype(int)
            img_y = np.clip((H_map - 1) - s, 0, H_map - 1).astype(int)
            clearance = dist_map[img_y, img_x]
            costs += np.where(clearance < vehicle_radius, MPC_COLLISION_PENALTY, 0.0)
            costs += MPC_TRACK_WEIGHT * track_cost(np.stack([x, s], axis=-1))
        costs += MPC_CONTROL_WEIGHT * np.sum(seqs ** 2, axis=(1, 2))
        costs -= MPC_PROGRESS_WEIGHT * traj[:, -1, 1]   # reward net forward distance covered
        return costs

    return cost_fn


# ============================================================
# WATCHDOG + KEYBOARD E-STOP
# ============================================================

class Watchdog:
    def __init__(self):
        self.last_frame_t = time.time()
        self.last_loop_t = time.time()
        self.e_stop = False
        self.start_t = time.time()

    def mark_frame(self):
        self.last_frame_t = time.time()

    def mark_loop(self):
        self.last_loop_t = time.time()

    def trip_check(self):
        now = time.time()
        warmed_up = (now - self.start_t) > STARTUP_GRACE_S
        if self.e_stop:
            return True, "E-STOP"
        if warmed_up and (now - self.last_frame_t) > MAX_FRAME_AGE_S:
            return True, "CAMERA STALE"
        if warmed_up and (now - self.last_loop_t) > MAX_LOOP_STALL_S:
            return True, "LOOP STALL"
        return False, ""


def keyboard_watchdog_thread(watchdog):
    while True:
        try:
            line = sys.stdin.readline()
        except Exception:
            return
        if line == "":
            return
        cmd = line.strip().lower()
        if cmd in ("q", "stop", ""):
            watchdog.e_stop = True
            print("[E-STOP] triggered from keyboard")
        elif cmd == "go":
            watchdog.e_stop = False
            print("[E-STOP] cleared")


# ============================================================
# MAIN
# ============================================================

def main():
    actuator = build_actuator()
    cap, cam_mode = open_camera()
    watchdog = Watchdog()
    threading.Thread(target=keyboard_watchdog_thread, args=(watchdog,), daemon=True).start()

    show_display = bool(os.environ.get("DISPLAY")) or MODE == "sim"
    perception = PerceptionWindow(CAP_WIDTH, CAP_HEIGHT, INITIAL_ROI, interactive=show_display)

    model = KinematicBicycle(wheelbase=VEHICLE_LENGTH_PX, max_steer=MAX_STEER_RAD,
                              max_speed=MPC_MAX_SIM_SPEED, min_speed=0.0)
    mpc = SamplingMPC(model, horizon=MPC_HORIZON, dt=MPC_DT, num_samples=MPC_SAMPLES)

    center_x_ref = None
    period = 1.0 / TARGET_LOOP_HZ
    vehicle_radius_px = VEHICLE_WIDTH_PX * SAFETY_MARGIN / 2.0
    prev_v = 0.0

    print(f"[MAIN] mode={MODE} camera={cam_mode} target_hz={TARGET_LOOP_HZ} "
          f"mpc_samples={MPC_SAMPLES} horizon={MPC_HORIZON}")
    print("[MAIN] Drag the Perception Window Calibration sliders to set what the "
          "car is allowed to 'see'. Type 'q'+Enter to E-STOP, 'go'+Enter to clear.")

    try:
        while True:
            tick_start = time.time()
            ok, frame = cap.read()
            if ok and frame is not None:
                watchdog.mark_frame()
            tripped, reason = watchdog.trip_check()

            if not ok or frame is None:
                actuator.stop()
                time.sleep(0.02)
                continue

            window, (x1, y1, x2, y2) = perception.crop(frame)
            if center_x_ref is None:
                center_x_ref = (x2 - x1) / 2.0

            if tripped:
                actuator.stop(latch=(reason == "E-STOP"))
                mpc.reset_warm_start()
                action = f"STOPPED ({reason})"
                dist_map = edges = None
                ref_points = []
            else:
                if actuator.latched and reason == "" and not watchdog.e_stop:
                    actuator.clear_latch()

                if window.size == 0:
                    actuator.stop()
                    action = "EMPTY WINDOW - STOP"
                    dist_map = edges = None
                    ref_points = []
                else:
                    dist_map, edges = obstacle_distance_map(window)
                    ref_points = build_reference_points(edges, VEHICLE_WIDTH_PX, center_x_ref, SAFETY_MARGIN)

                    state = (center_x_ref, 0.0, np.pi / 2.0, prev_v)
                    cost_fn = make_cost_fn(dist_map, ref_points, vehicle_radius_px)
                    best_control, best_traj, best_cost = mpc.plan(state, cost_fn)

                    steer_rad, accel = best_control
                    next_v = float(np.clip(prev_v + accel * MPC_DT, 0.0, MPC_MAX_SIM_SPEED))

                    if best_cost >= MPC_COLLISION_PENALTY:
                        # even the best rollout the planner could find still
                        # collides inside the window -- hard stop, don't guess.
                        steering_cmd, throttle_cmd = 0.0, 0.0
                        action = "NO COLLISION-FREE TRAJECTORY - STOP"
                        prev_v = 0.0
                        mpc.reset_warm_start()
                    else:
                        steering_cmd = float(np.clip(steer_rad / MAX_STEER_RAD, -1.0, 1.0))
                        throttle_cmd = float(np.clip(
                            (next_v / MPC_MAX_SIM_SPEED) * MAX_THROTTLE, 0.0, MAX_THROTTLE))
                        prev_v = next_v
                        if steering_cmd < -0.08:
                            action = "MPC: STEER LEFT"
                        elif steering_cmd > 0.08:
                            action = "MPC: STEER RIGHT"
                        else:
                            action = "MPC: STRAIGHT"

                    actuator.apply(steering_cmd, throttle_cmd)

            if int(tick_start * 2) % 10 == 0:
                s, t = actuator.last
                print(f"\r[{MODE:10s}][{action:32s}] steer={s:+.2f} throttle={t:.2f}   ",
                      end="", flush=True)

            if show_display:
                disp = frame.copy()
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(disp, "PERCEPTION WINDOW (only this is 'seen')", (x1, max(15, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
                for (px, ps) in ref_points or []:
                    cv2.circle(disp, (int(x1 + px), int(y2 - ps)), 3, (0, 255, 0), -1)
                cv2.putText(disp, f"[{MODE}] {action}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.imshow("Drive Brain (MPC)", disp)
                if edges is not None:
                    cv2.imshow("Perception Window Calibration", edges)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    watchdog.e_stop = True

            watchdog.mark_loop()
            elapsed = time.time() - tick_start
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        actuator.stop(latch=True)
        cap.release()
        if show_display:
            cv2.destroyAllWindows()
        print("\n[MAIN] Shutdown complete.")


if __name__ == "__main__":
    main()
