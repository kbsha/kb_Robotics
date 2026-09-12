"""
============================================================
DRIVE BRAIN -- perception + decision, actuator-agnostic
============================================================

This is the "brain" half of a brain/muscle split:

  BRAIN  (this file)   -- camera in, corridor perception, control
                           math, decision out. Runs on YOUR COMPUTER
                           today, before you have a JetRacer.
  MUSCLE (jetracer_server.py) -- the only code allowed to write to
                           real motors. Runs ON the JetRacer once you
                           have it. Has its own independent watchdog
                           -- it does not trust the network to keep
                           it safe.

ONE CONFIG LINE SWITCHES WHAT HAPPENS TO THE DECISION:

  MODE = "sim"         -> decisions are printed + drawn on screen,
                           NOTHING is ever sent anywhere. Use this now.
  MODE = "local_car"   -> this script is running directly on the
                           Jetson with a car attached; writes motors
                           in-process (needs `jetracer` installed).
  MODE = "remote_car"  -> this script keeps running on your laptop;
                           decisions are sent over UDP to
                           jetracer_server.py running on the car's
                           Jetson. Use this once the car exists and
                           you want to keep developing/perceiving from
                           your computer.

Nothing about the perception or control code changes between modes.
Only main() picks a different Actuator at the bottom.

REMOTE MODE IS NOT "SAFE BY DEFAULT"
-------------------------------------
Driving a real vehicle over Wi-Fi from a laptop adds latency and
packet loss on top of everything else. It only belongs at the very
slow, tightly supervised speeds the original design doc for this
project already called for, with someone standing next to the car
and jetracer_server.py's own watchdog active (it is, by default).
Don't raise MAX_THROTTLE in remote mode past what you've proven safe
in local_car mode first.
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

try:
    from jetracer.nvidia_racecar import NvidiaRacecar
    JETRACER_AVAILABLE = True
except Exception:
    JETRACER_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

MODE = "sim"   # "sim" | "local_car" | "remote_car"  <-- change this later

# --- Camera (works the same on your laptop and on the Jetson) ---
CAMERA_TYPE = "auto"        # "csi", "usb", "video_file", or "auto"
CAMERA_INDEX = 0
FALLBACK_VIDEO = "test_data/road5.mp4"
CAP_WIDTH, CAP_HEIGHT = 320, 240
CAP_FPS = 30
TARGET_LOOP_HZ = 15

# --- Vehicle footprint calibration (pixels, in the capture frame).
#     Made-up defaults for sim; measure your real car for local/remote. ---
VEHICLE_WIDTH_PX = 80
CENTER_X_OVERRIDE = None

ROI = (10, int(CAP_HEIGHT * 0.30), CAP_WIDTH - 20, int(CAP_HEIGHT * 0.65))

# --- Control tuning ---
SAFETY_WIDTH_MARGIN = 1.25
MAX_STEERING_NORM = 1.0
STEERING_RATE_LIMIT = 0.20
MAX_THROTTLE = 0.15          # hard ceiling, raise slowly once on real hardware
MIN_MOVING_THROTTLE = 0.09
THROTTLE_RATE_LIMIT = 0.06

# --- Watchdog (brain side) ---
MAX_FRAME_AGE_S = 0.5
MAX_LOOP_STALL_S = 0.5
STARTUP_GRACE_S = 3.0

# --- Remote mode networking ---
ROBOT_HOST = "192.168.1.50"   # set to the JetRacer's IP once it exists
ROBOT_PORT = 5555
REMOTE_SEND_HZ = TARGET_LOOP_HZ   # send a command every control tick


# ============================================================
# ACTUATOR ABSTRACTION
#   Every mode implements the same tiny interface: apply() / stop().
#   The perception+control code below never knows which one it has.
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
    """Never touches hardware. Just remembers the last command for the HUD."""
    def __init__(self):
        self.last = (0.0, 0.0)
        self._latch = False
        print("[ACTUATOR] SIM -- no commands sent anywhere, safe to run freely")

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
    """Runs on the Jetson itself, writes NvidiaRacecar in-process.
    Same safety contract as the standalone jetracer_corridor_drive.py:
    only object allowed to touch the motors, everything funnels
    through stop(), atexit/signal always zero it."""
    def __init__(self):
        if not JETRACER_AVAILABLE:
            raise RuntimeError(
                "MODE='local_car' but the jetracer library isn't installed here. "
                "Use MODE='sim' on your computer, or MODE='remote_car' to drive "
                "the JetRacer over the network from this machine."
            )
        self._car = NvidiaRacecar()
        self._car.steering = 0.0
        self._car.throttle = 0.0
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
                self._car.throttle = 0.0
                self._car.steering = 0.0
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
    """Brain stays on your laptop, sends {steering, throttle, seq, t}
    over UDP to jetracer_server.py running on the JetRacer. UDP because
    a dropped stale command should just be dropped, not retried and
    applied late -- jetracer_server.py's own watchdog is what actually
    keeps the car safe if packets stop arriving, not this class."""
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
        print(f"[ACTUATOR] REMOTE_CAR -- sending UDP commands to {host}:{port}")

    def _send(self, steering, throttle, e_stop=False):
        self.seq += 1
        payload = json.dumps({
            "seq": self.seq, "t": time.time(),
            "steering": float(steering), "throttle": float(throttle),
            "e_stop": bool(e_stop),
        }).encode("utf-8")
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
            print("[CAMERA] CSI camera opened via GStreamer")
            return cap, "csi"
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
            print("[CAMERA] Webcam/USB camera opened at index", CAMERA_INDEX)
            return cap, "usb"
        cap = cv2.VideoCapture(FALLBACK_VIDEO)
        if cap.isOpened():
            print("[CAMERA] No live camera found -- using fallback video:", FALLBACK_VIDEO)
            return cap, "video_file"
        raise RuntimeError("No camera source could be opened (csi/usb/fallback all failed)")

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
# FAST LAYER: free-space corridor detection (unchanged logic,
# auto-thresholded so lighting indoors/outdoors doesn't break it)
# ============================================================

def auto_canny(gray, sigma=0.33):
    v = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)


class FreeSpaceDetector:
    def __init__(self):
        self.smoothed_target_x = None
        self.alpha = 0.45

    @staticmethod
    def _edge_mask(bgr_roi):
        gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = auto_canny(gray)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        return edges

    @staticmethod
    def _free_segments(edge_row):
        occupied = edge_row > 0
        segments, start = [], None
        for i, is_occ in enumerate(occupied):
            if not is_occ and start is None:
                start = i
            elif is_occ and start is not None:
                segments.append((start, i))
                start = None
        if start is not None:
            segments.append((start, len(occupied)))
        return segments

    def analyze(self, roi_frame, vehicle_width_px, current_x_ref, required_margin):
        h, w = roi_frame.shape[:2]
        edges = self._edge_mask(roi_frame)

        row_fracs = [0.95, 0.85, 0.70, 0.55]
        weights = [0.40, 0.30, 0.20, 0.10]
        candidates = []
        nearest_width = None

        for frac, weight in zip(row_fracs, weights):
            y = int(np.clip(frac * (h - 1), 0, h - 1))
            segments = self._free_segments(edges[y, :])
            if not segments:
                continue
            ref = np.clip(current_x_ref, 0, w - 1)
            best, best_dist = None, None
            for (s, e) in segments:
                dist = 0 if s <= ref <= e else min(abs(ref - s), abs(ref - e))
                if best is None or dist < best_dist or (
                    dist == best_dist and (e - s) > (best[1] - best[0])
                ):
                    best, best_dist = (s, e), dist
            s, e = best
            width_px = e - s
            has_room = width_px >= vehicle_width_px * required_margin
            if frac == row_fracs[0]:
                nearest_width = width_px
            candidates.append((weight, (s + e) / 2.0, width_px, has_room))

        if not candidates:
            return {"target_x": current_x_ref, "corridor_width": 0.0,
                    "confidence": 0.0, "safe": False}

        usable = [c for c in candidates if c[3]] or candidates
        total_w = sum(c[0] for c in usable)
        target_x = sum(c[0] * c[1] for c in usable) / total_w
        min_width = min(c[2] for c in candidates)
        confidence = len(usable) / len(row_fracs)
        safe = nearest_width is not None and nearest_width >= vehicle_width_px * required_margin

        self.smoothed_target_x = target_x if self.smoothed_target_x is None else (
            self.alpha * target_x + (1 - self.alpha) * self.smoothed_target_x
        )

        return {"target_x": self.smoothed_target_x, "corridor_width": min_width,
                "nearest_width": nearest_width or 0.0, "confidence": float(confidence),
                "safe": bool(safe)}


class FastController:
    def __init__(self):
        self.prev_steering = 0.0
        self.prev_throttle = 0.0

    def update(self, corridor, current_x_ref, roi_w, advisory_bias=0.0):
        error = (corridor["target_x"] - current_x_ref) / max(roi_w / 2.0, 1.0)
        error = float(np.clip(error + advisory_bias, -1.0, 1.0))

        raw_steering = error * MAX_STEERING_NORM
        delta = float(np.clip(raw_steering - self.prev_steering,
                               -STEERING_RATE_LIMIT, STEERING_RATE_LIMIT))
        steering = float(np.clip(self.prev_steering + delta, -MAX_STEERING_NORM, MAX_STEERING_NORM))
        self.prev_steering = steering

        if not corridor["safe"] or corridor["confidence"] <= 0:
            target_throttle, action = 0.0, "NO SAFE CORRIDOR - STOP"
        else:
            tightness = min(1.0, corridor["corridor_width"] / (corridor["nearest_width"] + 1e-6))
            target_throttle = MAX_THROTTLE * corridor["confidence"] * (0.5 + 0.5 * tightness)
            target_throttle = max(MIN_MOVING_THROTTLE, min(MAX_THROTTLE, target_throttle))
            if steering < -0.08:
                action = "STEER LEFT"
            elif steering > 0.08:
                action = "STEER RIGHT"
            else:
                action = "STRAIGHT"

        t_delta = float(np.clip(target_throttle - self.prev_throttle,
                                 -THROTTLE_RATE_LIMIT, THROTTLE_RATE_LIMIT))
        throttle = float(np.clip(self.prev_throttle + t_delta, 0.0, MAX_THROTTLE))
        self.prev_throttle = throttle

        return steering, throttle, action


# ============================================================
# WATCHDOG + KEYBOARD E-STOP (brain side)
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

def build_actuator():
    if MODE == "sim":
        return SimActuator()
    if MODE == "local_car":
        return LocalCarActuator()
    if MODE == "remote_car":
        return RemoteCarActuator(ROBOT_HOST, ROBOT_PORT)
    raise ValueError(f"Unknown MODE: {MODE}")


def main():
    actuator = build_actuator()
    cap, cam_mode = open_camera()
    detector = FreeSpaceDetector()
    controller = FastController()
    watchdog = Watchdog()

    threading.Thread(target=keyboard_watchdog_thread, args=(watchdog,), daemon=True).start()

    show_display = bool(os.environ.get("DISPLAY")) or MODE == "sim"
    rx, ry, rw, rh = ROI
    center_x_ref_full = CENTER_X_OVERRIDE if CENTER_X_OVERRIDE is not None else CAP_WIDTH // 2
    period = 1.0 / TARGET_LOOP_HZ

    print(f"[MAIN] mode={MODE} camera={cam_mode} target_hz={TARGET_LOOP_HZ} "
          f"max_throttle={MAX_THROTTLE}")
    print("[MAIN] Type 'q' + Enter (or just Enter) to E-STOP. Type 'go' + Enter to clear it.")

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

            if tripped:
                actuator.stop(latch=(reason == "E-STOP"))
            else:
                if actuator.latched and reason == "" and not watchdog.e_stop:
                    actuator.clear_latch()

                x1, y1 = max(0, rx), max(0, ry)
                x2, y2 = min(frame.shape[1], rx + rw), min(frame.shape[0], ry + rh)
                roi_frame = frame[y1:y2, x1:x2]

                if roi_frame.size == 0:
                    actuator.stop()
                else:
                    current_x_ref = np.clip(center_x_ref_full - x1, 0, (x2 - x1) - 1)
                    corridor = detector.analyze(
                        roi_frame, VEHICLE_WIDTH_PX, current_x_ref, SAFETY_WIDTH_MARGIN
                    )
                    steering, throttle, action = controller.update(
                        corridor, current_x_ref, x2 - x1
                    )
                    actuator.apply(steering, throttle)

                    if int(tick_start * 2) % 10 == 0:
                        print(f"\r[{MODE:10s}][{action:22s}] steer={steering:+.2f} "
                              f"throttle={throttle:.2f} conf={corridor['confidence']:.2f} "
                              f"safe={corridor['safe']}   ", end="", flush=True)

                    if show_display:
                        disp = frame.copy()
                        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        color = (0, 255, 0) if corridor["safe"] else (0, 0, 255)
                        tx = int(x1 + corridor["target_x"])
                        cv2.line(disp, (tx, y2), (tx, y1), color, 2)
                        cv2.putText(disp, f"[{MODE}] {action} steer={steering:+.2f} thr={throttle:.2f}",
                                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.imshow("Drive Brain", disp)
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
