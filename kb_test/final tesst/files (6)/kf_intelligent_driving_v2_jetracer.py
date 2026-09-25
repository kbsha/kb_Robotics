"""
============================================================
KF-INTELLIGENT DRIVING (v2)
Interactive DriveFusion Perception + Simulation Cockpit
============================================================

WHAT CHANGED FROM v1
---------------------
The old version used DriveFusion's text output (parsed with regex,
updated every 1-3s) as the actual steering command. That is not a
real-time control loop -- a vehicle moving at any real speed travels
meters between VLM updates.

This version splits the system into two layers:

  1. FAST LAYER (every camera frame, no VLM dependency):
     - FreeSpaceDetector scans the dynamic perception window for the
       widest open corridor at least as wide as the calibrated vehicle
       footprint, wherever that space is -- not just "between two lane
       lines". This is what actually drives the steering + throttle.
     - A hard safety stop fires whenever no corridor wide enough for
       the vehicle exists near the front of the vehicle, independent
       of anything the VLM says.

  2. ADVISORY LAYER (DriveFusion, slow, ~1-3s):
     - Interprets voice/text commands ("turn left at the intersection",
       "look for a parking spot").
     - Reports hazards/scene description via voice + HUD.
     - Contributes a small directional BIAS to the fast layer's target
       (e.g. "prefer left") -- it can nudge, it cannot override safety.

JETRACER VARIANT
----------------
This copy sends fast_steering / fast_throttle over UDP to a JetRacer
(see jetracer_link.py and jetracer_server.py). It starts STOPPED; press
'g' in the video window for AUTO, 'm' for MANUAL, 'x'/SPACE to stop.
The IMPORTANT note below is from the original file and still applies to
the safety limits of a camera-only stack.

IMPORTANT
---------
This program visualizes perception and steering decisions and computes
simulated steering/throttle values. It does NOT send any command to a
physical actuator, motor controller, or drive-by-wire system -- there
is no serial/CAN/GPIO output anywhere in this file. Wiring the
`fast_steering` / `fast_throttle` values below to a real vehicle at
any but the slowest walking-pace, tightly supervised test speed
requires, at minimum: redundant non-camera ranging (ultrasonic/lidar)
that can force a stop independent of this vision pipeline, a physical
emergency stop, and a human safety operator with an override. A single
camera + VLM stack is a single point of failure and should not be the
sole thing standing between the vehicle and an obstacle.
============================================================
"""

import cv2
import numpy as np
import torch
import time
import threading
import textwrap
import re
import queue

import tkinter as tk

from jetracer_link import JetRacerLink

from PIL import Image


# ============================================================
# OPTIONAL VOICE
# ============================================================

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


from drivefusion import (
    DriveFusionForConditionalGeneration,
    DriveFusionProcessor,
)


















# ----------------------------------------test 




# ============================================================
# REMOTE JETRACER CAMERA
# ============================================================

JETRACER_IP = "10.0.127.250"     # <-- IP of the JetRacer (run `hostname -I` on it)
CAMERA_DEVICE = "http://%s:8081/video" % JETRACER_IP   # http, NOT https

# Only used if remote camera completely fails
FALLBACK_VIDEO = "test_data/road2.mp4"

WIDTH = 1280
HEIGHT = 720
FPS = 30



# ------------------test end ---------------







# ============================================================
# CONFIGURATION
#  to see the type -  (v4l2-ctl --list-devices)
# ============================================================

MODEL_ID = "DriveFusion/DriveFusion-V0.2"

# Remote JetRacer camera
# CAMERA_DEVICE = "https://10.0.120.38:8080"
# CAMERA_DEVICE = "http://10.0.120.38:8080/video"

# CAMERA_DEVICE = "/dev/video48"
FALLBACK_VIDEO = "test_data/road2.mp4"

WIDTH = 1280

HEIGHT = 720
FPS = 30

# VLM cadence -- this NO LONGER gates driving. It only gates how often
# the advisory/voice layer refreshes.
AI_INTERVAL = 1.0
MAX_NEW_TOKENS = 80

VOICE_LANGUAGE = "en-US"
VOICE_RATE = 185
VOICE_ENABLED = True
LISTEN_TIMEOUT = 4
PHRASE_TIME_LIMIT = 8

# --- Fast-layer safety/control tuning ---
SAFETY_WIDTH_MARGIN = 1.20      # required corridor width = vehicle_width * this
MAX_STEERING_DEG = 30.0
STEERING_RATE_LIMIT_DEG = 6.0   # max change per frame -> smooth, not jerky
BASE_THROTTLE = 0.30            # simulated units, 0..1
MIN_THROTTLE_FLOOR = 0.08
VLM_BIAS_WEIGHT = 0.15          # how much the advisory layer can nudge target (0..1)


# ============================================================
# GLOBAL STATE
# ============================================================

latest_frame = None
frame_lock = threading.Lock()

# --- JetRacer link ---
stream_ok = False          # True only when frames come from the real JetRacer stream
last_frame_time = 0.0      # time of the last frame received
link = JetRacerLink(JETRACER_IP, port=5005, token="change-me")   # token = AUTH_TOKEN on the car

camera_running = True
ai_running = True
ai_paused = False
ai_lock = threading.Lock()

current_prompt = (
    "Analyze the road and determine the safest "
    "next movement for the vehicle."
)

latest_response = "Waiting for DriveFusion..."
last_inference_time = 0.0
ai_count = 0
camera_fps = 0.0

new_prompt_event = threading.Event()

voice_running = True
voice_listening = False
voice_status = "VOICE READY"
speech_queue = queue.Queue()

# --- VLM advisory outputs (bias only, never authoritative) ---
vlm_bias_direction = 0.0     # -1 (left) .. +1 (right)
road_description = "Waiting..."
obstacle_description = "Waiting..."
lane_description = "Waiting..."

# --- Fast-layer authoritative outputs ---
fast_lock = threading.Lock()
fast_steering_deg = 0.0
fast_throttle = 0.0
fast_action = "INITIALIZING"
fast_corridor = None          # dict from FreeSpaceDetector, for HUD drawing

# --- Perception window (dynamic ROI) ---
roi_lock = threading.Lock()
roi_x, roi_y, roi_w, roi_h = 300, 260, 680, 400

# --- Vehicle calibration ---
vehicle_lock = threading.Lock()
vehicle_center_x = WIDTH // 2
vehicle_bottom_y = HEIGHT - 35
vehicle_length = 170
vehicle_width = 100
vehicle_offset_x = 0
vehicle_offset_y = 0
vehicle_scale = 1.0


# ============================================================
# CAMERA THREAD
# ============================================================

def camera_loop():
    global latest_frame, camera_running, camera_fps, stream_ok, last_frame_time

    print("\n[CAMERA] Opening:", CAMERA_DEVICE)
    # cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)

    cap = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_FFMPEG
    )


    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        print("[WARNING] Could not open camera:", CAMERA_DEVICE)
        cap.release()
        print("[CAMERA] Falling back to:", FALLBACK_VIDEO)
        cap = cv2.VideoCapture(FALLBACK_VIDEO)
        if not cap.isOpened():
            print("[ERROR] Could not open fallback video")
            camera_running = False
            return
        print("[CAMERA] Using fallback video")
    else:
        print("[CAMERA] Real camera connected")
        stream_ok = True

    print("[CAMERA] Resolution:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
          "x", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    fps_counter = 0
    fps_start = time.time()

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.01)
            continue

        with frame_lock:
            latest_frame = frame
        last_frame_time = time.time()

        fps_counter += 1
        now = time.time()
        if now - fps_start >= 1.0:
            camera_fps = fps_counter / (now - fps_start)
            fps_counter = 0
            fps_start = now

    cap.release()
    print("[CAMERA] Closed")


def get_frame():
    with frame_lock:
        return None if latest_frame is None else latest_frame.copy()


# ============================================================
# FAST LAYER: DRIVABLE-SPACE DETECTION (runs every frame)
# ============================================================

class FreeSpaceDetector:
    """
    Finds the widest obstacle-free horizontal corridor inside the
    dynamic perception window that is at least as wide as the
    calibrated vehicle footprint -- wherever that space happens to be,
    not just "between two painted lines". Runs on every frame; does
    not depend on the VLM at all.
    """

    def __init__(self):
        self.smoothed_target_x = None
        self.smoothed_width = None
        self.alpha = 0.4

    @staticmethod
    def _edge_mask(bgr_roi):
        gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 130)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        return edges

    @staticmethod
    def _segments_in_row(edge_row):
        """Return list of (start_x, end_x) obstacle-free runs in a 1D edge row."""
        occupied = edge_row > 0
        segments = []
        start = None
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

        # Scan several rows: near the vehicle (bottom of ROI, high weight)
        # out to further ahead (lower weight, lets it "see" a turn coming).
        row_fracs = [0.95, 0.85, 0.72, 0.55]
        weights = [0.40, 0.30, 0.20, 0.10]

        candidates = []  # (weight, center_x, width, has_enough_room)
        nearest_width = None

        for frac, weight in zip(row_fracs, weights):
            y = int(np.clip(frac * (h - 1), 0, h - 1))
            row = edges[y, :]
            segments = self._segments_in_row(row)

            if not segments:
                continue

            # pick the segment containing (or nearest to) the reference x
            ref = np.clip(current_x_ref, 0, w - 1)
            best = None
            best_dist = None
            for (s, e) in segments:
                width_px = e - s
                center = (s + e) / 2.0
                if s <= ref <= e:
                    dist = 0
                else:
                    dist = min(abs(ref - s), abs(ref - e))
                if best is None or dist < best_dist or (
                    dist == best_dist and width_px > (best[1] - best[0])
                ):
                    best = (s, e)
                    best_dist = dist

            if best is None:
                continue

            s, e = best
            width_px = e - s
            center = (s + e) / 2.0
            has_room = width_px >= vehicle_width_px * required_margin

            if frac == row_fracs[0]:
                nearest_width = width_px

            candidates.append((weight, center, width_px, has_room))

        if not candidates:
            return {
                "target_x": current_x_ref,
                "corridor_width": 0.0,
                "confidence": 0.0,
                "safe": False,
                "rows_seen": 0,
            }

        # Weighted target center, but throw out rows with no room at all
        # from the average (a far-away pinch point shouldn't drag the
        # target sideways if there's room right where the vehicle is).
        usable = [c for c in candidates if c[3]] or candidates
        total_w = sum(c[0] for c in usable)
        target_x = sum(c[0] * c[1] for c in usable) / total_w
        min_width = min(c[2] for c in candidates)

        confidence = len(usable) / len(row_fracs)
        safe = nearest_width is not None and nearest_width >= vehicle_width_px * required_margin

        # Temporal smoothing so the target doesn't jump frame to frame.
        if self.smoothed_target_x is None:
            self.smoothed_target_x = target_x
        else:
            self.smoothed_target_x = (
                self.alpha * target_x + (1 - self.alpha) * self.smoothed_target_x
            )

        return {
            "target_x": self.smoothed_target_x,
            "corridor_width": min_width,
            "nearest_width": nearest_width if nearest_width is not None else 0.0,
            "confidence": float(confidence),
            "safe": bool(safe),
            "rows_seen": len(candidates),
        }


class FastController:
    """dt-aware, rate-limited controller. Runs every frame, authoritative."""

    def __init__(self):
        self.prev_steering = 0.0
        self.prev_time = None

    def update(self, corridor, current_x_ref, roi_w, vlm_bias):
        now = time.time()
        dt = (now - self.prev_time) if self.prev_time else (1.0 / FPS)
        dt = max(dt, 1e-3)
        self.prev_time = now

        # Normalized error: -1 (target far left) .. +1 (target far right)
        error = (corridor["target_x"] - current_x_ref) / max(roi_w / 2.0, 1.0)
        error = float(np.clip(error, -1.0, 1.0))

        # Advisory VLM bias nudges the error slightly; it cannot create
        # a "safe" reading out of an unsafe one -- that's handled below.
        error = float(np.clip(error + vlm_bias * VLM_BIAS_WEIGHT, -1.0, 1.0))

        raw_steering_deg = error * MAX_STEERING_DEG

        max_delta = STEERING_RATE_LIMIT_DEG
        delta = float(np.clip(raw_steering_deg - self.prev_steering, -max_delta, max_delta))
        steering_deg = float(np.clip(self.prev_steering + delta, -MAX_STEERING_DEG, MAX_STEERING_DEG))
        self.prev_steering = steering_deg

        if not corridor["safe"]:
            throttle = 0.0
            action = "NO SAFE CORRIDOR - STOP"
        else:
            # Slow down as the corridor gets tighter or confidence drops.
            width_ratio = min(1.0, corridor["nearest_width"] / max(1.0, corridor["corridor_width"] * 0 + corridor["nearest_width"]))
            slack = max(0.0, corridor["nearest_width"] - corridor.get("corridor_width", 0))
            throttle = BASE_THROTTLE * corridor["confidence"]
            throttle *= (0.5 + 0.5 * min(1.0, corridor["corridor_width"] / (corridor["nearest_width"] + 1e-6)))
            throttle = max(MIN_THROTTLE_FLOOR, min(BASE_THROTTLE, throttle)) if corridor["confidence"] > 0 else 0.0

            if steering_deg < -4:
                action = "CORRIDOR: STEER LEFT"
            elif steering_deg > 4:
                action = "CORRIDOR: STEER RIGHT"
            else:
                action = "CORRIDOR: STRAIGHT"

        return steering_deg, throttle, action


free_space_detector = FreeSpaceDetector()
fast_controller = FastController()


# ============================================================
# DRIVEFUSION ADVISORY LAYER (slow, does not drive)
# ============================================================

def parse_bias(response):
    """Extract a soft directional bias from the VLM's STEERING field.
    This is advisory only -- see VLM_BIAS_WEIGHT and FastController."""
    text = response.upper()
    if re.search(r"\bLEFT\b", text):
        return -1.0
    if re.search(r"\bRIGHT\b", text):
        return 1.0
    return 0.0


def extract_model_fields(response):
    global road_description, obstacle_description, lane_description
    road = re.search(r"ROAD\s*:\s*(.*)", response, re.IGNORECASE)
    lane = re.search(r"LANE\s*:\s*(.*)", response, re.IGNORECASE)
    obstacle = re.search(r"OBSTACLE\s*:\s*(.*)", response, re.IGNORECASE)
    if road:
        road_description = road.group(1).strip()
    if lane:
        lane_description = lane.group(1).strip()
    if obstacle:
        obstacle_description = obstacle.group(1).strip()


# ============================================================
# TEXT TO SPEECH
# ============================================================

tts_engine = None
tts_lock = threading.Lock()


def initialize_tts():
    global tts_engine
    if not TTS_AVAILABLE:
        print("[VOICE] pyttsx3 not installed")
        return False
    try:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty("rate", VOICE_RATE)
        return True
    except Exception as e:
        print("[VOICE] TTS initialization error:", e)
        return False


def speak(text):
    if not VOICE_ENABLED or tts_engine is None:
        return
    text = re.sub(r"\s+", " ", text.replace("\n", ". "))
    if len(text) > 400:
        text = text[:400] + "."

    def worker():
        with tts_lock:
            try:
                tts_engine.say(text)
                tts_engine.runAndWait()
            except Exception as e:
                print("[VOICE] TTS error:", e)

    threading.Thread(target=worker, daemon=True).start()


# ============================================================
# VOICE LISTENER
# ============================================================

def voice_loop():
    global voice_listening, voice_status, current_prompt

    if not SPEECH_AVAILABLE:
        voice_status = "SPEECH MODULE MISSING"
        return

    try:
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.7
        recognizer.non_speaking_duration = 0.3
        microphone = sr.Microphone(device_index=None)
        print("[VOICE] Microphone ready")
    except Exception as e:
        voice_status = "MIC ERROR"
        print("[VOICE] Microphone initialization failed:", e)
        return

    try:
        with microphone as source:
            print("[VOICE] Calibrating microphone...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
        print("[VOICE] Calibration complete")
    except Exception as e:
        print("[VOICE] Calibration error:", e)

    while voice_running:
        try:
            voice_listening = True
            voice_status = "LISTENING..."
            with microphone as source:
                audio = recognizer.listen(
                    source, timeout=LISTEN_TIMEOUT, phrase_time_limit=PHRASE_TIME_LIMIT
                )
            voice_listening = False
            voice_status = "RECOGNIZING..."
            text = recognizer.recognize_google(audio, language=VOICE_LANGUAGE).strip()

            if text:
                print("\n[VOICE COMMAND]", text)
                voice_status = "COMMAND: " + text[:45]
                with ai_lock:
                    current_prompt = text
                new_prompt_event.set()

        except sr.WaitTimeoutError:
            voice_listening = False
            voice_status = "VOICE READY"
        except sr.UnknownValueError:
            voice_listening = False
            voice_status = "DID NOT UNDERSTAND"
        except Exception as e:
            voice_listening = False
            voice_status = "VOICE ERROR"
            print("[VOICE]", e)
            time.sleep(1)


# ============================================================
# DRIVEFUSION INFERENCE (advisory)
# ============================================================

def run_inference(model, processor, prompt):
    global latest_response, last_inference_time, ai_count, vlm_bias_direction

    frame = get_frame()
    if frame is None:
        return

    with roi_lock:
        x, y, w, h = roi_x, roi_y, roi_w, roi_h

    x1 = max(0, min(WIDTH - 1, x))
    y1 = max(0, min(HEIGHT - 1, y))
    x2 = max(x1 + 1, min(WIDTH, x + w))
    y2 = max(y1 + 1, min(HEIGHT, y + h))

    roi_frame = frame[y1:y2, x1:x2]
    if roi_frame.size == 0:
        return

    rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    with vehicle_lock:
        vw, vl = vehicle_width, vehicle_length

    instruction = f"""
You are the ADVISORY perception layer of KF-Intelligent Driving.
A separate fast, per-frame controller handles actual steering safety;
you provide scene understanding, hazard description, and a soft
directional preference only -- you do not have final authority over
the vehicle's motion.

USER COMMAND:
{prompt}

Use the supplied perception window as the primary driving region.
Vehicle calibration: width = {vw}px, length = {vl}px.

Return exactly this structure:

ROAD:
LANE:
OBSTACLE:
TARGET:
ACTION:
STEERING:
STEERING_DETAIL:
SPEED:
HAZARD:

STEERING must be one of: LEFT, RIGHT, STRAIGHT.
Keep STEERING_DETAIL short and precise.
"""

    message = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": instruction},
        ]}
    ]

    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")

    for key, value in inputs.items():
        if torch.is_tensor(value):
            inputs[key] = value.to("cuda", non_blocking=True)

    start = time.time()
    with torch.inference_mode():
        result = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    elapsed = time.time() - start

    last_inference_time = elapsed
    ai_count += 1

    tokens = result[0] if isinstance(result, tuple) else result
    input_length = inputs["input_ids"].shape[1]
    generated = tokens[:, input_length:]
    decoded = processor.batch_decode(generated, skip_special_tokens=True)
    response = decoded[0].strip() if decoded else "No response."

    with ai_lock:
        latest_response = response

    vlm_bias_direction = parse_bias(response)
    extract_model_fields(response)

    print(f"\n[AI #{ai_count}] {elapsed:.2f}s  bias={vlm_bias_direction:+.1f}")
    print(response)

    speak(build_voice_response(response))


def build_voice_response(response):
    action = re.search(r"ACTION\s*:\s*(.*)", response, re.IGNORECASE)
    detail = re.search(r"STEERING_DETAIL\s*:\s*(.*)", response, re.IGNORECASE)
    obstacle = re.search(r"OBSTACLE\s*:\s*(.*)", response, re.IGNORECASE)

    action_text = action.group(1).strip() if action else "no action specified"
    detail_text = detail.group(1).strip() if detail else ""
    obstacle_text = obstacle.group(1).strip() if obstacle else "no obstacle information"

    with fast_lock:
        safe_state = fast_action

    spoken = f"I see {obstacle_text}. Advisory action is {action_text}. Actual control state: {safe_state}."
    if detail_text:
        spoken += f" {detail_text}"
    return spoken


def ai_loop(model, processor):
    global ai_running
    print("\n[AI] Advisory thread started")
    last_run = 0.0

    while ai_running:
        event_received = new_prompt_event.wait(timeout=AI_INTERVAL)
        if not ai_running:
            break
        if ai_paused:
            new_prompt_event.clear()
            time.sleep(0.05)
            continue

        now = time.time()
        if not event_received and now - last_run < AI_INTERVAL:
            continue

        new_prompt_event.clear()
        with ai_lock:
            prompt = current_prompt

        run_inference(model, processor, prompt)
        last_run = time.time()


# ============================================================
# DRAWING
# ============================================================

def draw_vehicle_body(display):
    with vehicle_lock:
        cx = vehicle_center_x + vehicle_offset_x
        cy = vehicle_bottom_y + vehicle_offset_y
        length = int(vehicle_length * vehicle_scale)
        width = int(vehicle_width * vehicle_scale)

    pts = np.array([
        (int(cx - width / 2), int(cy)),
        (int(cx + width / 2), int(cy)),
        (int(cx + width * 0.38), int(cy - length)),
        (int(cx - width * 0.38), int(cy - length)),
    ], dtype=np.int32)

    cv2.fillPoly(display, [pts], (35, 35, 45))
    cv2.polylines(display, [pts], True, (0, 220, 255), 3)
    cv2.line(display, (int(cx), int(cy - length)), (int(cx), int(cy)), (0, 255, 0), 2)
    cv2.circle(display, (int(cx), int(cy - length)), 7, (0, 255, 255), -1)


def draw_perception_window(display):
    with roi_lock:
        x, y, w, h = roi_x, roi_y, roi_w, roi_h

    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 3)
    center = x + w // 2
    cv2.line(display, (center, y), (center, y + h), (0, 255, 0), 2)

    cv2.rectangle(display, (x, y - 30), (x + 310, y), (0, 0, 0), -1)
    cv2.putText(display, "DYNAMIC PERCEPTION WINDOW", (x + 8, y - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)


def draw_corridor(display):
    """Visualize the fast layer's chosen drivable corridor and vehicle-fit check."""
    with roi_lock:
        x, y, w, h = roi_x, roi_y, roi_w, roi_h
    with fast_lock:
        corridor = fast_corridor
        safe = corridor["safe"] if corridor else False

    if corridor is None:
        return

    target_x_full = int(x + corridor["target_x"])
    near_y = int(y + h * 0.95)
    far_y = int(y + h * 0.55)

    color = (0, 255, 0) if safe else (0, 0, 255)

    with vehicle_lock:
        vw = int(vehicle_width * vehicle_scale)

    cv2.line(display, (target_x_full, near_y), (target_x_full, far_y), color, 3)
    cv2.rectangle(
        display,
        (target_x_full - vw // 2, near_y - 6),
        (target_x_full + vw // 2, near_y + 6),
        color, 2,
    )
    label = "CORRIDOR OK" if safe else "CORRIDOR TOO NARROW"
    cv2.putText(display, label, (target_x_full - 70, far_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)


def draw_steering_indicator(display, steering_deg):
    cx = WIDTH // 2
    base_y = HEIGHT - 220

    cv2.line(display, (cx, base_y), (cx, base_y - 100), (100, 100, 100), 2)
    end_x = int(cx + steering_deg * 2.5)
    end_y = base_y - 90

    if steering_deg < -2:
        color = (0, 120, 255)
    elif steering_deg > 2:
        color = (255, 150, 0)
    else:
        color = (0, 255, 0)

    cv2.arrowedLine(display, (cx, base_y), (end_x, end_y), color, 5, tipLength=0.2)


def draw_hud(display):
    cv2.rectangle(display, (0, 0), (WIDTH, 75), (8, 10, 15), -1)
    cv2.putText(display, "KF-INTELLIGENT DRIVING v2", (20, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 120), 2)
    cv2.putText(display, "FAST CORRIDOR CONTROL + DRIVEFUSION ADVISORY", (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(display, f"CAM {camera_fps:.1f} FPS", (1060, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    status = "AI PAUSED" if ai_paused else "AI ADVISORY ACTIVE"
    color = (0, 160, 255) if ai_paused else (0, 255, 0)
    cv2.putText(display, status, (1060, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)


def draw_model_panel(display):
    panel_top = HEIGHT - 175
    overlay = display.copy()
    cv2.rectangle(overlay, (0, panel_top), (WIDTH, HEIGHT), (5, 7, 10), -1)
    display[:] = cv2.addWeighted(overlay, 0.90, display, 0.10, 0)

    with ai_lock:
        prompt = current_prompt
        response = latest_response

    with fast_lock:
        steering_deg = fast_steering_deg
        throttle = fast_throttle
        action = fast_action

    cv2.putText(display, "USER COMMAND:", (15, panel_top + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 220, 255), 1)
    cv2.putText(display, prompt[:100], (150, panel_top + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    cv2.putText(display, "ADVISORY:", (15, panel_top + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    lines = textwrap.wrap(response.replace("\n", " "), width=150)
    y = panel_top + 70
    for line in lines[:2]:
        cv2.putText(display, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1)
        y += 19

    color = (0, 0, 255) if "STOP" in action else ((0, 140, 255) if "LEFT" in action else
             ((255, 170, 0) if "RIGHT" in action else (0, 255, 0)))

    cv2.putText(display, f"CONTROL: {action}", (15, HEIGHT - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
    cv2.putText(display, f"STEER {steering_deg:+.1f} deg   THROTTLE {throttle:.2f}",
                (15, HEIGHT - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(display, f"AI {last_inference_time:.2f}s", (WIDTH - 120, HEIGHT - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)


# ============================================================
# GUI
# ============================================================

class ControlWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("KF-Intelligent Driving Control")
        self.root.geometry("650x620")
        self.root.configure(bg="#101318")

        tk.Label(self.root, text="KF-INTELLIGENT DRIVING", font=("Arial", 18, "bold"),
                 fg="#00ff88", bg="#101318").pack(pady=(12, 3))
        tk.Label(self.root, text="Fast corridor control + DriveFusion advisory",
                 font=("Arial", 10), fg="#bbbbbb", bg="#101318").pack(pady=(0, 12))

        tk.Label(self.root, text="ADVISORY COMMAND", fg="#00ffcc", bg="#101318",
                 font=("Arial", 10, "bold")).pack(anchor="w", padx=20)

        self.entry = tk.Entry(self.root, font=("Arial", 13), bg="#20252c", fg="white",
                               insertbackground="white")
        self.entry.pack(fill="x", padx=20, pady=7)
        self.entry.insert(0, current_prompt)
        self.entry.bind("<Return>", self.submit)

        frame = tk.Frame(self.root, bg="#101318")
        frame.pack(pady=5)

        tk.Button(frame, text="SEND", command=self.submit, bg="#008f5a", fg="white",
                  width=12, font=("Arial", 10, "bold")).pack(side="left", padx=4)
        self.pause_button = tk.Button(frame, text="PAUSE AI", command=self.toggle_pause,
                                       bg="#a06000", fg="white", width=12, font=("Arial", 10, "bold"))
        self.pause_button.pack(side="left", padx=4)
        tk.Button(frame, text="VOICE", command=self.voice_test, bg="#006fa6", fg="white",
                  width=12, font=("Arial", 10, "bold")).pack(side="left", padx=4)
        tk.Button(frame, text="QUIT", command=self.quit, bg="#9d2020", fg="white",
                  width=10, font=("Arial", 10, "bold")).pack(side="left", padx=4)

        tk.Label(self.root, text="PERCEPTION WINDOW CALIBRATION", fg="#00ffff", bg="#101318",
                 font=("Arial", 11, "bold")).pack(anchor="w", padx=20, pady=(15, 5))
        self.roi_x_scale = self.make_scale("ROI X", 0, WIDTH, roi_x, self.update_roi)
        self.roi_y_scale = self.make_scale("ROI Y", 0, HEIGHT, roi_y, self.update_roi)
        self.roi_w_scale = self.make_scale("ROI WIDTH", 100, WIDTH, roi_w, self.update_roi)
        self.roi_h_scale = self.make_scale("ROI HEIGHT", 100, HEIGHT, roi_h, self.update_roi)

        tk.Label(self.root, text="VEHICLE BODY CALIBRATION", fg="#ffaa00", bg="#101318",
                 font=("Arial", 11, "bold")).pack(anchor="w", padx=20, pady=(12, 5))
        self.vehicle_width_scale = self.make_scale("BODY WIDTH", 30, 300, vehicle_width, self.update_vehicle)
        self.vehicle_length_scale = self.make_scale("BODY LENGTH", 50, 350, vehicle_length, self.update_vehicle)
        self.vehicle_x_scale = self.make_scale("BODY X OFFSET", -300, 300, 0, self.update_vehicle)
        self.vehicle_y_scale = self.make_scale("BODY Y OFFSET", -150, 150, 0, self.update_vehicle)

        self.status_label = tk.Label(self.root, text="VOICE: READY", fg="#00ff88",
                                      bg="#101318", font=("Arial", 10, "bold"))
        self.status_label.pack(pady=10)
        self.update_status()

    def make_scale(self, label, minimum, maximum, value, command):
        container = tk.Frame(self.root, bg="#101318")
        container.pack(fill="x", padx=20)
        tk.Label(container, text=label, width=16, anchor="w", fg="#cccccc",
                 bg="#101318").pack(side="left")
        scale = tk.Scale(container, from_=minimum, to=maximum, orient="horizontal",
                          resolution=1, bg="#101318", fg="white", troughcolor="#303640",
                          highlightthickness=0, command=command)
        scale.set(value)
        scale.pack(side="left", fill="x", expand=True)
        return scale

    def update_roi(self, value=None):
        global roi_x, roi_y, roi_w, roi_h
        with roi_lock:
            roi_x = int(self.roi_x_scale.get())
            roi_y = int(self.roi_y_scale.get())
            roi_w = int(self.roi_w_scale.get())
            roi_h = int(self.roi_h_scale.get())

    def update_vehicle(self, value=None):
        global vehicle_width, vehicle_length, vehicle_offset_x, vehicle_offset_y
        with vehicle_lock:
            vehicle_width = int(self.vehicle_width_scale.get())
            vehicle_length = int(self.vehicle_length_scale.get())
            vehicle_offset_x = int(self.vehicle_x_scale.get())
            vehicle_offset_y = int(self.vehicle_y_scale.get())

    def submit(self, event=None):
        global current_prompt
        command = self.entry.get().strip()
        if not command:
            return
        with ai_lock:
            current_prompt = command
        print("\n[USER COMMAND]", command)
        new_prompt_event.set()
        speak("Command received.")

    def toggle_pause(self):
        global ai_paused
        ai_paused = not ai_paused
        if ai_paused:
            self.pause_button.config(text="RESUME AI")
            speak("Advisory layer paused. Fast corridor control continues.")
        else:
            self.pause_button.config(text="PAUSE AI")
            speak("Advisory layer resumed.")

    def voice_test(self):
        speak("KF-Intelligent Driving voice interface is ready.")

    def update_status(self):
        with fast_lock:
            action = fast_action
        self.status_label.config(text=f"VOICE: {voice_status}   |   CONTROL: {action}")
        self.root.after(250, self.update_status)

    def quit(self):
        global camera_running, ai_running, voice_running
        camera_running = False
        ai_running = False
        voice_running = False
        self.root.destroy()


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("KF-INTELLIGENT DRIVING v2")
print("Fast corridor control (every frame) + DriveFusion advisory (slow)")
print("=" * 80)

print("\n[1] Loading DriveFusion...")
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("[WARNING] CUDA not available -- running on CPU. Advisory layer will be slow;")
    print("          this does not affect the fast corridor controller.")

model = DriveFusionForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)
if device == "cpu":
    model = model.to(device)
model.eval()
print("[OK] DriveFusion loaded on", device)

print("\n[2] Loading processor...")
processor = DriveFusionProcessor.from_pretrained(MODEL_ID)
print("[OK] Processor loaded")

print("\n[3] Initializing voice...")
if initialize_tts():
    print("[VOICE] Text-to-speech ready")
else:
    print("[VOICE] Text-to-speech unavailable")

camera_thread = threading.Thread(target=camera_loop, daemon=True)
camera_thread.start()

ai_thread = threading.Thread(target=ai_loop, args=(model, processor), daemon=True)
ai_thread.start()

if SPEECH_AVAILABLE:
    voice_thread = threading.Thread(target=voice_loop, daemon=True)
    voice_thread.start()
else:
    print("[VOICE] Install speech_recognition first")

gui = ControlWindow()

print("\n[DISPLAY] Starting cockpit")

while camera_running:
    frame = get_frame()
    if frame is None:
        time.sleep(0.01)
        try:
            gui.root.update()
        except Exception:
            pass
        continue

    # --------------------------------------------------------
    # FAST LAYER: runs every frame, no VLM dependency.
    # --------------------------------------------------------
    with roi_lock:
        rx, ry, rw, rh = roi_x, roi_y, roi_w, roi_h

    x1 = max(0, min(WIDTH - 1, rx))
    y1 = max(0, min(HEIGHT - 1, ry))
    x2 = max(x1 + 1, min(WIDTH, rx + rw))
    y2 = max(y1 + 1, min(HEIGHT, ry + rh))
    roi_frame = frame[y1:y2, x1:x2]

    with vehicle_lock:
        vw = vehicle_width
        veh_cx = vehicle_center_x + vehicle_offset_x

    current_x_ref = np.clip(veh_cx - x1, 0, (x2 - x1) - 1)

    if roi_frame.size > 0:
        corridor = free_space_detector.analyze(
            roi_frame, vehicle_width_px=vw,
            current_x_ref=current_x_ref, required_margin=SAFETY_WIDTH_MARGIN,
        )
        steering_deg, throttle, action = fast_controller.update(
            corridor, current_x_ref, x2 - x1, vlm_bias_direction
        )

        with fast_lock:
            fast_steering_deg = steering_deg
            fast_throttle = throttle
            fast_action = action
            fast_corridor = corridor

    # --------------------------------------------------------
    # SEND TO JETRACER (AUTO uses KF output; MANUAL/STOP ignore it)
    # --------------------------------------------------------
    with fast_lock:
        _s_deg, _thr = fast_steering_deg, fast_throttle
    _fresh = stream_ok and (time.time() - last_frame_time) < 0.5
    link.update(_s_deg, _thr, frame_ok=_fresh, frame_time=last_frame_time)

    # --------------------------------------------------------
    # DRAW
    # --------------------------------------------------------
    display = frame.copy()
    draw_hud(display)
    draw_perception_window(display)
    draw_corridor(display)
    draw_vehicle_body(display)
    with fast_lock:
        draw_steering_indicator(display, fast_steering_deg)
    draw_model_panel(display)
    link.draw_status(display)

    cv2.imshow("KF-Intelligent Driving v2", display)

    key = cv2.waitKey(1) & 0xFF
    link.handle_key(key)
    if key == ord("q"):
        camera_running = False
        ai_running = False
        voice_running = False
        break

    try:
        gui.root.update()
    except tk.TclError:
        break

# ============================================================
# SHUTDOWN
# ============================================================

camera_running = False
ai_running = False
voice_running = False
new_prompt_event.set()

link.stop()   # tell the JetRacer to stop

camera_thread.join(timeout=2)
ai_thread.join(timeout=2)
cv2.destroyAllWindows()

try:
    gui.root.destroy()
except Exception:
    pass

print("\n" + "=" * 80)
print("KF-INTELLIGENT DRIVING SHUTDOWN")
print("=" * 80)
print("Advisory inferences:", ai_count)
