#!/usr/bin/env python3
"""
YOLO + JetRacer Real-Time Test
==============================

Pipeline:

Camera
   |
   v
YOLO Detection
   |
   v
Lane / Object Analysis
   |
   v
Steering Calculation
   |
   v
JetRacer Control

IMPORTANT:
The program starts in SAFE_MODE, meaning the physical
JetRacer motors are disabled.

Press:
    S = Toggle SAFE MODE
    R = Stop the JetRacer immediately
    Q = Quit

Change MODEL_PATH and CAMERA_DEVICE below.
"""

import time
import threading
import cv2
import numpy as np

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

# Path to your downloaded YOLO model
MODEL_PATH = "best.pt"

# Camera device
CAMERA_DEVICE = "/dev/video48"

# Camera resolution
WIDTH = 640
HEIGHT = 480

# Camera FPS
FPS = 30


# ============================================================
# JETRACER SETTINGS
# ============================================================

# IMPORTANT:
# Start with True!
SAFE_MODE = True

# Maximum allowed steering
MAX_STEERING = 0.35

# Normal driving speed
BASE_THROTTLE = 0.18

# Minimum throttle
MIN_THROTTLE = 0.0

# Maximum throttle
MAX_THROTTLE = 0.25

# Steering gain
STEERING_GAIN = 0.8

# Detection confidence
CONFIDENCE_THRESHOLD = 0.4

# Stop distance proxy.
# For normal object detection, this is based on bounding box size.
OBSTACLE_AREA_THRESHOLD = 50000


# ============================================================
# GLOBAL VARIABLES
# ============================================================

running = True

latest_frame = None
frame_lock = threading.Lock()

camera_fps = 0.0
inference_fps = 0.0

current_steering = 0.0
current_throttle = 0.0

safe_mode = SAFE_MODE

model = None
class_names = {}


# ============================================================
# JETRACER INITIALIZATION
# ============================================================

try:
    from jetracer.nvidia_racecar import NvidiaRacecar

    car = NvidiaRacecar()

    car.steering = 0.0
    car.throttle = 0.0

    JETRACER_AVAILABLE = True

    print("[JETRACER] JetRacer successfully initialized")

except Exception as e:

    car = None
    JETRACER_AVAILABLE = False

    print("[WARNING] Could not initialize JetRacer")
    print("[WARNING]", e)
    print("[WARNING] Running in camera/visualization mode")


# ============================================================
# LOAD YOLO MODEL
# ============================================================

def load_model():

    global model
    global class_names

    print()
    print("=" * 60)
    print("LOADING YOLO MODEL")
    print("=" * 60)

    print(f"Model path: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)

    class_names = model.names

    print()
    print("YOLO MODEL LOADED SUCCESSFULLY")
    print()
    print("MODEL CLASSES:")

    for class_id, class_name in class_names.items():

        print(f"  {class_id}: {class_name}")

    print("=" * 60)
    print()


# ============================================================
# CAMERA INITIALIZATION
# ============================================================

def open_camera():

    print()
    print("=" * 60)
    print("OPENING CAMERA")
    print("=" * 60)

    # Try V4L2 backend
    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)

    if not cap.isOpened():

        print("[WARNING] V4L2 camera failed")

        cap = cv2.VideoCapture(CAMERA_DEVICE)

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open camera: {CAMERA_DEVICE}"
        )

    # Camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # Try MJPEG
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *"MJPG"
        )
    )

    actual_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    actual_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    actual_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    print(
        f"Resolution: "
        f"{actual_width} x {actual_height}"
    )

    print(
        f"Camera FPS: "
        f"{actual_fps}"
    )

    print("=" * 60)
    print()

    return cap


# ============================================================
# CAMERA THREAD
# ============================================================

def camera_loop(cap):

    global latest_frame
    global camera_fps
    global running

    frame_counter = 0
    start_time = time.time()

    print("[CAMERA] Camera thread started")

    while running:

        success, frame = cap.read()

        if not success:

            print("[CAMERA] Failed to read frame")

            time.sleep(0.01)

            continue

        # Store newest frame only
        with frame_lock:

            latest_frame = frame.copy()

        frame_counter += 1

        elapsed = time.time() - start_time

        if elapsed >= 1.0:

            camera_fps = (
                frame_counter / elapsed
            )

            frame_counter = 0

            start_time = time.time()


# ============================================================
# HELPER FUNCTION:
# NORMALIZE YOLO CLASS NAME
# ============================================================

def normalize_class_name(name):

    return (
        str(name)
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


# ============================================================
# FIND LANE OBJECTS
# ============================================================

def identify_lane_objects(detections):

    """
    Attempts to identify:
        left lane
        right lane
        center line

    This works only if your YOLO model has
    lane-related classes.

    You may need to modify the keywords
    after seeing your model's class names.
    """

    left_lane = None
    right_lane = None
    center_lane = None

    for detection in detections:

        class_name = normalize_class_name(
            detection["class_name"]
        )

        # ------------------------------------------------
        # LEFT LANE
        # ------------------------------------------------

        if (
            "leftlane" in class_name
            or "leftline" in class_name
        ):

            left_lane = detection

        # ------------------------------------------------
        # RIGHT LANE
        # ------------------------------------------------

        elif (
            "rightlane" in class_name
            or "rightline" in class_name
        ):

            right_lane = detection

        # ------------------------------------------------
        # CENTER LANE
        # ------------------------------------------------

        elif (
            "centerlane" in class_name
            or "centreline" in class_name
            or "centerline" in class_name
            or "middleline" in class_name
        ):

            center_lane = detection

    return (
        left_lane,
        right_lane,
        center_lane
    )


# ============================================================
# CALCULATE LANE CENTER
# ============================================================

def calculate_lane_center(
    left_lane,
    right_lane,
    center_lane,
    image_width
):

    """
    Returns:

        lane_center_x

    Priority:

        1. Center lane
        2. Average left and right lanes
        3. Single lane estimation
        4. None
    """

    # ------------------------------------------------
    # CENTER LINE AVAILABLE
    # ------------------------------------------------

    if center_lane is not None:

        x1, y1, x2, y2 = (
            center_lane["bbox"]
        )

        return (
            (x1 + x2) / 2.0
        )

    # ------------------------------------------------
    # LEFT AND RIGHT LANES AVAILABLE
    # ------------------------------------------------

    if (
        left_lane is not None
        and right_lane is not None
    ):

        lx1, ly1, lx2, ly2 = (
            left_lane["bbox"]
        )

        rx1, ry1, rx2, ry2 = (
            right_lane["bbox"]
        )

        left_x = (
            lx1 + lx2
        ) / 2.0

        right_x = (
            rx1 + rx2
        ) / 2.0

        return (
            left_x + right_x
        ) / 2.0

    # ------------------------------------------------
    # ONLY LEFT LANE
    # ------------------------------------------------

    if left_lane is not None:

        lx1, ly1, lx2, ly2 = (
            left_lane["bbox"]
        )

        left_x = (
            lx1 + lx2
        ) / 2.0

        # Estimate road center
        return (
            left_x +
            image_width * 0.25
        )

    # ------------------------------------------------
    # ONLY RIGHT LANE
    # ------------------------------------------------

    if right_lane is not None:

        rx1, ry1, rx2, ry2 = (
            right_lane["bbox"]
        )

        right_x = (
            rx1 + rx2
        ) / 2.0

        # Estimate road center
        return (
            right_x -
            image_width * 0.25
        )

    return None


# ============================================================
# OBSTACLE DETECTION
# ============================================================

def detect_obstacle(detections):

    """
    Basic obstacle detector.

    Large bounding boxes in the lower
    part of the image are treated as
    potentially dangerous.

    You should later customize this
    based on your YOLO classes.
    """

    obstacle_detected = False

    obstacle_names = [

        "person",
        "car",
        "truck",
        "bus",
        "bicycle",
        "motorcycle",

        "obstacle",

        "pedestrian",

        "vehicle"
    ]

    for detection in detections:

        class_name = normalize_class_name(
            detection["class_name"]
        )

        x1, y1, x2, y2 = (
            detection["bbox"]
        )

        box_width = x2 - x1
        box_height = y2 - y1

        area = (
            box_width *
            box_height
        )

        # Check known obstacle classes
        for obstacle_name in obstacle_names:

            if obstacle_name in class_name:

                # Large enough to be relevant
                if (
                    area >
                    OBSTACLE_AREA_THRESHOLD
                ):

                    obstacle_detected = True

                    return True

    return obstacle_detected


# ============================================================
# CALCULATE STEERING
# ============================================================

def calculate_steering(
    lane_center_x,
    image_width
):

    """
    Convert visual lane error
    into JetRacer steering.

    JetRacer steering range:

        -1.0  to  +1.0

    We restrict it for safety.
    """

    if lane_center_x is None:

        return 0.0

    image_center = (
        image_width / 2.0
    )

    # Normalized error
    error = (
        lane_center_x -
        image_center
    ) / image_center

    # Steering direction
    steering = (
        STEERING_GAIN *
        error
    )

    # Limit steering
    steering = np.clip(
        steering,
        -MAX_STEERING,
        MAX_STEERING
    )

    return float(
        steering
    )


# ============================================================
# CALCULATE THROTTLE
# ============================================================

def calculate_throttle(
    steering,
    obstacle
):

    """
    Reduce speed when turning.

    Stop when obstacle detected.
    """

    if obstacle:

        return 0.0

    # Slow down during sharp turns
    steering_factor = (
        1.0 -
        abs(steering) /
        MAX_STEERING
    )

    throttle = (
        MIN_THROTTLE +
        (
            BASE_THROTTLE -
            MIN_THROTTLE
        ) *
        steering_factor
    )

    throttle = np.clip(
        throttle,
        MIN_THROTTLE,
        MAX_THROTTLE
    )

    return float(
        throttle
    )


# ============================================================
# JETRACER CONTROL
# ============================================================

def control_jetracer(
    steering,
    throttle
):

    global current_steering
    global current_throttle
    global safe_mode

    current_steering = steering
    current_throttle = throttle

    # -----------------------------------------------
    # SAFE MODE
    # -----------------------------------------------

    if safe_mode:

        if JETRACER_AVAILABLE:

            car.steering = 0.0
            car.throttle = 0.0

        return

    # -----------------------------------------------
    # PHYSICAL CONTROL
    # -----------------------------------------------

    if JETRACER_AVAILABLE:

        try:

            car.steering = steering

            car.throttle = throttle

        except Exception as e:

            print(
                "[JETRACER CONTROL ERROR]",
                e
            )


# ============================================================
# EMERGENCY STOP
# ============================================================

def emergency_stop():

    global current_steering
    global current_throttle

    current_steering = 0.0
    current_throttle = 0.0

    if JETRACER_AVAILABLE:

        try:

            car.steering = 0.0
            car.throttle = 0.0

            print(
                "[EMERGENCY STOP] "
                "JetRacer stopped"
            )

        except Exception as e:

            print(
                "[STOP ERROR]",
                e
            )


# ============================================================
# DRAW INFORMATION
# ============================================================

def draw_information(
    frame,
    detections,
    lane_center_x,
    steering,
    throttle,
    obstacle
):

    height, width = frame.shape[:2]

    # ------------------------------------------------
    # IMAGE CENTER
    # ------------------------------------------------

    image_center_x = (
        width // 2
    )

    cv2.line(
        frame,
        (
            image_center_x,
            0
        ),
        (
            image_center_x,
            height
        ),
        (255, 0, 0),
        2
    )

    # ------------------------------------------------
    # DETECTED LANE CENTER
    # ------------------------------------------------

    if lane_center_x is not None:

        lane_center_x_int = int(
            lane_center_x
        )

        cv2.line(
            frame,
            (
                lane_center_x_int,
                0
            ),
            (
                lane_center_x_int,
                height
            ),
            (0, 255, 0),
            3
        )

    # ------------------------------------------------
    # YOLO DETECTIONS
    # ------------------------------------------------

    for detection in detections:

        x1, y1, x2, y2 = (
            detection["bbox"]
        )

        class_name = (
            detection["class_name"]
        )

        confidence = (
            detection["confidence"]
        )

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2
        )

        label = (
            f"{class_name} "
            f"{confidence:.2f}"
        )

        cv2.putText(
            frame,
            label,
            (
                x1,
                max(25, y1 - 10)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    # ------------------------------------------------
    # STATUS TEXT
    # ------------------------------------------------

    mode_text = (
        "SAFE MODE"
        if safe_mode
        else "DRIVING MODE"
    )

    mode_color = (
        (0, 0, 255)
        if safe_mode
        else (0, 255, 0)
    )

    status_lines = [

        f"MODE: {mode_text}",

        f"CAMERA FPS: {camera_fps:.1f}",

        f"YOLO FPS: {inference_fps:.1f}",

        f"STEERING: {steering:.3f}",

        f"THROTTLE: {throttle:.3f}",

        f"OBSTACLE: {obstacle}",

        f"DETECTIONS: {len(detections)}"
    ]

    y_position = 30

    for index, text in enumerate(
        status_lines
    ):

        color = (
            mode_color
            if index == 0
            else (255, 255, 255)
        )

        cv2.putText(
            frame,
            text,
            (10, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

        y_position += 30

    # ------------------------------------------------
    # CONTROL BAR
    # ------------------------------------------------

    cv2.putText(
        frame,
        "Q: Quit | "
        "S: Safe/Drive | "
        "R: Emergency Stop",
        (
            10,
            height - 20
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    return frame


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    global running
    global safe_mode
    global inference_fps

    print()
    print("#" * 70)
    print("#       YOLO + JETRACER REAL-TIME CONTROL SYSTEM       #")
    print("#" * 70)
    print()

    # ------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------

    load_model()

    # ------------------------------------------------
    # OPEN CAMERA
    # ------------------------------------------------

    cap = open_camera()

    # ------------------------------------------------
    # START CAMERA THREAD
    # ------------------------------------------------

    camera_thread = threading.Thread(
        target=camera_loop,
        args=(cap,),
        daemon=True
    )

    camera_thread.start()

    # ------------------------------------------------
    # WAIT FOR FIRST FRAME
    # ------------------------------------------------

    print(
        "[SYSTEM] Waiting for camera..."
    )

    while latest_frame is None:

        time.sleep(0.1)

    print(
        "[SYSTEM] Camera ready"
    )

    print()
    print("KEYBOARD CONTROLS")
    print("S -> Toggle Safe Mode")
    print("R -> Emergency Stop")
    print("Q -> Quit")
    print()

    inference_counter = 0
    inference_start_time = time.time()

    # ========================================================
    # MAIN LOOP
    # ========================================================

    try:

        while running:

            # --------------------------------------------
            # GET NEWEST FRAME
            # --------------------------------------------

            with frame_lock:

                if latest_frame is None:

                    continue

                frame = latest_frame.copy()

            # --------------------------------------------
            # YOLO INFERENCE
            # --------------------------------------------

            results = model(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False
            )

            # --------------------------------------------
            # PROCESS DETECTIONS
            # --------------------------------------------

            detections = []

            for result in results:

                if result.boxes is None:

                    continue

                for box in result.boxes:

                    x1, y1, x2, y2 = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                    )

                    confidence = float(
                        box.conf[0]
                    )

                    class_id = int(
                        box.cls[0]
                    )

                    class_name = (
                        class_names[class_id]
                    )

                    detection = {

                        "bbox": (
                            x1,
                            y1,
                            x2,
                            y2
                        ),

                        "confidence":
                            confidence,

                        "class_id":
                            class_id,

                        "class_name":
                            class_name
                    }

                    detections.append(
                        detection
                    )

            # --------------------------------------------
            # IDENTIFY LANES
            # --------------------------------------------

            (
                left_lane,
                right_lane,
                center_lane

            ) = identify_lane_objects(
                detections
            )

            # --------------------------------------------
            # CALCULATE LANE CENTER
            # --------------------------------------------

            lane_center_x = (
                calculate_lane_center(

                    left_lane,
                    right_lane,
                    center_lane,

                    frame.shape[1]

                )
            )

            # --------------------------------------------
            # OBSTACLE DETECTION
            # --------------------------------------------

            obstacle = (
                detect_obstacle(
                    detections
                )
            )

            # --------------------------------------------
            # CONTROL CALCULATION
            # --------------------------------------------

            steering = (
                calculate_steering(

                    lane_center_x,

                    frame.shape[1]

                )
            )

            throttle = (
                calculate_throttle(

                    steering,

                    obstacle

                )
            )

            # --------------------------------------------
            # FAILSAFE:
            # NO LANE = STOP
            # --------------------------------------------

            if lane_center_x is None:

                throttle = 0.0

            # --------------------------------------------
            # SEND CONTROL
            # --------------------------------------------

            control_jetracer(

                steering,

                throttle

            )

            # --------------------------------------------
            # INFERENCE FPS
            # --------------------------------------------

            inference_counter += 1

            elapsed = (
                time.time() -
                inference_start_time
            )

            if elapsed >= 1.0:

                inference_fps = (
                    inference_counter /
                    elapsed
                )

                inference_counter = 0

                inference_start_time = time.time()

            # --------------------------------------------
            # VISUALIZATION
            # --------------------------------------------

            display_frame = (
                draw_information(

                    frame,

                    detections,

                    lane_center_x,

                    steering,

                    throttle,

                    obstacle

                )
            )

            cv2.imshow(

                "YOLO JetRacer Control",

                display_frame

            )

            # --------------------------------------------
            # KEYBOARD INPUT
            # --------------------------------------------

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # Quit
            if key == ord("q"):

                print(
                    "[SYSTEM] Quit requested"
                )

                running = False

            # Toggle safe mode
            elif key == ord("s"):

                safe_mode = (
                    not safe_mode
                )

                if safe_mode:

                    emergency_stop()

                    print(
                        "[SYSTEM] SAFE MODE ENABLED"
                    )

                else:

                    print()
                    print(
                        "[WARNING] DRIVING MODE ENABLED"
                    )
                    print(
                        "[WARNING] "
                        "JetRacer can now move!"
                    )
                    print()

            # Emergency stop
            elif key == ord("r"):

                emergency_stop()

    except KeyboardInterrupt:

        print()
        print(
            "[SYSTEM] Keyboard interrupt"
        )

    except Exception as e:

        print()
        print(
            "[SYSTEM ERROR]"
        )

        print(e)

    finally:

        # ====================================================
        # CLEANUP
        # ====================================================

        print()
        print(
            "[SYSTEM] Stopping JetRacer..."
        )

        running = False

        emergency_stop()

        cap.release()

        cv2.destroyAllWindows()

        print(
            "[SYSTEM] Shutdown complete"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()