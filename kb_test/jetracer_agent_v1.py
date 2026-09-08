#!/usr/bin/env python3

"""
============================================================
JETRACER VISION AGENT V1
============================================================

MISSION:

    FOLLOW LINE
        ↓
    DETECT POINT B
        ↓
    STOP

PERCEPTION:
    - USB Camera
    - YOLO Object Detection
    - Lane Detection
    - ArUco Point B Detection

AGENT:

    OBSERVE → THINK → DECIDE

IMPORTANT:

SAFE_MODE = True

The JetRacer motors will NOT move.
============================================================
"""

import time
import threading

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_DEVICE = "/dev/video48"

WIDTH = 640
HEIGHT = 480
FPS = 30

# Use your YOLO model here
MODEL_PATH = "yolov8n.pt"

YOLO_CONFIDENCE = 0.40

# Run YOLO less frequently than camera
YOLO_INTERVAL = 0.30


# ============================================================
# POINT B CONFIGURATION
# ============================================================

POINT_B_ID = 42

# Marker area at which we consider Point B reached
POINT_B_STOP_AREA = 50000


# ============================================================
# CONTROL CONFIGURATION
# ============================================================

SAFE_MODE = True

MAX_STEERING = 0.35
BASE_THROTTLE = 0.15


# ============================================================
# MISSION
# ============================================================

MISSION = "FOLLOW_LINE_TO_POINT_B"


# ============================================================
# GLOBAL CAMERA STATE
# ============================================================

latest_frame = None
frame_lock = threading.Lock()

running = True

camera_fps = 0.0
yolo_fps = 0.0


# ============================================================
# GLOBAL YOLO STATE
# ============================================================

latest_objects = []
latest_obstacle_detected = False

yolo_lock = threading.Lock()


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("=" * 60)
print("LOADING YOLO MODEL")
print("=" * 60)

model = YOLO(MODEL_PATH)

print("YOLO MODEL LOADED SUCCESSFULLY")
print("MODEL:", MODEL_PATH)
print("CLASSES:", model.names)


# ============================================================
# OPEN CAMERA
# ============================================================

def open_camera():

    print()
    print("=" * 60)
    print("OPENING CAMERA")
    print("=" * 60)

    cap = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2
    )

    # Try MJPEG format
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        HEIGHT
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        FPS
    )

    # Reduce buffering and latency if supported
    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open camera: {CAMERA_DEVICE}"
        )

    print("Camera opened successfully")
    print("Device:", CAMERA_DEVICE)

    return cap


# ============================================================
# CAMERA THREAD
# ============================================================

def camera_loop(cap):

    global latest_frame
    global camera_fps
    global running

    frame_count = 0
    start_time = time.time()

    failed_frames = 0

    print("[CAMERA] Camera thread started")

    while running:

        success, frame = cap.read()

        # ----------------------------------------------------
        # CAMERA FAILURE
        # ----------------------------------------------------

        if not success or frame is None:

            failed_frames += 1

            print(
                f"[CAMERA] Failed to grab frame "
                f"({failed_frames})"
            )

            time.sleep(0.05)

            # Do not immediately stop.
            # Allow temporary USB camera failures.
            if failed_frames >= 50:

                print(
                    "[CAMERA] Too many consecutive failures."
                )

                running = False

                break

            continue

        # ----------------------------------------------------
        # CAMERA RECOVERED
        # ----------------------------------------------------

        failed_frames = 0

        # ----------------------------------------------------
        # SAVE ONLY THE LATEST FRAME
        # ----------------------------------------------------

        with frame_lock:

            latest_frame = frame.copy()

        # ----------------------------------------------------
        # CALCULATE CAMERA FPS
        # ----------------------------------------------------

        frame_count += 1

        elapsed = time.time() - start_time

        if elapsed >= 1.0:

            camera_fps = frame_count / elapsed

            frame_count = 0

            start_time = time.time()


# ============================================================
# LANE DETECTION
# ============================================================

def detect_lane(frame):

    """
    Detect track lines.

    Pipeline:

        Camera Frame
            ↓
        Lower Region Of Interest
            ↓
        Grayscale
            ↓
        Gaussian Blur
            ↓
        Canny Edge Detection
            ↓
        Hough Line Detection
            ↓
        Estimate Lane Position
    """

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # REGION OF INTEREST
    # --------------------------------------------------------

    roi_y_start = int(height * 0.45)

    roi = frame[
        roi_y_start:height,
        0:width
    ]

    # --------------------------------------------------------
    # CONVERT TO GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # BLUR
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # --------------------------------------------------------
    # EDGE DETECTION
    # --------------------------------------------------------

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    # --------------------------------------------------------
    # HOUGH LINE DETECTION
    # --------------------------------------------------------

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=30,
        maxLineGap=40
    )

    lane_center = None

    # --------------------------------------------------------
    # PROCESS LINES
    # --------------------------------------------------------

    if lines is not None:

        x_positions = []

        # IMPORTANT:
        # Works with different OpenCV output shapes
        lines = np.asarray(
            lines
        ).reshape(-1, 4)

        for x1, y1, x2, y2 in lines:

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # Ignore nearly horizontal lines
            if abs(y2 - y1) < 15:

                continue

            # Bottom of ROI
            y_target = roi.shape[0] - 1

            # Calculate line x-coordinate
            # at bottom of the ROI
            if y2 != y1:

                x_target = x1 + (
                    (y_target - y1)
                    * (x2 - x1)
                    / (y2 - y1)
                )

                # Only keep valid positions
                if 0 <= x_target < width:

                    x_positions.append(
                        x_target
                    )

        # ----------------------------------------------------
        # ESTIMATE LANE CENTER
        # ----------------------------------------------------

        if len(x_positions) > 0:

            lane_center = int(
                np.mean(x_positions)
            )

    return lane_center, edges


# ============================================================
# POINT B DETECTION
# ============================================================

def detect_point_b(frame):

    """
    Detect ArUco Marker ID 42.

    Marker ID 42 represents Point B.
    """

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    aruco_dict = (
        cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
    )

    parameters = (
        cv2.aruco.DetectorParameters()
    )

    detector = cv2.aruco.ArucoDetector(
        aruco_dict,
        parameters
    )

    corners, ids, rejected = (
        detector.detectMarkers(
            gray
        )
    )

    point_b_detected = False
    marker_area = 0.0

    # --------------------------------------------------------
    # SEARCH FOR MARKER
    # --------------------------------------------------------

    if ids is not None:

        for i, marker_id in enumerate(
            ids.flatten()
        ):

            if int(marker_id) == POINT_B_ID:

                point_b_detected = True

                marker = corners[i][0]

                # Marker coordinates
                x_values = marker[:, 0]
                y_values = marker[:, 1]

                marker_width = (
                    np.max(x_values)
                    - np.min(x_values)
                )

                marker_height = (
                    np.max(y_values)
                    - np.min(y_values)
                )

                marker_area = (
                    marker_width
                    * marker_height
                )

                # Draw marker
                cv2.polylines(
                    frame,
                    [marker.astype(np.int32)],
                    True,
                    (0, 255, 0),
                    3
                )

                # Marker center
                center_x = int(
                    np.mean(x_values)
                )

                center_y = int(
                    np.mean(y_values)
                )

                cv2.circle(
                    frame,
                    (center_x, center_y),
                    6,
                    (0, 0, 255),
                    -1
                )

                # Label
                cv2.putText(
                    frame,
                    "POINT B - ID 42",
                    (
                        int(np.min(x_values)),
                        max(
                            30,
                            int(np.min(y_values)) - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

    return (
        point_b_detected,
        marker_area
    )


# ============================================================
# YOLO DETECTION
# ============================================================

def detect_objects(frame):

    """
    Run YOLO object detection.
    """

    global yolo_fps

    start_time = time.time()

    results = model(
        frame,
        conf=YOLO_CONFIDENCE,
        verbose=False
    )

    objects = []

    obstacle_detected = False

    obstacle_classes = {

        "person",
        "car",
        "truck",
        "bus",
        "bicycle",
        "motorcycle"

    }

    # --------------------------------------------------------
    # PROCESS YOLO RESULTS
    # --------------------------------------------------------

    for result in results:

        if result.boxes is None:

            continue

        for box in result.boxes:

            class_id = int(
                box.cls[0].item()
            )

            confidence = float(
                box.conf[0].item()
            )

            coordinates = (
                box.xyxy[0]
                .cpu()
                .numpy()
            )

            x1, y1, x2, y2 = coordinates

            class_name = model.names[
                class_id
            ]

            obj = {

                "name": class_name,

                "confidence": confidence,

                "bbox": (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                )

            }

            objects.append(
                obj
            )

            # ------------------------------------------------
            # OBSTACLE DETECTION
            # ------------------------------------------------

            if class_name in obstacle_classes:

                obstacle_detected = True

    # --------------------------------------------------------
    # YOLO FPS
    # --------------------------------------------------------

    elapsed = (
        time.time() - start_time
    )

    if elapsed > 0:

        yolo_fps = 1.0 / elapsed

    return (
        objects,
        obstacle_detected
    )


# ============================================================
# YOLO THREAD
# ============================================================

def yolo_loop():

    global latest_objects
    global latest_obstacle_detected
    global running

    print("[YOLO] Detection thread started")

    while running:

        frame = None

        # Get latest frame
        with frame_lock:

            if latest_frame is not None:

                frame = latest_frame.copy()

        if frame is None:

            time.sleep(0.05)

            continue

        try:

            objects, obstacle = (
                detect_objects(
                    frame
                )
            )

            with yolo_lock:

                latest_objects = objects

                latest_obstacle_detected = (
                    obstacle
                )

        except Exception as error:

            print(
                "[YOLO] Error:",
                error
            )

        # Do not overload the system
        time.sleep(YOLO_INTERVAL)


# ============================================================
# DRAW YOLO OBJECTS
# ============================================================

def draw_objects(frame, objects):

    for obj in objects:

        x1, y1, x2, y2 = (
            obj["bbox"]
        )

        label = (
            f"{obj['name']} "
            f"{obj['confidence']:.2f}"
        )

        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2
        )

        # Label
        cv2.putText(
            frame,
            label,
            (
                x1,
                max(
                    25,
                    y1 - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )


# ============================================================
# AGENT BRAIN
# ============================================================

def think(

    lane_center,

    frame_width,

    point_b_detected,

    point_b_area,

    obstacle_detected

):

    """
    AGENT DECISION PRIORITY:

    1. Obstacle detected
            ↓
        STOP

    2. Point B reached
            ↓
        STOP

    3. Point B visible
            ↓
        APPROACH

    4. Lane detected
            ↓
        FOLLOW LINE

    5. Nothing detected
            ↓
        STOP / SEARCH
    """

    # --------------------------------------------------------
    # PRIORITY 1: SAFETY
    # --------------------------------------------------------

    if obstacle_detected:

        return {

            "action": "STOP",

            "reason": "Obstacle detected",

            "steering": 0.0,

            "throttle": 0.0

        }

    # --------------------------------------------------------
    # PRIORITY 2: DESTINATION REACHED
    # --------------------------------------------------------

    if (
        point_b_detected
        and
        point_b_area >= POINT_B_STOP_AREA
    ):

        return {

            "action": "POINT_B_REACHED",

            "reason": "Point B marker is close",

            "steering": 0.0,

            "throttle": 0.0

        }

    # --------------------------------------------------------
    # PRIORITY 3: POINT B DETECTED
    # --------------------------------------------------------

    if point_b_detected:

        return {

            "action": "APPROACH_POINT_B",

            "reason": "Point B detected",

            "steering": 0.0,

            "throttle": 0.10

        }

    # --------------------------------------------------------
    # PRIORITY 4: FOLLOW LANE
    # --------------------------------------------------------

    if lane_center is not None:

        image_center = (
            frame_width / 2.0
        )

        error = (
            lane_center
            - image_center
        ) / image_center

        steering = np.clip(
            error * 0.5,
            -MAX_STEERING,
            MAX_STEERING
        )

        return {

            "action": "FOLLOW_LINE",

            "reason": "Lane detected",

            "steering": float(
                steering
            ),

            "throttle": BASE_THROTTLE

        }

    # --------------------------------------------------------
    # PRIORITY 5: LOST
    # --------------------------------------------------------

    return {

        "action": "SEARCHING",

        "reason": "Lane not detected",

        "steering": 0.0,

        "throttle": 0.0

    }


# ============================================================
# DRAW AGENT USER INTERFACE
# ============================================================

def draw_agent_ui(

    frame,

    decision,

    lane_center,

    point_b_detected,

    point_b_area,

    obstacle_detected

):

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # IMAGE CENTER
    # --------------------------------------------------------

    image_center = width // 2

    cv2.line(
        frame,
        (image_center, 0),
        (image_center, height),
        (255, 0, 0),
        2
    )

    # --------------------------------------------------------
    # LANE CENTER
    # --------------------------------------------------------

    if lane_center is not None:

        cv2.line(
            frame,
            (
                lane_center,
                int(height * 0.45)
            ),
            (
                lane_center,
                height
            ),
            (0, 255, 0),
            3
        )

    # --------------------------------------------------------
    # AGENT INFORMATION
    # --------------------------------------------------------

    info = [

        f"MISSION: {MISSION}",

        f"ACTION: {decision['action']}",

        f"REASON: {decision['reason']}",

        f"STEERING: {decision['steering']:.3f}",

        f"THROTTLE: {decision['throttle']:.3f}",

        f"CAMERA FPS: {camera_fps:.1f}",

        f"YOLO FPS: {yolo_fps:.1f}",

        f"POINT B: {point_b_detected}",

        f"MARKER AREA: {point_b_area:.0f}",

        f"OBSTACLE: {obstacle_detected}",

        f"SAFE MODE: {SAFE_MODE}"

    ]

    y = 30

    for text in info:

        # Black background
        cv2.putText(
            frame,
            text,
            (11, y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3
        )

        # White text
        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        y += 28


# ============================================================
# MAIN
# ============================================================

def main():

    global latest_frame
    global running

    print()
    print("=" * 60)
    print("JETRACER VISION AGENT V1")
    print("=" * 60)

    print()
    print("MISSION:")
    print(MISSION)

    print()

    # --------------------------------------------------------
    # OPEN CAMERA
    # --------------------------------------------------------

    cap = open_camera()

    # --------------------------------------------------------
    # START CAMERA THREAD
    # --------------------------------------------------------

    camera_thread = threading.Thread(
        target=camera_loop,
        args=(cap,),
        daemon=True
    )

    camera_thread.start()

    # --------------------------------------------------------
    # WAIT FOR FIRST FRAME
    # --------------------------------------------------------

    print("Waiting for camera...")

    wait_start = time.time()

    while True:

        with frame_lock:

            camera_ready = (
                latest_frame is not None
            )

        if camera_ready:

            break

        if time.time() - wait_start > 10:

            raise RuntimeError(
                "Camera did not provide frames."
            )

        time.sleep(0.05)

    print("Camera ready!")

    # --------------------------------------------------------
    # START YOLO THREAD
    # --------------------------------------------------------

    yolo_thread = threading.Thread(
        target=yolo_loop,
        daemon=True
    )

    yolo_thread.start()

    # --------------------------------------------------------
    # SAFETY MESSAGE
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("SAFE MODE ENABLED")
    print("THE JETRACER WILL NOT MOVE")
    print("=" * 60)

    print()
    print("Press Q to quit.")

    try:

        while running:

            # ------------------------------------------------
            # GET LATEST FRAME
            # ------------------------------------------------

            with frame_lock:

                if latest_frame is None:

                    continue

                frame = latest_frame.copy()

            # ------------------------------------------------
            # LANE DETECTION
            # ------------------------------------------------

            lane_center, edges = (
                detect_lane(
                    frame
                )
            )

            # ------------------------------------------------
            # POINT B DETECTION
            # ------------------------------------------------

            (
                point_b_detected,
                point_b_area

            ) = detect_point_b(
                frame
            )

            # ------------------------------------------------
            # GET LATEST YOLO RESULTS
            # ------------------------------------------------

            with yolo_lock:

                objects = (
                    latest_objects.copy()
                )

                obstacle_detected = (
                    latest_obstacle_detected
                )

            # ------------------------------------------------
            # AGENT THINKING
            # ------------------------------------------------

            decision = think(

                lane_center,

                frame.shape[1],

                point_b_detected,

                point_b_area,

                obstacle_detected

            )

            # ------------------------------------------------
            # DRAW YOLO RESULTS
            # ------------------------------------------------

            draw_objects(
                frame,
                objects
            )

            # ------------------------------------------------
            # DRAW AGENT INFORMATION
            # ------------------------------------------------

            draw_agent_ui(

                frame,

                decision,

                lane_center,

                point_b_detected,

                point_b_area,

                obstacle_detected

            )

            # ------------------------------------------------
            # DISPLAY CAMERA
            # ------------------------------------------------

            cv2.imshow(
                "JetRacer Vision Agent V1",
                frame
            )

            # ------------------------------------------------
            # DISPLAY LANE EDGES
            # ------------------------------------------------

            cv2.imshow(
                "Lane Detection",
                edges
            )

            # ------------------------------------------------
            # KEYBOARD CONTROL
            # ------------------------------------------------

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):

                print("Q pressed. Stopping agent.")

                running = False

                break

    except KeyboardInterrupt:

        print()
        print("Keyboard interrupt received.")

    except Exception as error:

        print()
        print("=" * 60)
        print("AGENT ERROR")
        print("=" * 60)

        print(error)

    finally:

        # ----------------------------------------------------
        # SAFE SHUTDOWN
        # ----------------------------------------------------

        print()
        print("Stopping agent safely...")

        running = False

        time.sleep(0.5)

        if cap is not None:

            cap.release()

        cv2.destroyAllWindows()

        print("Camera released.")
        print("Windows closed.")
        print("JetRacer Vision Agent stopped safely.")


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    main()