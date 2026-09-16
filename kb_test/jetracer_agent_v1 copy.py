
#!/usr/bin/env python3

"""
============================================================
JETRACER VISION AGENT V2
============================================================

MISSION:

    FOLLOW LINE
        ↓
    DETECT OBSTACLE
        ↓
    ESTIMATE RISK 0–10
        ↓
    STOP / SLOW / GO
        ↓
    DETECT POINT B
        ↓
    STOP

PERCEPTION:

    - USB Camera
    - Manual Perception Window / ROI
    - YOLO Object Detection
    - Lane Detection
    - ArUco Point B Detection

SAFETY:

    SAFE_MODE = True

    The JetRacer motors WILL NOT MOVE.

IMPORTANT:

    The obstacle score in this version is a CAMERA-BASED
    PROXIMITY HEURISTIC.

    It is NOT a true physical distance measurement.

    For real autonomous operation, use an independent
    distance sensor such as:

        - LiDAR
        - ToF
        - Depth Camera
        - Ultrasonic sensor

CONTROL MODEL:

    risk_score:

        0–2   = LOW
        3–5   = MEDIUM
        6–7   = HIGH
        8–10  = CRITICAL

    speed_command:

        0     = STOP
        1–2   = VERY SLOW
        3–4   = SLOW
        5–6   = MODERATE
        7–8   = FAST
        9–10  = MAXIMUM

SAFETY POLICY:

    If obstacle risk >= 6:

        speed_command = 0
        action = STOP

    If obstacle risk is 3–5:

        speed_command <= 3

    If no obstacle:

        normal line-following speed is allowed.

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

CAMERA_DEVICE = "/dev/video49"

WIDTH = 640
HEIGHT = 480
FPS = 30

MODEL_PATH = "yolov8n.pt"

YOLO_CONFIDENCE = 0.40

# YOLO update interval.
#
# Lower value = faster obstacle updates.
#
# Example:
#
# 0.10 = approximately 10 updates/sec
# 0.20 = approximately 5 updates/sec
# 0.30 = approximately 3 updates/sec
#
YOLO_INTERVAL = 0.15


# ============================================================
# MANUAL PERCEPTION WINDOW
# ============================================================

"""
The perception window is manually defined as percentages
of the camera image.

Example:

    X:
        20% → 80%

    Y:
        40% → 100%

This means:

    Ignore the upper 40% of the image.

    Look for obstacles mainly in the lower/front area.

You can tune these values later.
"""

PERCEPTION_X_MIN = 0.20
PERCEPTION_X_MAX = 0.80

PERCEPTION_Y_MIN = 0.40
PERCEPTION_Y_MAX = 1.00


# ============================================================
# POINT B CONFIGURATION
# ============================================================

POINT_B_ID = 42

POINT_B_STOP_AREA = 50000


# ============================================================
# CONTROL CONFIGURATION
# ============================================================

SAFE_MODE = True

MAX_STEERING = 0.35

BASE_SPEED = 4

SLOW_SPEED = 2

VERY_SLOW_SPEED = 1

STOP_SPEED = 0


# ============================================================
# OBSTACLE CONFIGURATION
# ============================================================

"""
These are COARSE camera-based risk thresholds.

They are not centimeters/meters.

The risk score is calculated from:

    1. Bounding box area
    2. Vertical position
    3. Whether the obstacle overlaps the
       central driving corridor

Larger + lower + more central:

        ↓

Higher risk
"""

RISK_LOW_MAX = 2

RISK_MEDIUM_MAX = 5

RISK_HIGH_MAX = 7


# Minimum fraction of bounding box that must overlap
# the perception window.
#
# 0.20 means at least 20% of the obstacle bounding box
# must be inside the perception window.
MIN_ROI_OVERLAP = 0.20


# Minimum fraction of the obstacle width that should overlap
# the center driving corridor before applying additional risk.
CENTER_CORRIDOR_X_MIN = 0.30
CENTER_CORRIDOR_X_MAX = 0.70


# ============================================================
# OBSTACLE CLASSES
# ============================================================

OBSTACLE_CLASSES = {

    "person",
    "car",
    "truck",
    "bus",
    "bicycle",
    "motorcycle",

    # Useful for generic YOLO models
    "dog",
    "cat",

}


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

latest_risk_score = 0

latest_obstacle_count = 0

yolo_lock = threading.Lock()


# ============================================================
# GLOBAL SAFETY STATE
# ============================================================

last_obstacle_time = 0.0

OBSTACLE_HOLD_TIME = 0.50


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
# PERCEPTION WINDOW
# ============================================================

def get_perception_window(frame):

    """
    Calculate the manual perception window.

    Returns:

        x1
        y1
        x2
        y2

    Coordinates are in pixels.
    """

    height, width = frame.shape[:2]

    x1 = int(
        width * PERCEPTION_X_MIN
    )

    x2 = int(
        width * PERCEPTION_X_MAX
    )

    y1 = int(
        height * PERCEPTION_Y_MIN
    )

    y2 = int(
        height * PERCEPTION_Y_MAX
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# BOUNDING BOX OVERLAP
# ============================================================

def calculate_bbox_overlap(
    bbox,
    roi
):

    """
    Calculate how much of a bounding box lies
    inside the perception window.

    Returns:

        overlap_ratio

    0.0 = completely outside

    1.0 = completely inside
    """

    bx1, by1, bx2, by2 = bbox

    rx1, ry1, rx2, ry2 = roi

    # --------------------------------------------------------
    # Intersection
    # --------------------------------------------------------

    ix1 = max(
        bx1,
        rx1
    )

    iy1 = max(
        by1,
        ry1
    )

    ix2 = min(
        bx2,
        rx2
    )

    iy2 = min(
        by2,
        ry2
    )

    # No intersection

    if ix2 <= ix1 or iy2 <= iy1:

        return 0.0

    intersection_area = (
        (ix2 - ix1)
        *
        (iy2 - iy1)
    )

    bbox_area = (
        max(
            1,
            bx2 - bx1
        )
        *
        max(
            1,
            by2 - by1
        )
    )

    overlap_ratio = (
        intersection_area
        /
        bbox_area
    )

    return float(
        overlap_ratio
    )


# ============================================================
# CENTER CORRIDOR OVERLAP
# ============================================================

def calculate_center_overlap(
    bbox,
    frame_width
):

    """
    Determine whether the obstacle overlaps
    the central driving corridor.

    Returns:

        center_overlap_ratio
    """

    x1, y1, x2, y2 = bbox

    corridor_x1 = int(
        frame_width
        *
        CENTER_CORRIDOR_X_MIN
    )

    corridor_x2 = int(
        frame_width
        *
        CENTER_CORRIDOR_X_MAX
    )

    intersection_x1 = max(
        x1,
        corridor_x1
    )

    intersection_x2 = min(
        x2,
        corridor_x2
    )

    if intersection_x2 <= intersection_x1:

        return 0.0

    obstacle_width = max(
        1,
        x2 - x1
    )

    overlap_width = (
        intersection_x2
        -
        intersection_x1
    )

    return float(
        overlap_width
        /
        obstacle_width
    )


# ============================================================
# RISK SCORE
# ============================================================

def calculate_risk_score(
    bbox,
    confidence,
    frame_shape
):

    """
    Calculate a camera-based obstacle risk score.

    Range:

        0–10

    Factors:

        - bounding-box size
        - vertical position
        - confidence
        - center corridor overlap

    IMPORTANT:

        This is NOT physical distance.
    """

    frame_height, frame_width = (
        frame_shape[:2]
    )

    x1, y1, x2, y2 = bbox

    # --------------------------------------------------------
    # Bounding box dimensions
    # --------------------------------------------------------

    bbox_width = max(
        1,
        x2 - x1
    )

    bbox_height = max(
        1,
        y2 - y1
    )

    bbox_area = (
        bbox_width
        *
        bbox_height
    )

    frame_area = (
        frame_width
        *
        frame_height
    )

    area_ratio = (
        bbox_area
        /
        frame_area
    )

    # --------------------------------------------------------
    # Vertical position
    # --------------------------------------------------------

    center_y = (
        (y1 + y2)
        /
        2.0
    )

    vertical_position = (
        center_y
        /
        frame_height
    )

    # --------------------------------------------------------
    # Center overlap
    # --------------------------------------------------------

    center_overlap = (
        calculate_center_overlap(
            bbox,
            frame_width
        )
    )

    # --------------------------------------------------------
    # Start risk
    # --------------------------------------------------------

    risk = 0.0

    # --------------------------------------------------------
    # SIZE CONTRIBUTION
    # --------------------------------------------------------

    if area_ratio >= 0.35:

        risk += 5.0

    elif area_ratio >= 0.20:

        risk += 4.0

    elif area_ratio >= 0.10:

        risk += 3.0

    elif area_ratio >= 0.05:

        risk += 2.0

    elif area_ratio >= 0.02:

        risk += 1.0

    # --------------------------------------------------------
    # VERTICAL POSITION
    # --------------------------------------------------------

    if vertical_position >= 0.85:

        risk += 3.0

    elif vertical_position >= 0.70:

        risk += 2.0

    elif vertical_position >= 0.55:

        risk += 1.0

    # --------------------------------------------------------
    # CENTER CORRIDOR
    # --------------------------------------------------------

    if center_overlap >= 0.70:

        risk += 2.0

    elif center_overlap >= 0.40:

        risk += 1.0

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    if confidence >= 0.80:

        risk += 1.0

    # --------------------------------------------------------
    # LIMIT
    # --------------------------------------------------------

    risk = int(
        np.clip(
            round(risk),
            0,
            10
        )
    )

    return risk


# ============================================================
# CAMERA
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

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *"MJPG"
        )
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

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open camera: "
            f"{CAMERA_DEVICE}"
        )

    print(
        "Camera opened successfully"
    )

    print(
        "Device:",
        CAMERA_DEVICE
    )

    print(
        f"Resolution: {WIDTH}x{HEIGHT}"
    )

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

    print(
        "[CAMERA] Camera thread started"
    )

    while running:

        success, frame = cap.read()

        # ----------------------------------------------------
        # FAILURE
        # ----------------------------------------------------

        if not success or frame is None:

            failed_frames += 1

            print(
                "[CAMERA] Failed to grab frame "
                f"({failed_frames})"
            )

            time.sleep(0.05)

            if failed_frames >= 50:

                print(
                    "[CAMERA] Too many failures."
                )

                running = False

                break

            continue

        # ----------------------------------------------------
        # RECOVERED
        # ----------------------------------------------------

        failed_frames = 0

        # ----------------------------------------------------
        # ONLY KEEP LATEST FRAME
        # ----------------------------------------------------

        with frame_lock:

            latest_frame = frame.copy()

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        frame_count += 1

        elapsed = (
            time.time()
            -
            start_time
        )

        if elapsed >= 1.0:

            camera_fps = (
                frame_count
                /
                elapsed
            )

            frame_count = 0

            start_time = time.time()


# ============================================================
# LANE DETECTION
# ============================================================

def detect_lane(frame):

    """
    Detect lane center using:

        ROI
        ↓
        grayscale
        ↓
        blur
        ↓
        Canny
        ↓
        Hough lines
    """

    height, width = (
        frame.shape[:2]
    )

    roi_y_start = int(
        height * 0.45
    )

    roi = frame[
        roi_y_start:height,
        0:width
    ]

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=30,
        maxLineGap=40
    )

    lane_center = None

    if lines is not None:

        x_positions = []

        lines = np.asarray(
            lines
        ).reshape(
            -1,
            4
        )

        for x1, y1, x2, y2 in lines:

            x1 = int(x1)

            y1 = int(y1)

            x2 = int(x2)

            y2 = int(y2)

            # Ignore horizontal lines

            if abs(
                y2 - y1
            ) < 15:

                continue

            y_target = (
                roi.shape[0] - 1
            )

            if y2 != y1:

                x_target = (
                    x1
                    +
                    (
                        (
                            y_target
                            -
                            y1
                        )
                        *
                        (
                            x2
                            -
                            x1
                        )
                        /
                        (
                            y2
                            -
                            y1
                        )
                    )
                )

                if (
                    0
                    <=
                    x_target
                    <
                    width
                ):

                    x_positions.append(
                        x_target
                    )

        if len(
            x_positions
        ) > 0:

            lane_center = int(
                np.mean(
                    x_positions
                )
            )

    return (
        lane_center,
        edges
    )


# ============================================================
# POINT B DETECTION
# ============================================================

def detect_point_b(frame):

    """
    Detect ArUco marker ID 42.
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

    if ids is not None:

        for i, marker_id in enumerate(
            ids.flatten()
        ):

            if int(
                marker_id
            ) == POINT_B_ID:

                point_b_detected = True

                marker = corners[i][0]

                x_values = marker[:, 0]

                y_values = marker[:, 1]

                marker_width = (
                    np.max(x_values)
                    -
                    np.min(x_values)
                )

                marker_height = (
                    np.max(y_values)
                    -
                    np.min(y_values)
                )

                marker_area = (
                    marker_width
                    *
                    marker_height
                )

                cv2.polylines(
                    frame,
                    [
                        marker.astype(
                            np.int32
                        )
                    ],
                    True,
                    (0, 255, 0),
                    3
                )

                center_x = int(
                    np.mean(
                        x_values
                    )
                )

                center_y = int(
                    np.mean(
                        y_values
                    )
                )

                cv2.circle(
                    frame,
                    (
                        center_x,
                        center_y
                    ),
                    6,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    frame,
                    "POINT B - ID 42",
                    (
                        int(
                            np.min(
                                x_values
                            )
                        ),
                        max(
                            30,
                            int(
                                np.min(
                                    y_values
                                )
                            )
                            -
                            10
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

    global yolo_fps

    start_time = time.time()

    frame_height, frame_width = (
        frame.shape[:2]
    )

    perception_roi = (
        get_perception_window(
            frame
        )
    )

    px1, py1, px2, py2 = (
        perception_roi
    )

    # --------------------------------------------------------
    # YOLO RUNS ON FULL IMAGE
    # --------------------------------------------------------

    results = model(
        frame,
        conf=YOLO_CONFIDENCE,
        verbose=False
    )

    objects = []

    obstacle_detected = False

    highest_risk = 0

    obstacle_count = 0

    # --------------------------------------------------------
    # PROCESS RESULTS
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

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            class_name = model.names[
                class_id
            ]

            # ------------------------------------------------
            # CHECK PERCEPTION WINDOW
            # ------------------------------------------------

            overlap = (
                calculate_bbox_overlap(
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    perception_roi
                )
            )

            inside_perception_window = (
                overlap
                >=
                MIN_ROI_OVERLAP
            )

            risk = 0

            is_obstacle = (
                class_name
                in
                OBSTACLE_CLASSES
            )

            # ------------------------------------------------
            # RISK ONLY FOR OBSTACLES
            # INSIDE PERCEPTION WINDOW
            # ------------------------------------------------

            if (
                is_obstacle
                and
                inside_perception_window
            ):

                obstacle_count += 1

                risk = calculate_risk_score(
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    confidence,
                    frame.shape
                )

                obstacle_detected = True

                highest_risk = max(
                    highest_risk,
                    risk
                )

            # ------------------------------------------------
            # OBJECT RECORD
            # ------------------------------------------------

            obj = {

                "name": class_name,

                "confidence": confidence,

                "bbox": (
                    x1,
                    y1,
                    x2,
                    y2
                ),

                "roi_overlap": overlap,

                "inside_perception": (
                    inside_perception_window
                ),

                "risk_score": risk

            }

            objects.append(
                obj
            )

    # --------------------------------------------------------
    # YOLO FPS
    # --------------------------------------------------------

    elapsed = (
        time.time()
        -
        start_time
    )

    if elapsed > 0:

        yolo_fps = (
            1.0
            /
            elapsed
        )

    return (
        objects,
        obstacle_detected,
        highest_risk,
        obstacle_count
    )


# ============================================================
# YOLO THREAD
# ============================================================

def yolo_loop():

    global latest_objects

    global latest_obstacle_detected

    global latest_risk_score

    global latest_obstacle_count

    global last_obstacle_time

    global running

    print(
        "[YOLO] Detection thread started"
    )

    while running:

        frame = None

        # ----------------------------------------------------
        # GET LATEST FRAME
        # ----------------------------------------------------

        with frame_lock:

            if latest_frame is not None:

                frame = (
                    latest_frame.copy()
                )

        if frame is None:

            time.sleep(
                0.02
            )

            continue

        try:

            (
                objects,
                obstacle,
                risk,
                obstacle_count

            ) = detect_objects(
                frame
            )

            # ------------------------------------------------
            # OBSTACLE DETECTED
            # ------------------------------------------------

            if obstacle:

                last_obstacle_time = (
                    time.time()
                )

            # ------------------------------------------------
            # HOLD OBSTACLE STATE
            #
            # This prevents a single missed YOLO frame
            # from immediately changing STOP → GO.
            # ------------------------------------------------

            obstacle_recent = (
                time.time()
                -
                last_obstacle_time
                <=
                OBSTACLE_HOLD_TIME
            )

            if obstacle_recent:

                final_obstacle = True

            else:

                final_obstacle = False

                risk = 0

            # ------------------------------------------------
            # UPDATE SHARED STATE
            # ------------------------------------------------

            with yolo_lock:

                latest_objects = (
                    objects
                )

                latest_obstacle_detected = (
                    final_obstacle
                )

                latest_risk_score = (
                    risk
                )

                latest_obstacle_count = (
                    obstacle_count
                )

        except Exception as error:

            print(
                "[YOLO] ERROR:",
                error
            )

        time.sleep(
            YOLO_INTERVAL
        )


# ============================================================
# DRAW PERCEPTION WINDOW
# ============================================================

def draw_perception_window(
    frame
):

    x1, y1, x2, y2 = (
        get_perception_window(
            frame
        )
    )

    # --------------------------------------------------------
    # WINDOW BORDER
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (
            x1,
            y1
        ),
        (
            x2,
            y2
        ),
        (255, 0, 255),
        2
    )

    # --------------------------------------------------------
    # LABEL
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "PERCEPTION WINDOW",
        (
            x1 + 5,
            max(
                25,
                y1 - 10
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 255),
        2
    )

    # --------------------------------------------------------
    # CENTER CORRIDOR
    # --------------------------------------------------------

    height, width = (
        frame.shape[:2]
    )

    cx1 = int(
        width
        *
        CENTER_CORRIDOR_X_MIN
    )

    cx2 = int(
        width
        *
        CENTER_CORRIDOR_X_MAX
    )

    cv2.line(
        frame,
        (
            cx1,
            y1
        ),
        (
            cx1,
            y2
        ),
        (255, 100, 0),
        1
    )

    cv2.line(
        frame,
        (
            cx2,
            y1
        ),
        (
            cx2,
            y2
        ),
        (255, 100, 0),
        1
    )


# ============================================================
# DRAW OBJECTS
# ============================================================

def draw_objects(
    frame,
    objects
):

    for obj in objects:

        x1, y1, x2, y2 = (
            obj["bbox"]
        )

        name = obj["name"]

        confidence = (
            obj["confidence"]
        )

        inside = (
            obj["inside_perception"]
        )

        risk = (
            obj["risk_score"]
        )

        # ----------------------------------------------------
        # ONLY OBSTACLES INSIDE ROI GET RISK DISPLAY
        # ----------------------------------------------------

        if inside and risk > 0:

            label = (
                f"{name} "
                f"{confidence:.2f} "
                f"RISK:{risk}/10"
            )

            thickness = 3

        else:

            label = (
                f"{name} "
                f"{confidence:.2f}"
            )

            thickness = 2

        # ----------------------------------------------------
        # BOUNDING BOX
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (
                x1,
                y1
            ),
            (
                x2,
                y2
            ),
            (0, 255, 255),
            thickness
        )

        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

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
            0.55,
            (0, 255, 255),
            2
        )

        # ----------------------------------------------------
        # ROI STATUS
        # ----------------------------------------------------

        if inside:

            cv2.putText(
                frame,
                "IN PERCEPTION",
                (
                    x1,
                    min(
                        frame.shape[0] - 10,
                        y2 + 20
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )


# ============================================================
# RISK DESCRIPTION
# ============================================================

def risk_description(
    risk
):

    if risk <= 2:

        return "LOW"

    if risk <= 5:

        return "MEDIUM"

    if risk <= 7:

        return "HIGH"

    return "CRITICAL"


# ============================================================
# AGENT BRAIN
# ============================================================

def think(

    lane_center,

    frame_width,

    point_b_detected,

    point_b_area,

    obstacle_detected,

    risk_score

):

    """
    AGENT DECISION PRIORITY:

    1. CRITICAL / HIGH OBSTACLE
            ↓
        STOP

    2. MEDIUM OBSTACLE
            ↓
        VERY SLOW

    3. POINT B REACHED
            ↓
        STOP

    4. POINT B VISIBLE
            ↓
        APPROACH

    5. LANE DETECTED
            ↓
        FOLLOW LINE

    6. NOTHING
            ↓
        STOP

    SAFETY PRINCIPLE:

        Obstacle decision always overrides
        navigation decision.
    """

    # ========================================================
    # SAFETY PRIORITY 1
    # ========================================================

    if (
        obstacle_detected
        and
        risk_score >= 6
    ):

        return {

            "action": "STOP",

            "reason": (
                "Obstacle risk "
                f"{risk_score}/10"
            ),

            "steering": 0.0,

            "throttle": 0.0,

            "speed_command": 0,

            "risk_score": risk_score,

            "risk_level": (
                risk_description(
                    risk_score
                )
            )

        }

    # ========================================================
    # SAFETY PRIORITY 2
    # ========================================================

    if (
        obstacle_detected
        and
        risk_score >= 3
    ):

        return {

            "action": "SLOW",

            "reason": (
                "Obstacle detected "
                f"risk={risk_score}/10"
            ),

            "steering": 0.0,

            "throttle": 0.10,

            "speed_command": (
                VERY_SLOW_SPEED
            ),

            "risk_score": risk_score,

            "risk_level": (
                risk_description(
                    risk_score
                )
            )

        }

    # ========================================================
    # POINT B REACHED
    # ========================================================

    if (
        point_b_detected
        and
        point_b_area
        >=
        POINT_B_STOP_AREA
    ):

        return {

            "action": "POINT_B_REACHED",

            "reason": (
                "Point B marker is close"
            ),

            "steering": 0.0,

            "throttle": 0.0,

            "speed_command": 0,

            "risk_score": risk_score,

            "risk_level": (
                risk_description(
                    risk_score
                )
            )

        }

    # ========================================================
    # POINT B DETECTED
    # ========================================================

    if point_b_detected:

        return {

            "action": "APPROACH_POINT_B",

            "reason": (
                "Point B detected"
            ),

            "steering": 0.0,

            "throttle": 0.10,

            "speed_command": 2,

            "risk_score": risk_score,

            "risk_level": (
                risk_description(
                    risk_score
                )
            )

        }

    # ========================================================
    # FOLLOW LANE
    # ========================================================

    if lane_center is not None:

        image_center = (
            frame_width
            /
            2.0
        )

        error = (
            lane_center
            -
            image_center
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

            "throttle": (
                BASE_SPEED
                /
                10.0
            ),

            "speed_command": (
                BASE_SPEED
            ),

            "risk_score": risk_score,

            "risk_level": (
                risk_description(
                    risk_score
                )
            )

        }

    # ========================================================
    # LOST
    # ========================================================

    return {

        "action": "STOP",

        "reason": (
            "Lane not detected"
        ),

        "steering": 0.0,

        "throttle": 0.0,

        "speed_command": 0,

        "risk_score": risk_score,

        "risk_level": (
            risk_description(
                risk_score
            )
        )

    }


# ============================================================
# DRAW AGENT UI
# ============================================================

def draw_agent_ui(

    frame,

    decision,

    lane_center,

    point_b_detected,

    point_b_area,

    obstacle_detected,

    obstacle_count

):

    height, width = (
        frame.shape[:2]
    )

    # --------------------------------------------------------
    # IMAGE CENTER
    # --------------------------------------------------------

    image_center = (
        width // 2
    )

    cv2.line(
        frame,
        (
            image_center,
            0
        ),
        (
            image_center,
            height
        ),
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
                int(
                    height * 0.45
                )
            ),
            (
                lane_center,
                height
            ),
            (0, 255, 0),
            3
        )

    # --------------------------------------------------------
    # PERCEPTION WINDOW
    # --------------------------------------------------------

    draw_perception_window(
        frame
    )

    # --------------------------------------------------------
    # INFORMATION
    # --------------------------------------------------------

    risk = decision[
        "risk_score"
    ]

    info = [

        f"MISSION: {MISSION}",

        f"ACTION: {decision['action']}",

        f"REASON: {decision['reason']}",

        f"RISK: {risk}/10",

        f"RISK LEVEL: {decision['risk_level']}",

        f"SPEED COMMAND: "
        f"{decision['speed_command']}/10",

        f"STEERING: "
        f"{decision['steering']:.3f}",

        f"THROTTLE: "
        f"{decision['throttle']:.3f}",

        f"OBSTACLE: "
        f"{obstacle_detected}",

        f"OBSTACLES IN ROI: "
        f"{obstacle_count}",

        f"POINT B: "
        f"{point_b_detected}",

        f"MARKER AREA: "
        f"{point_b_area:.0f}",

        f"CAMERA FPS: "
        f"{camera_fps:.1f}",

        f"YOLO FPS: "
        f"{yolo_fps:.1f}",

        f"SAFE MODE: "
        f"{SAFE_MODE}"

    ]

    y = 25

    for text in info:

        # Background

        cv2.putText(
            frame,
            text,
            (
                11,
                y + 1
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            3
        )

        # Text

        cv2.putText(
            frame,
            text,
            (
                10,
                y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )

        y += 23


# ============================================================
# SAFE MOTOR OUTPUT
# ============================================================

def send_motor_command(
    steering,
    speed_command
):

    """
    MOTOR OUTPUT PLACEHOLDER.

    SAFE_MODE = True:

        No motor command is sent.

    Future real motor interface should be connected
    here only after the perception and safety system
    has been thoroughly validated.
    """

    if SAFE_MODE:

        return

    # ========================================================
    # DO NOT IMPLEMENT REAL MOTOR CONTROL HERE YET.
    # ========================================================
    #
    # Future example architecture:
    #
    # motor_controller.set_steering(...)
    # motor_controller.set_speed(...)
    #
    # ========================================================

    pass


# ============================================================
# MAIN
# ============================================================

def main():

    global running

    print()

    print("=" * 60)

    print("JETRACER VISION AGENT V2")

    print("=" * 60)

    print()

    print(
        "MISSION:"
    )

    print(
        MISSION
    )

    print()

    print(
        "PERCEPTION WINDOW:"
    )

    print(
        f"X: "
        f"{PERCEPTION_X_MIN * 100:.0f}% "
        f"→ "
        f"{PERCEPTION_X_MAX * 100:.0f}%"
    )

    print(
        f"Y: "
        f"{PERCEPTION_Y_MIN * 100:.0f}% "
        f"→ "
        f"{PERCEPTION_Y_MAX * 100:.0f}%"
    )

    print()

    print(
        "RISK SYSTEM:"
    )

    print(
        "0–2   LOW"
    )

    print(
        "3–5   MEDIUM"
    )

    print(
        "6–7   HIGH"
    )

    print(
        "8–10  CRITICAL"
    )

    print()

    # ========================================================
    # OPEN CAMERA
    # ========================================================

    cap = open_camera()

    # ========================================================
    # START CAMERA THREAD
    # ========================================================

    camera_thread = threading.Thread(
        target=camera_loop,
        args=(cap,),
        daemon=True
    )

    camera_thread.start()

    # ========================================================
    # WAIT FOR CAMERA
    # ========================================================

    print(
        "Waiting for camera..."
    )

    wait_start = time.time()

    while True:

        with frame_lock:

            camera_ready = (
                latest_frame
                is not None
            )

        if camera_ready:

            break

        if (
            time.time()
            -
            wait_start
            >
            10
        ):

            running = False

            cap.release()

            raise RuntimeError(
                "Camera did not provide frames."
            )

        time.sleep(
            0.05
        )

    print(
        "Camera ready!"
    )

    # ========================================================
    # START YOLO
    # ========================================================

    yolo_thread = threading.Thread(
        target=yolo_loop,
        daemon=True
    )

    yolo_thread.start()

    # ========================================================
    # SAFE MODE
    # ========================================================

    print()

    print("=" * 60)

    print(
        "SAFE MODE ENABLED"
    )

    print(
        "THE JETRACER WILL NOT MOVE"
    )

    print("=" * 60)

    print()

    print(
        "Press Q to quit."
    )

    # ========================================================
    # MAIN LOOP
    # ========================================================

    try:

        while running:

            # ------------------------------------------------
            # GET FRAME
            # ------------------------------------------------

            with frame_lock:

                if latest_frame is None:

                    continue

                frame = (
                    latest_frame.copy()
                )

            # ------------------------------------------------
            # LANE
            # ------------------------------------------------

            (
                lane_center,
                edges
            ) = detect_lane(
                frame
            )

            # ------------------------------------------------
            # POINT B
            # ------------------------------------------------

            (
                point_b_detected,
                point_b_area
            ) = detect_point_b(
                frame
            )

            # ------------------------------------------------
            # YOLO STATE
            # ------------------------------------------------

            with yolo_lock:

                objects = (
                    latest_objects.copy()
                )

                obstacle_detected = (
                    latest_obstacle_detected
                )

                risk_score = (
                    latest_risk_score
                )

                obstacle_count = (
                    latest_obstacle_count
                )

            # ------------------------------------------------
            # THINK
            # ------------------------------------------------

            decision = think(

                lane_center,

                frame.shape[1],

                point_b_detected,

                point_b_area,

                obstacle_detected,

                risk_score

            )

            # ------------------------------------------------
            # SEND MOTOR COMMAND
            # ------------------------------------------------

            send_motor_command(

                decision[
                    "steering"
                ],

                decision[
                    "speed_command"
                ]

            )

            # ------------------------------------------------
            # DRAW OBJECTS
            # ------------------------------------------------

            draw_objects(
                frame,
                objects
            )

            # ------------------------------------------------
            # DRAW UI
            # ------------------------------------------------

            draw_agent_ui(

                frame,

                decision,

                lane_center,

                point_b_detected,

                point_b_area,

                obstacle_detected,

                obstacle_count

            )

            # ------------------------------------------------
            # DISPLAY CAMERA
            # ------------------------------------------------

            cv2.imshow(
                "JetRacer Vision Agent V2",
                frame
            )

            # ------------------------------------------------
            # DISPLAY EDGES
            # ------------------------------------------------

            cv2.imshow(
                "Lane Detection",
                edges
            )

            # ------------------------------------------------
            # KEYBOARD
            # ------------------------------------------------

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if key == ord("q"):

                print()

                print(
                    "Q pressed."
                )

                print(
                    "Stopping agent safely."
                )

                running = False

                break

    except KeyboardInterrupt:

        print()

        print(
            "Keyboard interrupt received."
        )

    except Exception as error:

        print()

        print("=" * 60)

        print(
            "AGENT ERROR"
        )

        print("=" * 60)

        print(
            error
        )

    finally:

        # ====================================================
        # SAFETY SHUTDOWN
        # ====================================================

        print()

        print(
            "Stopping agent safely..."
        )

        running = False

        # Explicit zero-speed command.
        #
        # SAFE_MODE means this does not reach the motors.

        send_motor_command(
            0.0,
            0
        )

        time.sleep(
            0.5
        )

        if cap is not None:

            cap.release()

        cv2.destroyAllWindows()

        print(
            "Camera released."
        )

        print(
            "Windows closed."
        )

        print(
            "JetRacer Vision Agent V2 stopped safely."
        )


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    main()

