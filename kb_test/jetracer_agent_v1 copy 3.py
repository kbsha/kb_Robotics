import cv2
import json
import os
import numpy as np
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_ID = 49

MODEL_PATH = "yolov8n.pt"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

BIRD_WIDTH = 640
BIRD_HEIGHT = 600

CONFIDENCE = 0.40

CALIBRATION_FILE = "calibration.json"

# Objects treated as obstacles.
OBSTACLE_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "chair",
    "dog",
    "cat",
    "bench",
    "backpack",
    "suitcase",
}

# ============================================================
# DEFAULT CALIBRATION
# ============================================================

default_src = np.float32([
    [150, 100],                       # top-left
    [490, 100],                       # top-right
    [620, 450],                       # bottom-right
    [20, 450],                        # bottom-left
])

default_dst = np.float32([
    [100, 0],
    [540, 0],
    [540, BIRD_HEIGHT],
    [100, BIRD_HEIGHT],
])

# Default STOP zone
default_stop_zone = {
    "x1": 80,
    "y1": 330,
    "x2": 560,
    "y2": 580
}

# ============================================================
# GLOBAL VARIABLES
# ============================================================

src_points = default_src.copy()

stop_zone = default_stop_zone.copy()

dragging_point = -1

mouse_window = "CAMERA - DRAG CORNERS"

show_camera = True
show_bird = True

# ============================================================
# LOAD / SAVE CALIBRATION
# ============================================================

def save_calibration():

    data = {
        "src_points": src_points.tolist(),
        "stop_zone": stop_zone,
        "bird_width": BIRD_WIDTH,
        "bird_height": BIRD_HEIGHT
    }

    with open(CALIBRATION_FILE, "w") as f:
        json.dump(data, f, indent=4)

    print("\nCalibration saved to:", CALIBRATION_FILE)


def load_calibration():

    global src_points
    global stop_zone

    if not os.path.exists(CALIBRATION_FILE):
        print("No calibration file found.")
        return

    try:

        with open(CALIBRATION_FILE, "r") as f:
            data = json.load(f)

        points = data.get("src_points")

        if points and len(points) == 4:
            src_points = np.float32(points)

        saved_zone = data.get("stop_zone")

        if saved_zone:
            stop_zone = saved_zone

        print("Calibration loaded.")

    except Exception as e:

        print("Could not load calibration:", e)


# ============================================================
# RESET
# ============================================================

def reset_calibration():

    global src_points
    global stop_zone

    src_points = default_src.copy()
    stop_zone = default_stop_zone.copy()

    print("Calibration reset.")


# ============================================================
# CAMERA MOUSE CALIBRATION
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global dragging_point
    global src_points

    # Mouse pressed
    if event == cv2.EVENT_LBUTTONDOWN:

        distances = np.sqrt(
            (src_points[:, 0] - x) ** 2 +
            (src_points[:, 1] - y) ** 2
        )

        closest = np.argmin(distances)

        if distances[closest] < 30:
            dragging_point = closest

    # Mouse moving
    elif event == cv2.EVENT_MOUSEMOVE:

        if dragging_point != -1:

            src_points[dragging_point] = [x, y]

    # Mouse released
    elif event == cv2.EVENT_LBUTTONUP:

        dragging_point = -1


# ============================================================
# DRAW CAMERA CALIBRATION
# ============================================================

def draw_camera_calibration(frame):

    display = frame.copy()

    # Draw polygon
    pts = src_points.astype(np.int32)

    cv2.polylines(
        display,
        [pts],
        True,
        (0, 255, 255),
        2
    )

    labels = [
        "TOP-LEFT",
        "TOP-RIGHT",
        "BOTTOM-RIGHT",
        "BOTTOM-LEFT"
    ]

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 0, 255)
    ]

    for i, point in enumerate(pts):

        x, y = point

        cv2.circle(
            display,
            (x, y),
            10,
            colors[i],
            -1
        )

        cv2.circle(
            display,
            (x, y),
            14,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            labels[i],
            (x + 12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colors[i],
            2
        )

    # Instructions
    cv2.rectangle(
        display,
        (0, 0),
        (640, 75),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        display,
        "DRAG 4 CORNERS TO CALIBRATE",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        display,
        "S=SAVE   R=RESET   Q=QUIT",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    return display


# ============================================================
# STOP ZONE
# ============================================================

def point_inside_stop_zone(x, y):

    return (
        stop_zone["x1"] <= x <= stop_zone["x2"]
        and
        stop_zone["y1"] <= y <= stop_zone["y2"]
    )


def draw_stop_zone(frame, stop):

    x1 = stop_zone["x1"]
    y1 = stop_zone["y1"]
    x2 = stop_zone["x2"]
    y2 = stop_zone["y2"]

    if stop:

        color = (0, 0, 255)

        # Transparent red area
        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            color,
            -1
        )

        frame[:] = cv2.addWeighted(
            overlay,
            0.25,
            frame,
            0.75,
            0
        )

    else:

        color = (0, 255, 255)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        3
    )

    cv2.putText(
        frame,
        "STOP ZONE",
        (x1 + 10, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )


# ============================================================
# STATUS DISPLAY
# ============================================================

def draw_status(frame, stop):

    if stop:

        cv2.rectangle(
            frame,
            (0, 0),
            (BIRD_WIDTH, 100),
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            "STOP",
            (235, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.3,
            (255, 255, 255),
            5
        )

    else:

        cv2.rectangle(
            frame,
            (0, 0),
            (BIRD_WIDTH, 100),
            (0, 120, 0),
            -1
        )

        cv2.putText(
            frame,
            "CLEAR",
            (215, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (255, 255, 255),
            4
        )


# ============================================================
# MAIN
# ============================================================

def main():

    global src_points
    global stop_zone
    global CONFIDENCE

    print("=" * 60)
    print("YOLOv8 BIRD VIEW OBSTACLE DETECTION")
    print("=" * 60)

    print("\nControls:")
    print("  Drag corners = bird-view calibration")
    print("  S             = save calibration")
    print("  R             = reset calibration")
    print("  +/-           = change YOLO confidence")
    print("  Q             = quit")
    print()

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("Loading YOLOv8...")

    model = YOLO(MODEL_PATH)

    print("YOLOv8 loaded.")

    # --------------------------------------------------------
    # LOAD CALIBRATION
    # --------------------------------------------------------

    load_calibration()

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    cap = cv2.VideoCapture(CAMERA_ID)

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    if not cap.isOpened():

        print("ERROR: Cannot open camera.")
        return

    # --------------------------------------------------------
    # MOUSE
    # --------------------------------------------------------

    cv2.namedWindow(mouse_window)

    cv2.setMouseCallback(
        mouse_window,
        mouse_callback
    )

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:

            print("Camera frame error.")
            break

        frame = cv2.resize(
            frame,
            (CAMERA_WIDTH, CAMERA_HEIGHT)
        )

        # ----------------------------------------------------
        # CAMERA CALIBRATION VIEW
        # ----------------------------------------------------

        camera_view = draw_camera_calibration(frame)

        cv2.imshow(
            mouse_window,
            camera_view
        )

        # ----------------------------------------------------
        # BIRD VIEW
        # ----------------------------------------------------

        destination = np.float32([
            [0, 0],
            [BIRD_WIDTH, 0],
            [BIRD_WIDTH, BIRD_HEIGHT],
            [0, BIRD_HEIGHT]
        ])

        matrix = cv2.getPerspectiveTransform(
            src_points,
            destination
        )

        bird = cv2.warpPerspective(
            frame,
            matrix,
            (BIRD_WIDTH, BIRD_HEIGHT)
        )

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        results = model.predict(
            bird,
            conf=CONFIDENCE,
            verbose=False
        )

        stop = False

        obstacle_count = 0

        # ----------------------------------------------------
        # PROCESS YOLO RESULTS
        # ----------------------------------------------------

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                confidence = float(
                    box.conf[0]
                )

                class_id = int(
                    box.cls[0]
                )

                class_name = model.names[class_id]

                # Only obstacle classes
                if class_name not in OBSTACLE_CLASSES:
                    continue

                obstacle_count += 1

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # ------------------------------------------------
                # IMPORTANT:
                # Bottom center represents approximate ground point
                # ------------------------------------------------

                ground_x = int(
                    (x1 + x2) / 2
                )

                ground_y = int(y2)

                in_zone = point_inside_stop_zone(
                    ground_x,
                    ground_y
                )

                if in_zone:

                    stop = True
                    color = (0, 0, 255)

                else:

                    color = (0, 255, 0)

                # ------------------------------------------------
                # BOUNDING BOX
                # ------------------------------------------------

                cv2.rectangle(
                    bird,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # ------------------------------------------------
                # GROUND POINT
                # ------------------------------------------------

                cv2.circle(
                    bird,
                    (ground_x, ground_y),
                    7,
                    color,
                    -1
                )

                # ------------------------------------------------
                # LABEL
                # ------------------------------------------------

                label = (
                    f"{class_name} "
                    f"{confidence:.2f}"
                )

                if in_zone:

                    label += " STOP"

                cv2.putText(
                    bird,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

        # ----------------------------------------------------
        # STOP ZONE
        # ----------------------------------------------------

        draw_stop_zone(
            bird,
            stop
        )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        draw_status(
            bird,
            stop
        )

        # ----------------------------------------------------
        # INFORMATION
        # ----------------------------------------------------

        cv2.putText(
            bird,
            f"Obstacles: {obstacle_count}",
            (10, BIRD_HEIGHT - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            bird,
            f"Confidence: {CONFIDENCE:.2f}",
            (400, BIRD_HEIGHT - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # SHOW BIRD VIEW
        # ----------------------------------------------------

        cv2.imshow(
            "BIRD VIEW - YOLOv8",
            bird
        )

        # ----------------------------------------------------
        # KEYBOARD
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord("q"):

            break

        # Save
        elif key == ord("s"):

            save_calibration()

        # Reset
        elif key == ord("r"):

            reset_calibration()

        # Increase confidence
        elif key == ord("+") or key == ord("="):

            # global CONFIDENCE

            CONFIDENCE = min(
                0.95,
                CONFIDENCE + 0.05
            )

            print(
                f"Confidence: {CONFIDENCE:.2f}"
            )

        # Decrease confidence
        elif key == ord("-"):

            CONFIDENCE = max(
                0.10,
                CONFIDENCE - 0.05
            )

            print(
                f"Confidence: {CONFIDENCE:.2f}"
            )

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    cap.release()

    cv2.destroyAllWindows()

    print("\nProgram stopped.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
