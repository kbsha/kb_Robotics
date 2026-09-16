import cv2
import json
import os
import numpy as np
from ultralytics import YOLO


# ============================================================
# YOLOv8 + CAMERA + MANUAL BIRD VIEW + STOP ZONE
# ============================================================

# -----------------------------
# CAMERA
# -----------------------------

CAMERA_ID = 49

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# -----------------------------
# BIRD VIEW
# -----------------------------

BIRD_WIDTH = 640
BIRD_HEIGHT = 600

# -----------------------------
# YOLO
# -----------------------------

MODEL_PATH = "yolov8n.pt"

# Start LOW for debugging.
# Increase later if there are too many false detections.
CONFIDENCE = 0.20

# -----------------------------
# CALIBRATION
# -----------------------------

CALIBRATION_FILE = "calibration.json"


# ============================================================
# OBJECT CLASSES
# ============================================================

# Standard YOLOv8 COCO classes that we treat as obstacles.

OBSTACLE_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "train",
    "boat",
    "bench",
    "chair",
    "dog",
    "cat",
    "backpack",
    "suitcase",
}


# ============================================================
# DEFAULT BIRD-VIEW CALIBRATION
# ============================================================

default_src_points = np.float32([
    [120, 100],       # TOP LEFT
    [520, 100],       # TOP RIGHT
    [620, 450],       # BOTTOM RIGHT
    [20, 450],        # BOTTOM LEFT
])


# ============================================================
# DEFAULT STOP ZONE
# ============================================================

default_stop_zone = {
    "x1": 80,
    "y1": 320,
    "x2": 560,
    "y2": 580,
}


# ============================================================
# GLOBAL VARIABLES
# ============================================================

src_points = default_src_points.copy()

stop_zone = default_stop_zone.copy()

dragging_point = -1

CONFIDENCE = 0.20


# ============================================================
# LOAD CALIBRATION
# ============================================================

def load_calibration():

    global src_points
    global stop_zone

    if not os.path.exists(CALIBRATION_FILE):

        print("No calibration file found.")
        print("Using default calibration.")

        return

    try:

        with open(CALIBRATION_FILE, "r") as f:

            data = json.load(f)

        points = data.get("src_points")

        if points is not None and len(points) == 4:

            src_points = np.float32(points)

        zone = data.get("stop_zone")

        if zone is not None:

            stop_zone = zone

        print("Calibration loaded.")

    except Exception as e:

        print("Calibration loading error:", e)


# ============================================================
# SAVE CALIBRATION
# ============================================================

def save_calibration():

    data = {

        "src_points":
            src_points.tolist(),

        "stop_zone":
            stop_zone,

        "bird_width":
            BIRD_WIDTH,

        "bird_height":
            BIRD_HEIGHT,

    }

    with open(CALIBRATION_FILE, "w") as f:

        json.dump(
            data,
            f,
            indent=4
        )

    print()
    print("================================")
    print("CALIBRATION SAVED")
    print("================================")
    print()


# ============================================================
# RESET
# ============================================================

def reset_calibration():

    global src_points
    global stop_zone

    src_points = default_src_points.copy()

    stop_zone = default_stop_zone.copy()

    print("Calibration reset.")


# ============================================================
# CAMERA MOUSE CONTROL
# ============================================================

def camera_mouse_callback(event, x, y, flags, param):

    global dragging_point
    global src_points

    if event == cv2.EVENT_LBUTTONDOWN:

        distances = np.sqrt(
            (src_points[:, 0] - x) ** 2
            +
            (src_points[:, 1] - y) ** 2
        )

        closest_point = np.argmin(distances)

        if distances[closest_point] < 35:

            dragging_point = closest_point

    elif event == cv2.EVENT_MOUSEMOVE:

        if dragging_point != -1:

            # Keep points inside camera
            x = max(0, min(CAMERA_WIDTH - 1, x))
            y = max(0, min(CAMERA_HEIGHT - 1, y))

            src_points[dragging_point] = [
                x,
                y
            ]

    elif event == cv2.EVENT_LBUTTONUP:

        dragging_point = -1


# ============================================================
# STOP ZONE MOUSE CONTROL
# ============================================================

stop_drag_mode = False

stop_drag_corner = -1


def bird_mouse_callback(event, x, y, flags, param):

    global stop_drag_mode
    global stop_drag_corner
    global stop_zone

    x1 = stop_zone["x1"]
    y1 = stop_zone["y1"]
    x2 = stop_zone["x2"]
    y2 = stop_zone["y2"]

    corners = [
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),
    ]

    if event == cv2.EVENT_LBUTTONDOWN:

        distances = []

        for cx, cy in corners:

            distance = np.sqrt(
                (cx - x) ** 2
                +
                (cy - y) ** 2
            )

            distances.append(distance)

        closest = np.argmin(distances)

        if distances[closest] < 30:

            stop_drag_mode = True

            stop_drag_corner = closest

    elif event == cv2.EVENT_MOUSEMOVE:

        if stop_drag_mode:

            x = max(
                0,
                min(BIRD_WIDTH - 1, x)
            )

            y = max(
                100,
                min(BIRD_HEIGHT - 1, y)
            )

            if stop_drag_corner == 0:

                stop_zone["x1"] = x
                stop_zone["y1"] = y

            elif stop_drag_corner == 1:

                stop_zone["x2"] = x
                stop_zone["y1"] = y

            elif stop_drag_corner == 2:

                stop_zone["x2"] = x
                stop_zone["y2"] = y

            elif stop_drag_corner == 3:

                stop_zone["x1"] = x
                stop_zone["y2"] = y

    elif event == cv2.EVENT_LBUTTONUP:

        stop_drag_mode = False
        stop_drag_corner = -1


# ============================================================
# DRAW CAMERA CALIBRATION
# ============================================================

def draw_camera_view(frame):

    output = frame.copy()

    points = src_points.astype(np.int32)

    # Draw calibration polygon

    cv2.polylines(
        output,
        [points],
        True,
        (0, 255, 255),
        2
    )

    names = [
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

    for i, point in enumerate(points):

        x, y = point

        cv2.circle(
            output,
            (x, y),
            10,
            colors[i],
            -1
        )

        cv2.circle(
            output,
            (x, y),
            14,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output,
            names[i],
            (x + 12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colors[i],
            2
        )

    # Header

    cv2.rectangle(
        output,
        (0, 0),
        (640, 75),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        output,
        "DRAG CORNERS = BIRD VIEW CALIBRATION",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    cv2.putText(
        output,
        "S=SAVE   R=RESET   Q=QUIT",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    return output


# ============================================================
# DRAW STOP ZONE
# ============================================================

def draw_stop_zone(frame, stop):

    x1 = int(stop_zone["x1"])
    y1 = int(stop_zone["y1"])
    x2 = int(stop_zone["x2"])
    y2 = int(stop_zone["y2"])

    if stop:

        color = (0, 0, 255)

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

    # Rectangle

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        3
    )

    # Four draggable corners

    corners = [
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2)
    ]

    for cx, cy in corners:

        cv2.circle(
            frame,
            (cx, cy),
            10,
            color,
            -1
        )

        cv2.circle(
            frame,
            (cx, cy),
            13,
            (255, 255, 255),
            2
        )

    cv2.putText(
        frame,
        "STOP ZONE - DRAG CORNERS",
        (x1 + 5, max(115, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )


# ============================================================
# STATUS
# ============================================================

def draw_status(frame, stop):

    if stop:

        # RED

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

        # GREEN

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

    global CONFIDENCE

    print()
    print("====================================================")
    print("YOLOv8 BIRD VIEW OBSTACLE DETECTION")
    print("====================================================")
    print()

    print("Loading YOLOv8 model...")

    model = YOLO(MODEL_PATH)

    print("YOLO model loaded successfully.")
    print()

    # Print available classes

    print("YOLO classes loaded:")
    print(
        ", ".join(model.names.values())
    )
    print()

    # Load calibration

    load_calibration()

    # Camera

    cap = cv2.VideoCapture(
        CAMERA_ID
    )

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

    print("Camera opened successfully.")
    print()

    # Windows

    camera_window = "1 - RAW CAMERA / CALIBRATION"

    bird_window = "2 - BIRD VIEW / YOLO"

    cv2.namedWindow(
        camera_window,
        cv2.WINDOW_NORMAL
    )

    cv2.namedWindow(
        bird_window,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        camera_window,
        CAMERA_WIDTH,
        CAMERA_HEIGHT
    )

    cv2.resizeWindow(
        bird_window,
        BIRD_WIDTH,
        BIRD_HEIGHT
    )

    # Mouse

    cv2.setMouseCallback(
        camera_window,
        camera_mouse_callback
    )

    cv2.setMouseCallback(
        bird_window,
        bird_mouse_callback
    )

    print("====================================================")
    print("CONTROLS")
    print("====================================================")
    print("Drag the four corners in the camera window.")
    print("Drag STOP ZONE corners in the bird-view window.")
    print()
    print("S = Save calibration")
    print("R = Reset calibration")
    print("+ = Increase confidence")
    print("- = Decrease confidence")
    print("Q = Quit")
    print("====================================================")
    print()

    frame_number = 0

    while True:

        # ----------------------------------------------------
        # READ CAMERA
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:

            print("ERROR: Camera frame unavailable.")

            break

        frame = cv2.resize(
            frame,
            (
                CAMERA_WIDTH,
                CAMERA_HEIGHT
            )
        )

        # ----------------------------------------------------
        # RAW CAMERA
        # ----------------------------------------------------

        camera_view = draw_camera_view(
            frame
        )

        cv2.imshow(
            camera_window,
            camera_view
        )

        # ----------------------------------------------------
        # BIRD VIEW TRANSFORMATION
        # ----------------------------------------------------

        destination_points = np.float32([
            [0, 0],
            [BIRD_WIDTH - 1, 0],
            [BIRD_WIDTH - 1, BIRD_HEIGHT - 1],
            [0, BIRD_HEIGHT - 1]
        ])

        matrix = cv2.getPerspectiveTransform(
            src_points,
            destination_points
        )

        bird = cv2.warpPerspective(
            frame,
            matrix,
            (
                BIRD_WIDTH,
                BIRD_HEIGHT
            )
        )

        # ----------------------------------------------------
        # YOLO DETECTION
        # ----------------------------------------------------

        results = model.predict(
            source=bird,
            conf=CONFIDENCE,
            verbose=False
        )

        stop = False

        total_detections = 0

        obstacle_detections = 0

        # ----------------------------------------------------
        # PROCESS DETECTIONS
        # ----------------------------------------------------

        for result in results:

            if result.boxes is None:

                continue

            for box in result.boxes:

                total_detections += 1

                confidence = float(
                    box.conf[0]
                )

                class_id = int(
                    box.cls[0]
                )

                class_name = model.names[
                    class_id
                ]

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # --------------------------------------------
                # GROUND POINT
                # --------------------------------------------

                ground_x = int(
                    (x1 + x2) / 2
                )

                ground_y = int(
                    y2
                )

                # --------------------------------------------
                # CHECK OBSTACLE CLASS
                # --------------------------------------------

                is_obstacle = (
                    class_name
                    in OBSTACLE_CLASSES
                )

                # --------------------------------------------
                # CHECK STOP ZONE
                # --------------------------------------------

                in_stop_zone = (
                    stop_zone["x1"]
                    <= ground_x
                    <= stop_zone["x2"]
                    and
                    stop_zone["y1"]
                    <= ground_y
                    <= stop_zone["y2"]
                )

                # --------------------------------------------
                # COLOR
                # --------------------------------------------

                if is_obstacle and in_stop_zone:

                    color = (
                        0,
                        0,
                        255
                    )

                    stop = True

                    obstacle_detections += 1

                elif is_obstacle:

                    color = (
                        0,
                        255,
                        255
                    )

                    obstacle_detections += 1

                else:

                    color = (
                        255,
                        150,
                        0
                    )

                # --------------------------------------------
                # BOUNDING BOX
                # --------------------------------------------

                cv2.rectangle(
                    bird,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # --------------------------------------------
                # GROUND POINT
                # --------------------------------------------

                cv2.circle(
                    bird,
                    (
                        ground_x,
                        ground_y
                    ),
                    7,
                    color,
                    -1
                )

                # --------------------------------------------
                # LABEL
                # --------------------------------------------

                label = (
                    f"{class_name} "
                    f"{confidence:.2f}"
                )

                if in_stop_zone:

                    label += "  STOP ZONE"

                cv2.putText(
                    bird,
                    label,
                    (
                        x1,
                        max(110, y1 - 8)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    color,
                    2
                )

        # ----------------------------------------------------
        # DRAW STOP ZONE
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
        # DEBUG INFORMATION
        # ----------------------------------------------------

        cv2.putText(
            bird,
            f"YOLO detections: {total_detections}",
            (10, BIRD_HEIGHT - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            bird,
            f"Obstacles: {obstacle_detections}",
            (10, BIRD_HEIGHT - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            bird,
            f"Conf: {CONFIDENCE:.2f}",
            (470, BIRD_HEIGHT - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # DISPLAY BIRD VIEW
        # ----------------------------------------------------

        cv2.imshow(
            bird_window,
            bird
        )

        # ----------------------------------------------------
        # KEYBOARD
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        # Quit

        if key == ord("q"):

            break

        # Save calibration

        elif key == ord("s"):

            save_calibration()

        # Reset

        elif key == ord("r"):

            reset_calibration()

        # Increase confidence

        elif key == ord("+") or key == ord("="):

            CONFIDENCE = min(
                0.95,
                CONFIDENCE + 0.05
            )

            print(
                f"Confidence = {CONFIDENCE:.2f}"
            )

        # Decrease confidence

        elif key == ord("-"):

            CONFIDENCE = max(
                0.05,
                CONFIDENCE - 0.05
            )

            print(
                f"Confidence = {CONFIDENCE:.2f}"
            )

        frame_number += 1

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    cap.release()

    cv2.destroyAllWindows()

    print()
    print("Program stopped.")


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
