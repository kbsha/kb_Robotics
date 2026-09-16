#!/usr/bin/env python3

import cv2
import time
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

CAMERA_DEVICE = "/dev/video49"

WIDTH = 640
HEIGHT = 480
FPS = 30

MODEL_PATH = "yolov8n.pt"

CONFIDENCE = 0.40


# ============================================================
# MANUAL PERCEPTION WINDOW
# ============================================================
#
# Change these four numbers manually.
#
# Example:
#
# X = 20% to 80%
# Y = 40% to 100%
#
# ============================================================

WINDOW_X1 = 0.20
WINDOW_X2 = 0.80

WINDOW_Y1 = 0.40
WINDOW_Y2 = 1.00


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
    "dog",
    "cat",
}


# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH = "yolov8n.pt"

print("Loading YOLO...")

model = YOLO(MODEL_PATH)

print("YOLO loaded.")


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_DEVICE,
    cv2.CAP_V4L2
)

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

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


if not cap.isOpened():

    raise RuntimeError(
        f"Could not open camera {CAMERA_DEVICE}"
    )


print("Camera opened.")

print()
print("======================================")
print("JETRACER YOLO OBSTACLE DETECTOR")
print("======================================")
print()
print("Press Q to quit.")
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Camera frame failed.")

        continue


    height, width = frame.shape[:2]


    # ========================================================
    # CALCULATE PERCEPTION WINDOW
    # ========================================================

    wx1 = int(
        width * WINDOW_X1
    )

    wx2 = int(
        width * WINDOW_X2
    )

    wy1 = int(
        height * WINDOW_Y1
    )

    wy2 = int(
        height * WINDOW_Y2
    )


    # ========================================================
    # DRAW PERCEPTION WINDOW
    # ========================================================

    cv2.rectangle(
        frame,
        (wx1, wy1),
        (wx2, wy2),
        (255, 0, 255),
        3
    )


    cv2.putText(
        frame,
        "PERCEPTION WINDOW",
        (wx1, wy1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 255),
        2
    )


    # ========================================================
    # YOLO
    # ========================================================

    results = model(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )


    obstacle_detected = False

    obstacle_name = ""

    obstacle_confidence = 0.0


    # ========================================================
    # PROCESS DETECTIONS
    # ========================================================

    for result in results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            name = model.names[
                class_id
            ]


            # ------------------------------------------------
            # Ignore non-obstacle objects
            # ------------------------------------------------

            if name not in OBSTACLE_CLASSES:

                continue


            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )


            # ------------------------------------------------
            # Object center
            # ------------------------------------------------

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )


            # ------------------------------------------------
            # Check if object center is inside window
            # ------------------------------------------------

            inside_window = (

                wx1 <= center_x <= wx2

                and

                wy1 <= center_y <= wy2
            )


            # ------------------------------------------------
            # DRAW DETECTION
            # ------------------------------------------------

            if inside_window:

                obstacle_detected = True

                obstacle_name = name

                obstacle_confidence = confidence


                # RED BOX = STOP OBSTACLE

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )


                cv2.circle(
                    frame,
                    (center_x, center_y),
                    6,
                    (0, 0, 255),
                    -1
                )


                cv2.putText(
                    frame,
                    f"STOP: {name} "
                    f"{confidence:.2f}",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )


            else:

                # YELLOW BOX = DETECTED BUT
                # OUTSIDE PERCEPTION WINDOW

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    2
                )


                cv2.putText(
                    frame,
                    f"{name} "
                    f"{confidence:.2f}",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )


    # ========================================================
    # SAFETY DECISION
    # ========================================================

    if obstacle_detected:

        action = "STOP"

        speed_command = 0

        status_text = (
            f"OBSTACLE: {obstacle_name} "
            f"{obstacle_confidence:.2f}"
        )

    else:

        action = "CLEAR"

        speed_command = 10

        status_text = "NO OBSTACLE"


    # ========================================================
    # DISPLAY STATUS
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (400, 95),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        f"ACTION: {action}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"SPEED: {speed_command}/10",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        status_text,
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "JetRacer YOLO Obstacle",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# SHUTDOWN
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("YOLO obstacle detector stopped.")