import cv2

CAMERA_DEVICE = "/dev/video48"  # Change this after checking devices

cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)

if not cap.isOpened():
    print(f"ERROR: Cannot open {CAMERA_DEVICE}")
    exit()

print(f"Camera opened: {CAMERA_DEVICE}")

count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("FAILED TO GRAB FRAME")
        break

    count += 1

    print(
        f"Frame {count}: "
        f"{frame.shape}"
    )

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()