import cv2

# ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

# Point B marker ID
POINT_B_ID = 42

# Marker size in pixels
MARKER_SIZE = 600

# Generate marker
marker = cv2.aruco.generateImageMarker(
    aruco_dict,
    POINT_B_ID,
    MARKER_SIZE
)

cv2.imwrite("point_b_marker_42.png", marker)

print("Point B marker created successfully!")
print("Marker ID:", POINT_B_ID)
print("File: point_b_marker_42.png")
