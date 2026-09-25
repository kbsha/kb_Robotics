#!/usr/bin/env python3
"""
JETRACER SIDE  --  run this ON the JetRacer:   python3 jetracer_server.py

  * Streams the camera as MJPEG       ->  http://<jetracer-ip>:8081/video
  * Listens for drive commands (UDP)  ->  port 5005
  * Applies them to NvidiaRacecar (same steering/throttle convention as
    your manualdrive.py:  steering +1 = left, -1 = right)
  * Safety: no valid command for WATCHDOG_S seconds -> throttle 0,
    and throttle is clamped to MAX_FORWARD / MAX_REVERSE no matter what
    the PC sends.

Python 3.6 compatible (JetPack 4.x). No extra pip packages needed.
NOTE: only one process can hold the camera -- stop any Jupyter notebook
that is using it before starting this.
"""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2

# ============================================================
# CONFIG
# ============================================================
CAMERA_MODE = "csi"          # "csi" (ribbon camera) or "usb"
CSI_SENSOR_ID = 0
USB_DEVICE = "/dev/video0"

WIDTH, HEIGHT, FPS = 1280, 720, 30   # keep = WIDTH/HEIGHT in the KF script
JPEG_QUALITY = 70                    # lower = less bandwidth / CPU

HTTP_PORT = 8081                     # video
CONTROL_PORT = 5005                  # UDP drive commands
AUTH_TOKEN = "change-me"             # must match the PC side

WATCHDOG_S = 0.5                     # no command for this long -> stop
MAX_FORWARD = 0.40                   # hard throttle limits on the car
MAX_REVERSE = -0.30
MAX_PULSE_S = 1.0                    # a command carrying "d" (pulse seconds) is
                                     # cut to 0 throttle by the car itself after d

# ============================================================
# SHARED STATE
# ============================================================
stop_event = threading.Event()


class FrameBuffer(object):
    """Holds the newest JPEG; every client thread wakes when it changes."""

    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg = None
        self.count = 0

    def put(self, jpeg):
        with self.cond:
            self.jpeg = jpeg
            self.count += 1
            self.cond.notify_all()

    def wait_new(self, last_count, timeout=2.0):
        with self.cond:
            if self.count == last_count:
                self.cond.wait(timeout)
            return self.jpeg, self.count


frames = FrameBuffer()


# ============================================================
# CAMERA
# ============================================================
def open_camera():
    if CAMERA_MODE == "csi":
        pipeline = (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), width=%d, height=%d, "
            "format=NV12, framerate=%d/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=%d, height=%d, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! "
            "appsink drop=1 max-buffers=1"
            % (CSI_SENSOR_ID, WIDTH, HEIGHT, FPS, WIDTH, HEIGHT)
        )
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    cap = cv2.VideoCapture(USB_DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    return cap


def camera_loop():
    cap = open_camera()
    if not cap.isOpened():
        print("[CAMERA] ERROR: could not open camera. Is another process "
              "(e.g. a Jupyter notebook) using it?  Check CAMERA_MODE.")
        stop_event.set()
        return

    print("[CAMERA] Opened (%s)" % CAMERA_MODE)
    n, t0, size_warned = 0, time.time(), False

    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue

        h, w = frame.shape[:2]
        if (w, h) != (WIDTH, HEIGHT) and not size_warned:
            print("[CAMERA] WARNING: camera gives %dx%d, not %dx%d. Set the "
                  "same size in the KF script." % (w, h, WIDTH, HEIGHT))
            size_warned = True

        ok, buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            frames.put(buf.tobytes())

        n += 1
        if time.time() - t0 >= 10.0:
            print("[CAMERA] %.1f fps" % (n / (time.time() - t0)))
            n, t0 = 0, time.time()

    cap.release()
    print("[CAMERA] Closed")


# ============================================================
# HTTP MJPEG SERVER
# ============================================================
class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):   # keep the console quiet
        pass

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/video":
            self.send_response(200)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            last = 0
            try:
                while not stop_event.is_set():
                    jpeg, count = frames.wait_new(last)
                    if jpeg is None or count == last:
                        continue
                    last = count
                    self.wfile.write(b"--FRAME\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n")
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        elif path == "/":
            body = b'<html><body style="background:#111"><img src="/video" ' \
                   b'style="max-width:100%"></body></html>'
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ============================================================
# CONTROL RECEIVER (UDP) + WATCHDOG
# ============================================================
def clip(v, lo, hi):
    return max(lo, min(hi, v))


def control_loop(car):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", CONTROL_PORT))
    sock.settimeout(0.02)
    print("[CONTROL] Listening on UDP %d" % CONTROL_PORT)

    last_rx = 0.0
    linked = False
    pulse_until = 0.0

    while not stop_event.is_set():
        try:
            data, addr = sock.recvfrom(1024)
        except socket.timeout:
            data, addr = None, None

        now = time.time()

        if data:
            try:
                msg = json.loads(data.decode("utf-8"))
                if msg.get("k") != AUTH_TOKEN:
                    raise ValueError("bad token")
                steering = clip(float(msg["s"]), -1.0, 1.0)
                throttle = clip(float(msg["t"]), MAX_REVERSE, MAX_FORWARD)
                dur = clip(float(msg.get("d", 0.0)), 0.0, MAX_PULSE_S)
            except Exception:
                continue   # ignore garbage / wrong token

            car.steering = steering
            car.throttle = throttle
            last_rx = now
            pulse_until = (now + dur) if (dur > 0 and throttle != 0) else 0.0
            if not linked:
                print("[CONTROL] Link up from %s" % addr[0])
                linked = True

        # Pulse timer: the car ends the pulse itself, independent of the network
        if pulse_until and now >= pulse_until:
            car.throttle = 0.0
            pulse_until = 0.0

        # Watchdog
        if linked and now - last_rx > WATCHDOG_S:
            car.throttle = 0.0
            car.steering = 0.0
            linked = False
            print("[CONTROL] No commands for %.1fs -> STOPPED" % WATCHDOG_S)

    car.throttle = 0.0
    car.steering = 0.0
    sock.close()


# ============================================================
# MAIN
# ============================================================
def main():
    from jetracer.nvidia_racecar import NvidiaRacecar

    car = NvidiaRacecar()
    car.throttle = 0.0
    car.steering = 0.0

    threading.Thread(target=camera_loop, daemon=True).start()
    threading.Thread(target=control_loop, args=(car,), daemon=True).start()

    server = ThreadedHTTPServer(("0.0.0.0", HTTP_PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print("=" * 60)
    print("JetRacer server running")
    print("  Video   : http://<this-jetracer-ip>:%d/video" % HTTP_PORT)
    print("  Control : UDP port %d" % CONTROL_PORT)
    print("  (find the IP with:  hostname -I)   Ctrl+C to quit")
    print("=" * 60)

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        car.throttle = 0.0
        car.steering = 0.0
        server.shutdown()
        time.sleep(0.3)
        print("Stopped.")


if __name__ == "__main__":
    main()
