"""
PC SIDE  --  jetracer_link.py   (put next to the KF script)

Sends steering/throttle to jetracer_server.py over UDP.

Modes (keys work in the OpenCV video window, which must be focused):
    x or SPACE : STOP  (default at start, sends 0/0)
    g          : AUTO   -> car follows the KF fast layer (steering + throttle)
    m          : MANUAL -> keyboard, same feel as manualdrive.py
                    w / s : throttle +0.06 / -0.06
                    a / d : steer left / right (0.15 steps)
                    c     : center steering
                    b     : brake (throttle = 0)

Safety built in:
    * starts STOPPED; nothing moves until you press g or m
    * AUTO is forced to 0 throttle if the video frame is stale (>0.5 s
      old) or if the KF script is on its fallback video instead of the car
    * the JetRacer itself also stops if commands stop arriving
"""

import json
import socket
import time


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


class JetRacerLink(object):
    STOP, AUTO, MANUAL = "STOP", "AUTO", "MANUAL"

    def __init__(self,
                 ip,
                 port=5005,
                 token="change-me",          # same as AUTH_TOKEN on the car
                 max_steering_deg=30.0,      # = MAX_STEERING_DEG in KF script
                 steering_sign=-1.0,         # flip to +1.0 if the car steers the WRONG way in AUTO
                 auto_throttle_gain=1.0,     # multiply KF throttle (0..0.3)
                 auto_throttle_max=0.35,     # cap for AUTO
                 manual_throttle_max=0.40,
                 manual_throttle_step=0.06,  # from manualdrive.py
                 manual_steering_step=0.15,  # from manualdrive.py
                 send_hz=30.0):
        self.addr = (ip, port)
        self.token = token
        self.max_steering_deg = max_steering_deg
        self.steering_sign = steering_sign
        self.auto_throttle_gain = auto_throttle_gain
        self.auto_throttle_max = auto_throttle_max
        self.manual_throttle_max = manual_throttle_max
        self.manual_throttle_step = manual_throttle_step
        self.manual_steering_step = manual_steering_step
        self.min_period = 1.0 / send_hz

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.mode = self.STOP
        self.man_throttle = 0.0
        self.man_steering = 0.0
        self.sent_steering = 0.0
        self.sent_throttle = 0.0
        self.note = ""
        self._last_send = 0.0

    # ---------------------------------------------------------
    def handle_key(self, key):
        """Pass the value of cv2.waitKey(1) & 0xFF."""
        if key == 255 or key < 0:
            return

        if key in (ord("x"), ord(" ")):
            self.mode = self.STOP
            self.man_throttle = self.man_steering = 0.0
        elif key == ord("g"):
            self.mode = self.AUTO
        elif key == ord("m"):
            self.mode = self.MANUAL
            self.man_throttle = self.man_steering = 0.0
        elif self.mode == self.MANUAL:
            if key == ord("w"):
                self.man_throttle = min(self.man_throttle + self.manual_throttle_step,
                                        self.manual_throttle_max)
            elif key == ord("s"):
                self.man_throttle = max(self.man_throttle - self.manual_throttle_step,
                                        -self.manual_throttle_max)
            elif key == ord("b"):
                self.man_throttle = 0.0
            elif key == ord("a"):      # left  (+ on the car, like KEY_LEFT in manualdrive.py)
                self.man_steering = min(self.man_steering + self.manual_steering_step, 1.0)
            elif key == ord("d"):      # right
                self.man_steering = max(self.man_steering - self.manual_steering_step, -1.0)
            elif key == ord("c"):
                self.man_steering = 0.0

    # ---------------------------------------------------------
    def update(self, steering_deg, throttle, frame_ok=True):
        """
        Call once per loop with the KF fast-layer output.
        frame_ok=False (stale/fallback video) forces AUTO to stop.
        Returns the (steering, throttle) actually sent.
        """
        self.note = ""
        if self.mode == self.STOP:
            s, t = 0.0, 0.0

        elif self.mode == self.AUTO:
            if not frame_ok:
                s, t = 0.0, 0.0
                self.note = "NO LIVE VIDEO - HOLDING STOP"
            else:
                s = _clip(self.steering_sign * steering_deg / self.max_steering_deg, -1.0, 1.0)
                t = _clip(throttle * self.auto_throttle_gain, 0.0, self.auto_throttle_max)

        else:  # MANUAL
            s, t = self.man_steering, self.man_throttle

        self._send(s, t)
        return s, t

    def _send(self, steering, throttle, force=False):
        now = time.time()
        if not force and now - self._last_send < self.min_period:
            return
        self._last_send = now
        self.sent_steering, self.sent_throttle = steering, throttle
        msg = json.dumps({"k": self.token, "s": round(steering, 4), "t": round(throttle, 4)})
        try:
            self.sock.sendto(msg.encode("utf-8"), self.addr)
        except OSError as e:
            self.note = "SEND ERROR: %s" % e

    # ---------------------------------------------------------
    def draw_status(self, display):
        import cv2
        color = {"STOP": (0, 0, 255), "AUTO": (0, 255, 0), "MANUAL": (0, 200, 255)}[self.mode]
        h = display.shape[0]
        text = "JETRACER: %s  steer %+.2f  thr %+.2f" % (
            self.mode, self.sent_steering, self.sent_throttle)
        cv2.putText(display, text, (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        hint = "[g] auto  [m] manual  [x/space] STOP   manual: w/s a/d c b"
        cv2.putText(display, hint, (15, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        if self.note:
            cv2.putText(display, self.note, (15, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    def stop(self):
        """Send a few stop packets and close. Call on shutdown."""
        self.mode = self.STOP
        for _ in range(5):
            self._send(0.0, 0.0, force=True)
            time.sleep(0.02)
        self.sock.close()
