"""
PC SIDE  --  jetracer_link.py   (put next to the KF script)

Sends steering/throttle to jetracer_server.py over UDP.

WHY PULSES
    The video arrives late, so a continuous drive command is always acting
    on a picture that is already old.  In AUTO the car therefore works in
    a stop-and-go cycle:

        SETTLE  car is stopped, wait for the body to stop + video to catch up
        SENSE   collect a few FRESH frames, take the median KF answer
                (if the KF says "no safe corridor" -> keep waiting, no move)
        STEER   turn the wheels while still stopped (steering lead)
        MOVE    one short, fixed throttle pulse with the latched steering
        ... back to SETTLE

    Nothing is re-decided in the middle of a pulse, so the delay no longer
    matters.  Press [p] to switch back to continuous driving to compare.

Modes (keys work in the OpenCV video window, which must be focused):
    x or SPACE : STOP  (default at start)
    g          : AUTO   -> pulse cycle driven by the KF fast layer
    m          : MANUAL -> keyboard
                    w / s : throttle +0.06 / -0.06      i / k : +0.01 / -0.01
                    a / d : steer left / right (0.15)   j / l : +0.02 / -0.02
                    c     : center steering (= trim)    b     : brake (0)
    p          : toggle AUTO between PULSE and CONTINUOUS
    o          : open / close the CALIBRATION window (fine sliders)
    t          : one TEST PULSE (STOP or MANUAL only) using the sliders
    v          : save calibration to jetracer_calib.json (also saved on exit)

Calibration (sliders, live):
    steer trim      wheel angle that drives the car dead straight
    steer gain      overall steering strength
    steer expo      >1 = softer around center (small errors -> tiny steering)
    steer deadband  KF steering below this many degrees is ignored
    steer limit     max steering magnitude in AUTO
    thr gain / min / max   AUTO throttle scaling, motor start threshold, cap
    pulse / settle / steer lead (ms)   timing of the stop-and-go cycle
    test thr        throttle used by the [t] test pulse

Safety built in:
    * starts STOPPED; nothing moves until you press g or m
    * AUTO returns to STOP-and-wait if the video frame is stale (>0.5 s old)
      or the KF script is on its fallback video instead of the car
    * every pulse is time-limited on the PC AND (with the updated
      jetracer_server.py) on the car itself
    * the JetRacer also stops if commands stop arriving
"""

import json
import math
import os
import socket
import time


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _median(vals):
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


CALIB_WINDOW = "JetRacer calibration"

# key, slider label, lo, hi (integer slider range), scale (value = int * scale)
_SLIDERS = [
    ("steering_trim",         "steer trim x.001",      -300,  300, 0.001),
    ("steering_gain",         "steer gain x.01",         10, 1000, 0.01),
    ("steering_expo",         "steer expo x.01",        100,  250, 0.01),
    ("steering_deadband_deg", "steer deadband x.1deg",    0,   50, 0.1),
    ("steering_max",          "steer limit x.01",        30,  100, 0.01),
    ("throttle_gain",         "thr gain x.01",           20,  200, 0.01),
    ("throttle_min",          "thr min x.001",            0,  400, 0.001),
    ("throttle_max",          "thr max x.001",           50,  400, 0.001),
    ("pulse_ms",              "pulse ms",                50, 1000, 1.0),
    ("settle_ms",             "settle ms",              100, 1500, 1.0),
    ("steer_lead_ms",         "steer lead ms",            0,  600, 1.0),
    ("test_throttle",         "test thr x.001",           0,  400, 0.001),
]
_SLIDER_INFO = dict((k, (lo, hi, sc)) for k, _l, lo, hi, sc in _SLIDERS)


class JetRacerLink(object):
    STOP, AUTO, MANUAL = "STOP", "AUTO", "MANUAL"
    SETTLE, SENSE, STEER, MOVE = "SETTLE", "SENSE", "STEER", "MOVE"

    def __init__(self,
                 ip,
                 port=5005,
                 token="change-me",          # same as AUTH_TOKEN on the car
                 max_steering_deg=30.0,      # = MAX_STEERING_DEG in KF script
                 steering_sign=-1.0,         # flip to +1.0 if the car steers the WRONG way in AUTO
                 auto_throttle_gain=1.0,     # default for the "thr gain" slider
                 auto_throttle_max=0.35,     # default for the "thr max" slider
                 manual_throttle_max=0.40,
                 manual_throttle_step=0.06,  # from manualdrive.py
                 manual_steering_step=0.15,  # from manualdrive.py
                 send_hz=30.0,
                 pulse_mode=True,            # AUTO = stop-and-go (False = continuous)
                 sense_frames=3,             # fresh frames needed before each pulse
                 calib_path=None):
        self.addr = (ip, port)
        self.token = token
        self.max_steering_deg = max_steering_deg
        self.steering_sign = steering_sign
        self.manual_throttle_max = manual_throttle_max
        self.manual_throttle_step = manual_throttle_step
        self.manual_steering_step = manual_steering_step
        self.min_period = 1.0 / send_hz
        self.pulse_mode = pulse_mode
        self.sense_frames = max(1, int(sense_frames))

        # ---- calibration (tunable live, saved to disk) ----
        self.cal = {
            "steering_trim": 0.0,
            "steering_gain": 1.0,
            "steering_expo": 1.3,
            "steering_deadband_deg": 1.0,
            "steering_max": 1.0,
            "throttle_gain": auto_throttle_gain,
            "throttle_min": 0.0,
            "throttle_max": auto_throttle_max,
            "pulse_ms": 250.0,
            "settle_ms": 450.0,
            "steer_lead_ms": 200.0,
            "test_throttle": 0.15,
        }
        if calib_path is None:
            calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "jetracer_calib.json")
        self.calib_path = calib_path
        self.load_calibration()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.mode = self.STOP
        self.man_throttle = 0.0
        self.man_steering = 0.0
        self.sent_steering = self.cal["steering_trim"]
        self.sent_throttle = 0.0
        self.note = ""
        self._last_send = 0.0
        self._force = False

        # pulse-cycle state
        self.phase = self.SETTLE
        self._t_phase = time.time()
        self._samples = []
        self._last_ft = None
        self._lat_s = self.cal["steering_trim"]     # latched steering command
        self._lat_t = 0.0                           # latched throttle command
        self._steer_cmd = self.cal["steering_trim"]  # last steering we commanded
        self.pulse_count = 0

        # obstacle override
        self._obstacle = False

        # raw-mode flag (set/cleared inside update_raw)
        self._raw_mode = False

        # test pulse
        self._test_end = 0.0

        # calibration window
        self._ui_open = False
        self._ui_last_draw = 0.0

    # =========================================================
    # calibration: load / save / mapping
    # =========================================================
    def load_calibration(self):
        try:
            with open(self.calib_path, "r") as f:
                data = json.load(f)
        except (IOError, OSError, ValueError):
            return
        for k in self.cal:
            if k in data and isinstance(data[k], (int, float)):
                lo, hi, sc = _SLIDER_INFO[k]
                self.cal[k] = _clip(float(data[k]), lo * sc, hi * sc)

    def save_calibration(self):
        try:
            with open(self.calib_path, "w") as f:
                json.dump(self.cal, f, indent=2, sort_keys=True)
            self.note = "CALIBRATION SAVED"
            return True
        except (IOError, OSError) as e:
            self.note = "SAVE ERROR: %s" % e
            return False

    def map_steering(self, steering_deg):
        """KF degrees -> calibrated car steering command (-1..1)."""
        c = self.cal
        x = self.steering_sign * steering_deg
        db = c["steering_deadband_deg"]
        mag = abs(x)
        if mag <= db:
            return _clip(c["steering_trim"], -1.0, 1.0)
        mag = min(1.0, (mag - db) / max(self.max_steering_deg - db, 1e-6))
        y = min(mag ** c["steering_expo"] * c["steering_gain"], c["steering_max"])
        y = math.copysign(y, x)
        return _clip(y + c["steering_trim"], -1.0, 1.0)

    def map_throttle(self, kf_throttle):
        """KF throttle (0..0.3) -> calibrated car throttle."""
        if kf_throttle <= 0.0:
            return 0.0
        c = self.cal
        t = max(kf_throttle * c["throttle_gain"], c["throttle_min"])
        return _clip(t, 0.0, c["throttle_max"])

    # =========================================================
    # keys
    # =========================================================
    def handle_key(self, key):
        """Pass the value of cv2.waitKey(1) & 0xFF."""
        if key == 255 or key < 0:
            return
        trim = self.cal["steering_trim"]

        if key in (ord("x"), ord(" ")):
            self.mode = self.STOP
            self.man_throttle = self.man_steering = 0.0
            self._test_end = 0.0
            self._force = True
        elif key == ord("g"):
            if self.mode != self.AUTO:
                self._enter(self.SETTLE, time.time())
                self._steer_cmd = self.sent_steering
            self.mode = self.AUTO
        elif key == ord("m"):
            self.mode = self.MANUAL
            self.man_throttle = self.man_steering = 0.0
        elif key == ord("p"):
            self.pulse_mode = not self.pulse_mode
            self._enter(self.SETTLE, time.time())
        elif key == ord("o"):
            self.toggle_calibration_window()
        elif key == ord("v"):
            self.save_calibration()
        elif key == ord("t"):
            if self.mode == self.AUTO:
                self.note = "TEST PULSE: leave AUTO first (x or m)"
            else:
                self.man_throttle = 0.0
                self._test_end = time.time() + self.cal["pulse_ms"] / 1000.0
        elif self.mode == self.MANUAL:
            if key == ord("w"):
                self.man_throttle = min(self.man_throttle + self.manual_throttle_step,
                                        self.manual_throttle_max)
            elif key == ord("s"):
                self.man_throttle = max(self.man_throttle - self.manual_throttle_step,
                                        -self.manual_throttle_max)
            elif key == ord("i"):
                self.man_throttle = min(self.man_throttle + 0.01, self.manual_throttle_max)
            elif key == ord("k"):
                self.man_throttle = max(self.man_throttle - 0.01, -self.manual_throttle_max)
            elif key == ord("b"):
                self.man_throttle = 0.0
            elif key == ord("a"):      # left (+ on the car, like KEY_LEFT in manualdrive.py)
                self.man_steering = min(self.man_steering + self.manual_steering_step, 1.0)
            elif key == ord("d"):      # right
                self.man_steering = max(self.man_steering - self.manual_steering_step, -1.0)
            elif key == ord("j"):
                self.man_steering = min(self.man_steering + 0.02, 1.0)
            elif key == ord("l"):
                self.man_steering = max(self.man_steering - 0.02, -1.0)
            elif key == ord("c"):
                self.man_steering = 0.0

    # =========================================================
    # main entry: call once per loop
    # =========================================================
    def update(self, steering_deg, throttle, frame_ok=True, frame_time=None, obstacle=False):
        """
        steering_deg, throttle : KF fast-layer output
        frame_ok               : False (stale/fallback video) forces AUTO to stop
        frame_time             : time the newest video frame arrived (optional)
        obstacle               : True = black obstacle close ahead; forces the car
                                 to stop and stay in SETTLE until the path clears,
                                 then re-evaluates before the next pulse.
        Returns the (steering, throttle) actually sent.
        """
        self._obstacle = obstacle
        now = time.time()
        self.note = ""
        self._poll_calibration_window(now)
        trim = self.cal["steering_trim"]
        dur = 0.0

        if self.mode == self.STOP:
            s, t = trim, 0.0
        elif self.mode == self.AUTO:
            s, t, dur = self._auto(now, steering_deg, throttle, frame_ok, frame_time)
        else:  # MANUAL
            s = _clip(self.man_steering + trim, -1.0, 1.0)
            t = self.man_throttle

        # manual test pulse (STOP / MANUAL only)
        if self._test_end and self.mode != self.AUTO:
            if now < self._test_end:
                t = self.cal["test_throttle"]
                dur = self._test_end - now
                self.note = "TEST PULSE"
            else:
                self._test_end = 0.0
                self._force = True

        self._steer_cmd = s
        self._send(s, t, force=self._force, duration=dur)
        self._force = False
        return s, t

    def update_raw(self, steering, throttle, frame_ok=True, frame_time=None, obstacle=False):
        """
        Like update() but steering (-1..1) and throttle are already in car units
        (e.g. from MPC).  Steering trim is still added; throttle is still capped
        by cal['throttle_max'].  The pulse cycle and obstacle override run as normal.
        """
        self._raw_mode = True
        result = self.update(steering, throttle,
                             frame_ok=frame_ok, frame_time=frame_time, obstacle=obstacle)
        self._raw_mode = False
        return result

    # ---------------------------------------------------------
    def _enter(self, phase, now):
        self.phase = phase
        self._t_phase = now
        if phase == self.SETTLE:
            self._samples = []
            self._force = True      # make sure the stop packet goes out now

    def _auto(self, now, steering_deg, throttle, frame_ok, frame_time):
        """Returns (steering, throttle, pulse_duration_for_server)."""
        c = self.cal

        # -- obstacle override: highest priority -----------------
        if self._obstacle:
            if self.phase != self.SETTLE:
                self._enter(self.SETTLE, now)   # abort any active pulse immediately
            self._t_phase = now                 # keep resetting settle timer until clear
            self.note = "!!! OBSTACLE STOP !!!"
            return self._steer_cmd, 0.0, 0.0

        if not frame_ok:
            if self.phase != self.SETTLE:
                self._enter(self.SETTLE, now)
            self._t_phase = now       # restart the settle timer while video is bad
            self.note = "NO LIVE VIDEO - HOLDING STOP"
            return self._steer_cmd, 0.0, 0.0

        # ---------------- continuous mode (old behaviour, calibrated) --------
        if not self.pulse_mode:
            if self._raw_mode:
                s = _clip(steering_deg * c["steering_gain"] + c["steering_trim"], -1.0, 1.0)
                t = _clip(throttle, 0.0, c["throttle_max"])
            else:
                s, t = self.map_steering(steering_deg), self.map_throttle(throttle)
            return s, t, 0.0

        # ---------------- stop-and-go pulse cycle ----------------------------
        if self.phase == self.SETTLE:
            if now - self._t_phase >= c["settle_ms"] / 1000.0:
                self._enter(self.SENSE, now)
                self._last_ft = None
            return self._steer_cmd, 0.0, 0.0

        if self.phase == self.SENSE:
            new_frame = frame_time is None or frame_time != self._last_ft
            fresh = frame_time is None or frame_time > self._t_phase
            if new_frame and fresh:
                self._last_ft = frame_time
                self._samples.append((steering_deg, throttle))
                self._samples = self._samples[-self.sense_frames:]

            n = len(self._samples)
            if n >= self.sense_frames:
                if min(t for _s, t in self._samples) <= 0.0:
                    self.note = "NO SAFE CORRIDOR - WAITING"
                else:
                    kf_s = _median([s for s, _t in self._samples])
                    kf_t = _median([t for _s, t in self._samples])
                    if self._raw_mode:
                        self._lat_s = _clip(kf_s * c["steering_gain"] + c["steering_trim"], -1.0, 1.0)
                        self._lat_t = _clip(kf_t, 0.0, c["throttle_max"])
                    else:
                        self._lat_s = self.map_steering(kf_s)
                        self._lat_t = self.map_throttle(kf_t)
                    self._enter(self.STEER, now)
                    if self._lat_t <= 0.0:
                        self._enter(self.SETTLE, now)
            else:
                self.note = "SENSING %d/%d" % (n, self.sense_frames)
            return self._steer_cmd, 0.0, 0.0

        if self.phase == self.STEER:
            if now - self._t_phase >= c["steer_lead_ms"] / 1000.0:
                self._enter(self.MOVE, now)
                self._force = True
                return self._lat_s, self._lat_t, c["pulse_ms"] / 1000.0
            return self._lat_s, 0.0, 0.0

        # MOVE
        pulse = c["pulse_ms"] / 1000.0
        if now - self._t_phase >= pulse:
            self.pulse_count += 1
            self._enter(self.SETTLE, now)
            return self._lat_s, 0.0, 0.0
        return self._lat_s, self._lat_t, pulse - (now - self._t_phase)

    # =========================================================
    # UDP
    # =========================================================
    def _send(self, steering, throttle, force=False, duration=0.0):
        now = time.time()
        if not force and now - self._last_send < self.min_period:
            return
        self._last_send = now
        self.sent_steering, self.sent_throttle = steering, throttle
        pkt = {"k": self.token, "s": round(steering, 4), "t": round(throttle, 4)}
        if throttle != 0.0 and duration > 0.0:
            pkt["d"] = round(duration, 3)      # server also cuts the pulse itself
        msg = json.dumps(pkt)
        try:
            self.sock.sendto(msg.encode("utf-8"), self.addr)
            if force and throttle == 0.0:      # a lost stop packet is the one that hurts
                self.sock.sendto(msg.encode("utf-8"), self.addr)
        except OSError as e:
            self.note = "SEND ERROR: %s" % e

    # =========================================================
    # calibration window (OpenCV trackbars)
    # =========================================================
    def toggle_calibration_window(self):
        import cv2
        if self._ui_open:
            cv2.destroyWindow(CALIB_WINDOW)
            self._ui_open = False
            return
        cv2.namedWindow(CALIB_WINDOW, cv2.WINDOW_NORMAL)
        for key, label, lo, hi, sc in _SLIDERS:
            pos = int(round(self.cal[key] / sc)) - lo
            cv2.createTrackbar(label, CALIB_WINDOW, _clip(pos, 0, hi - lo), hi - lo,
                               lambda _v: None)
        self._ui_open = True
        self._ui_last_draw = 0.0

    def _poll_calibration_window(self, now):
        if not self._ui_open:
            return
        import cv2
        try:
            if cv2.getWindowProperty(CALIB_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                self._ui_open = False        # user closed it with the X
                return
            for key, label, lo, hi, sc in _SLIDERS:
                pos = cv2.getTrackbarPos(label, CALIB_WINDOW)
                if pos >= 0:
                    self.cal[key] = round((pos + lo) * sc, 4)
        except cv2.error:
            self._ui_open = False
            return
        if now - self._ui_last_draw >= 0.1:
            self._ui_last_draw = now
            self._draw_calibration_panel()

    def _draw_calibration_panel(self):
        import cv2
        import numpy as np
        c = self.cal
        lines = [
            ("STEERING", None),
            ("trim      %+.3f   (wheels straight = car drives straight)" % c["steering_trim"], 0),
            ("gain      %.2f" % c["steering_gain"], 0),
            ("expo      %.2f    (>1 softer near center)" % c["steering_expo"], 0),
            ("deadband  %.1f deg" % c["steering_deadband_deg"], 0),
            ("limit     %.2f" % c["steering_max"], 0),
            ("THROTTLE (AUTO)", None),
            ("gain %.2f   min %.3f   max %.3f" % (c["throttle_gain"], c["throttle_min"], c["throttle_max"]), 0),
            ("PULSE CYCLE", None),
            ("pulse %d ms  settle %d ms  steer lead %d ms" %
             (c["pulse_ms"], c["settle_ms"], c["steer_lead_ms"]), 0),
            ("TEST", None),
            ("[t] pulse at throttle %.3f for %d ms" % (c["test_throttle"], c["pulse_ms"]), 0),
            ("[v] save   [o] close   [t] only in STOP/MANUAL", 1),
        ]
        img = np.full((30 + 24 * len(lines), 560, 3), 25, np.uint8)
        y = 24
        for text, kind in lines:
            if kind is None:
                cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
            else:
                col = (170, 170, 170) if kind == 1 else (235, 235, 235)
                cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            y += 24
        cv2.imshow(CALIB_WINDOW, img)

    # =========================================================
    # HUD
    # =========================================================
    def draw_status(self, display):
        import cv2
        color = {"STOP": (0, 0, 255), "AUTO": (0, 255, 0), "MANUAL": (0, 200, 255)}[self.mode]
        text = "JETRACER: %s  steer %+.2f  thr %+.2f" % (
            self.mode, self.sent_steering, self.sent_throttle)
        cv2.putText(display, text, (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        hint = "[g] auto  [m] manual  [x/space] STOP  [p] pulse/cont  [o] calib  [t] test  [v] save"
        cv2.putText(display, hint, (15, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        if self.mode == self.AUTO:
            if self.pulse_mode:
                el = time.time() - self._t_phase
                info = "PULSE %s  %.2fs  pulses=%d" % (self.phase, el, self.pulse_count)
            else:
                info = "CONTINUOUS (calibrated)"
            cv2.putText(display, info, (15, 166), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 120), 1)
        if self.note:
            cv2.putText(display, self.note, (15, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    def stop(self):
        """Send a few stop packets, save calibration, close. Call on shutdown."""
        self.mode = self.STOP
        self._test_end = 0.0
        if self._ui_open:
            try:
                import cv2
                cv2.destroyWindow(CALIB_WINDOW)
            except Exception:
                pass
            self._ui_open = False
        self.save_calibration()
        for _ in range(5):
            self._send(self.cal["steering_trim"], 0.0, force=True)
            time.sleep(0.02)
        self.sock.close()
