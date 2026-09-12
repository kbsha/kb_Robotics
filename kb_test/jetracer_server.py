"""
============================================================
JETRACER SERVER -- runs ON the car, later
============================================================
Copy this file onto the JetRacer's Jetson once you have it. It is
the ONLY thing in the remote setup that writes to NvidiaRacecar, and
it does not trust the network to keep it safe: if commands from
drive_brain.py (running on your laptop, MODE="remote_car") stop
arriving, or an explicit e_stop packet arrives, it forces the motors
to zero immediately and stays there until a fresh, unstopped command
arrives. A dropped Wi-Fi connection defaults to "stopped", not
"keep doing the last thing".

Run it with:  python3 jetracer_server.py
It prints its own IP-independent status; point drive_brain.py's
ROBOT_HOST at whatever IP this machine has on your network.

Without `jetracer` installed (e.g. testing this file itself on a
laptop) it runs in dry-run and just prints what it would send to the
motors.
============================================================
"""

import atexit
import json
import signal
import socket
import sys
import time

try:
    from jetracer.nvidia_racecar import NvidiaRacecar
    JETRACER_AVAILABLE = True
except Exception:
    JETRACER_AVAILABLE = False

try:
    import Jetson.GPIO as GPIO
    GPIO_AVAILABLE = True
except Exception:
    GPIO_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 5555

# Hard ceiling enforced HERE too, independent of whatever the brain
# sends -- never trust a single side of a network link with the only
# copy of a safety limit.
MAX_THROTTLE = 0.15
MAX_STEERING = 1.0

# Chassis trim -- adjust for your specific car.
STEERING_GAIN = 1.0
STEERING_OFFSET = 0.0
THROTTLE_GAIN = 1.0

# If no valid command arrives within this long, force stop. This is
# the real safety net for remote driving -- packet loss, laptop
# closed, Wi-Fi out of range, all default to "stopped".
COMMAND_TIMEOUT_S = 0.35

E_STOP_GPIO_PIN = None   # e.g. 17, wired to GND, for a physical button


# ============================================================
# CAR
# ============================================================

class SafeCar:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run or not JETRACER_AVAILABLE
        self._car = None
        self._latch = False
        if self.dry_run:
            print("[CAR] DRY RUN -- jetracer library not found, printing only")
        else:
            self._car = NvidiaRacecar()
            self._car.steering_gain = STEERING_GAIN
            self._car.steering_offset = STEERING_OFFSET
            self._car.throttle_gain = THROTTLE_GAIN
            self._car.steering = 0.0
            self._car.throttle = 0.0
            print("[CAR] NvidiaRacecar initialized")

    def apply(self, steering, throttle):
        if self._latch:
            steering, throttle = 0.0, 0.0
        steering = max(-MAX_STEERING, min(MAX_STEERING, steering))
        throttle = max(0.0, min(MAX_THROTTLE, throttle))
        if self.dry_run:
            print(f"\r[CAR] steer={steering:+.2f} throttle={throttle:.2f}   ", end="", flush=True)
        else:
            try:
                self._car.steering = steering
                self._car.throttle = throttle
            except Exception as e:
                print("[CAR] write failed, forcing stop:", e)
                self.stop(latch=True)

    def stop(self, latch=False):
        if latch:
            self._latch = True
        if not self.dry_run and self._car is not None:
            try:
                self._car.throttle = 0.0
                self._car.steering = 0.0
            except Exception:
                pass

    def clear_latch(self):
        self._latch = False

    @property
    def latched(self):
        return self._latch


def install_shutdown_handlers(car):
    def _force_stop(*_):
        car.stop(latch=True)
    atexit.register(_force_stop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda s, f: (_force_stop(), sys.exit(0)))
        except Exception:
            pass


def gpio_estop_watch(get_flag_setter):
    if not (GPIO_AVAILABLE and E_STOP_GPIO_PIN is not None):
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(E_STOP_GPIO_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    while True:
        if GPIO.input(E_STOP_GPIO_PIN) == GPIO.LOW:
            get_flag_setter(True)
        time.sleep(0.05)


# ============================================================
# MAIN
# ============================================================

def main():
    car = SafeCar()
    install_shutdown_handlers(car)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LISTEN_HOST, LISTEN_PORT))
    sock.settimeout(0.05)

    print(f"[SERVER] listening on {LISTEN_HOST}:{LISTEN_PORT} "
          f"dry_run={car.dry_run} max_throttle={MAX_THROTTLE} "
          f"command_timeout={COMMAND_TIMEOUT_S}s")

    last_cmd_t = 0.0
    e_stop_latched = False

    def set_estop(v):
        nonlocal e_stop_latched
        e_stop_latched = v

    import threading
    threading.Thread(target=gpio_estop_watch, args=(set_estop,), daemon=True).start()

    try:
        while True:
            now = time.time()

            try:
                data, addr = sock.recvfrom(2048)
                msg = json.loads(data.decode("utf-8"))
                if msg.get("e_stop"):
                    e_stop_latched = True
                    print("\n[SERVER] E-STOP received from", addr)
                elif not e_stop_latched:
                    last_cmd_t = now
                    car.apply(float(msg.get("steering", 0.0)), float(msg.get("throttle", 0.0)))
            except socket.timeout:
                pass
            except Exception as e:
                print("\n[SERVER] bad packet, ignoring:", e)

            stale = (now - last_cmd_t) > COMMAND_TIMEOUT_S
            if e_stop_latched or stale:
                car.stop(latch=e_stop_latched)

    except KeyboardInterrupt:
        pass
    finally:
        car.stop(latch=True)
        sock.close()
        print("\n[SERVER] Shutdown complete, motors zeroed.")


if __name__ == "__main__":
    main()
