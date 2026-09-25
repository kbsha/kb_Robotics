import torch
import numpy as np
import socket
import json
import time

import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Layout


# ============================================================
# CONFIGURATION
# ============================================================

JETRACER_IP = "192.168.1.100"   # <-- CHANGE THIS
JETRACER_PORT = 5005

MODEL_PATH = "two_point_model.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)


# ============================================================
# LOAD MODEL
# ============================================================

model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval()

print("Model loaded:", MODEL_PATH)


# ============================================================
# UDP CONNECTION
# ============================================================

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

JETRACER_ADDRESS = (JETRACER_IP, JETRACER_PORT)


# ============================================================
# CONTROL VARIABLES
# ============================================================

steering_gain = 1.0
steering_bias = 0.0

latest_steering = 0.0
latest_throttle = 0.0


# ============================================================
# SEND CONTROL TO JETRACER
# ============================================================

def send_control(steering, throttle):

    steering = float(np.clip(steering, -1.0, 1.0))
    throttle = float(np.clip(throttle, -1.0, 1.0))

    message = {
        "steering": steering,
        "throttle": throttle
    }

    data = json.dumps(message).encode("utf-8")

    sock.sendto(data, JETRACER_ADDRESS)


# ============================================================
# WIDGETS
# ============================================================

network_output_sliderx1 = widgets.FloatSlider(
    description="X1",
    min=-1.0,
    max=1.0,
    value=0,
    step=0.01,
    layout={"width": "350px"}
)

network_output_sliderx2 = widgets.FloatSlider(
    description="X2",
    min=-1.0,
    max=1.0,
    value=0,
    step=0.01,
    layout={"width": "350px"}
)

network_output_slidery1 = widgets.FloatSlider(
    description="Y1",
    min=-1.0,
    max=1.0,
    value=0,
    step=0.01,
    layout={"width": "350px"}
)

network_output_slidery2 = widgets.FloatSlider(
    description="Y2",
    min=-1.0,
    max=1.0,
    value=0,
    step=0.01,
    layout={"width": "350px"}
)


steering_gain_slider = widgets.FloatSlider(
    description="Steering Gain",
    min=-1.0,
    max=1.0,
    value=1.0,
    step=0.01,
    layout={"width": "300px"}
)

steering_bias_slider = widgets.FloatSlider(
    description="Steering Bias",
    min=-0.5,
    max=0.5,
    value=0.0,
    step=0.01,
    layout={"width": "300px"}
)

steering_value_slider = widgets.FloatSlider(
    description="Steering",
    min=-1.0,
    max=1.0,
    value=0.0,
    step=0.01,
    layout={"width": "350px"}
)

throttle_slider = widgets.FloatSlider(
    description="Throttle",
    min=-1.0,
    max=1.0,
    value=0.0,
    step=0.01,
    orientation="vertical",
    layout={"height": "250px"}
)


# ============================================================
# MANUAL CONTROL
# ============================================================

def manual_control(change=None):

    steering = steering_value_slider.value
    throttle = throttle_slider.value

    send_control(steering, throttle)


steering_value_slider.observe(
    manual_control,
    names="value"
)

throttle_slider.observe(
    manual_control,
    names="value"
)


# ============================================================
# DISPLAY
# ============================================================

display(
    widgets.HBox(
        [
            widgets.VBox(
                [
                    network_output_sliderx1,
                    network_output_sliderx2,
                    network_output_slidery1,
                    network_output_slidery2,

                    widgets.HTML("<hr>"),

                    steering_gain_slider,
                    steering_bias_slider,

                    widgets.HTML("<hr>"),

                    steering_value_slider
                ]
            ),

            throttle_slider
        ],
        layout=Layout(align_items="center")
    )
)
