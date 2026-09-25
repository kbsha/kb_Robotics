import torch
import torch.nn as nn
import torchvision.models as models
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

# Point to your standard pth model, NOT the trt one
MODEL_PATH = "/home/kibremogesfenta/Desktop/KB-Fenta-Personal/My PhD Research/kb_robotics2024/kb_Robotics/kb_test/final tesst/files (6)/local renet test/two_point_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


# ============================================================
# INITIALIZE & LOAD PYTORCH RESNET18 ARCHITECTURE
# ============================================================

try:
    # 1. Instantiate a standard ResNet18 framework (weights=None because we load yours)
    # Using legacy weights argument for compatibility with older checkpoints
    model = models.resnet18(pretrained=False)
    
    # 2. Modify the final fully connected layer to output 4 coordinate regression values (X1, Y1, X2, Y2)
    model.fc = nn.Linear(model.fc.in_features, 4)
    
    # 3. Load your model weights checkpoint 
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    # Parse whether it's stored inside a state_dict wrapper or direct weights
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        # If it was saved as a whole model object, extract its state_dict
        model.load_state_dict(checkpoint.state_dict())

    model = model.to(DEVICE)
    model.eval()
    print("Standard PyTorch ResNet18 model and weights loaded successfully!")

except Exception as e:
    print(f"Failed to load architecture because: {e}")
    print("If this fails, your checkpoint file may be using a different backbone than standard ResNet18.")


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

network_output_sliderx1 = widgets.FloatSlider(description="X1", min=-1.0, max=1.0, value=0, step=0.01, layout={"width": "350px"})
network_output_sliderx2 = widgets.FloatSlider(description="X2", min=-1.0, max=1.0, value=0, step=0.01, layout={"width": "350px"})
network_output_slidery1 = widgets.FloatSlider(description="Y1", min=-1.0, max=1.0, value=0, step=0.01, layout={"width": "350px"})
network_output_slidery2 = widgets.FloatSlider(description="Y2", min=-1.0, max=1.0, value=0, step=0.01, layout={"width": "350px"})

steering_gain_slider = widgets.FloatSlider(description="Steering Gain", min=-1.0, max=1.0, value=1.0, step=0.01, layout={"width": "300px"})
steering_bias_slider = widgets.FloatSlider(description="Steering Bias", min=-0.5, max=0.5, value=0.0, step=0.01, layout={"width": "300px"})
steering_value_slider = widgets.FloatSlider(description="Steering", min=-1.0, max=1.0, value=0.0, step=0.01, layout={"width": "350px"})
throttle_slider = widgets.FloatSlider(description="Throttle", min=-1.0, max=1.0, value=0.0, step=0.01, orientation="vertical", layout={"height": "250px"})


# ============================================================
# MANUAL CONTROL
# ============================================================

def manual_control(change=None):
    steering = steering_value_slider.value
    throttle = throttle_slider.value
    send_control(steering, throttle)

steering_value_slider.observe(manual_control, names="value")
throttle_slider.observe(manual_control, names="value")


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
