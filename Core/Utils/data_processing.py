import numpy as np

# Decompose wind speed and wind direction into u and v components of wind speed
def calculate_u(speed, direction):
    u = -speed * np.sin(np.radians(direction))
    return u


def calculate_v(speed, direction):
    v = -speed * np.cos(np.radians(direction))
    return v
