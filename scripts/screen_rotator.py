import os
import platform

def rotate_screen(yaw, pitch, roll):
    if abs(pitch) > 25:
        orientation = "portrait" if pitch > 0 else "reverse portrait"
    elif abs(yaw) > 25:
        orientation = "landscape" if yaw > 0 else "reverse landscape"
    else:
        orientation = "default"

    print(f"Rotating to: {orientation} (yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f})")
    # Placeholder: Implement real rotation using OS APIs if needed.