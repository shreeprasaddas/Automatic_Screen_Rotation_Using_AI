import subprocess

# Rotation constants
DMDO_DEFAULT = "normal"
DMDO_90 = "left"
DMDO_180 = "inverted"
DMDO_270 = "right"

def set_display_rotation(orientation, display="eDP-1"):
    try:
        result = subprocess.run(
            ["xrandr", "--output", display, "--rotate", orientation],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        print(f"✅ Rotated screen to {orientation}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to rotate screen: {e.stderr}")
