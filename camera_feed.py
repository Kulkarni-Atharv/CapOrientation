"""
camera_feed.py  –  RPi Global Shutter Camera – natural colour live feed
Run: python camera_feed.py

Controls
--------
SPACE  → capture burst (5 frames saved to Desktop)
ESC    → quit
"""

from picamera2 import Picamera2
import cv2
import time
import os

# ── Output folder ──────────────────────────────────────────────────────────
desktop_path = os.path.expanduser("~/Desktop")

# ── Camera init ─────────────────────────────────────────────────────────────
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    controls={
        "FrameRate":  30,        # 30 fps lets sensor expose longer → brighter
        "AeEnable":   True,      # Auto Exposure ON
        "AwbEnable":  True,      # Auto White Balance ON
        "AeMeteringMode": 0,     # 0 = CentreWeighted metering
    }
)

picam2.configure(config)
picam2.start()

# Let AE + AWB fully converge before showing the feed
print("Warming up camera (2 s) ...")
time.sleep(2)

# ⚠️  DO NOT set manual ExposureTime / AnalogueGain here.
#     Forcing ExposureTime=1000 µs made the image pitch dark.
#     Auto-exposure picks ~18 000 µs on its own – exactly like rpicam-hello.

count = 1
print("Camera ready.  Press SPACE to capture burst | ESC to quit")

while True:
    frame = picam2.capture_array()

    # ✅ Picamera2 RGB888 → OpenCV BGR (fixes colour swap)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow("Live Feed – CapsuleVision", bgr)
    key = cv2.waitKey(1) & 0xFF

    # ── Burst capture ────────────────────────────────────────────────
    if key == 32:   # SPACE
        print("Capturing burst ...")
        for i in range(5):
            f     = picam2.capture_array()
            f_bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            fname = os.path.join(desktop_path, f"capture_{count:04d}.jpg")
            cv2.imwrite(fname, f_bgr)
            print(f"  Saved: {fname}")
            count += 1
            time.sleep(0.05)

    elif key == 27:  # ESC
        break

cv2.destroyAllWindows()
picam2.stop()
print("Camera stopped.")
