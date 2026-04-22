"""
camera_feed.py  –  Minimal RPi Global Shutter Camera live feed
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

# ── Camera init ────────────────────────────────────────────────────────────
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    controls={"FrameRate": 60}
)
picam2.configure(config)
picam2.start()
time.sleep(2)   # let sensor stabilise + AWB settle

# ── Optional: fast shutter for sharp moving objects ─────────────────────
picam2.set_controls({
    "ExposureTime": 1000,   # µs  (lower = sharper, may be darker)
    "AnalogueGain": 12.0,   # boost brightness to compensate
})

count = 1
print("Press SPACE to capture burst | ESC to exit")

while True:
    frame = picam2.capture_array()

    # ✅ FIX: Picamera2 gives RGB, OpenCV expects BGR → swap channels
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow("Live Feed – CapsuleVision", bgr)
    key = cv2.waitKey(1) & 0xFF

    # ── Burst capture ──────────────────────────────────────────────────
    if key == 32:  # SPACE
        print("Capturing burst...")
        for i in range(5):
            f = picam2.capture_array()
            f_bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            filename = os.path.join(desktop_path, f"capture_{count:04d}.jpg")
            cv2.imwrite(filename, f_bgr)
            print(f"  Saved: {filename}")
            count += 1
            time.sleep(0.05)

    elif key == 27:  # ESC
        break

cv2.destroyAllWindows()
picam2.stop()
print("Camera stopped.")
