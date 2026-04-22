"""
Helper Script to Set Up the YOLOv8 AI Model

Ultralytics YOLOv8 requires a '.pt' weights file to run. 

By default, if you don't have a custom capsule model, the system will use 'yolov8n.pt' 
(the standard YOLOv8 Nano model), which Ultralytics will download automatically the first 
time you run the code.

However, the generic 'yolov8n.pt' is trained on common objects (dogs, cars, people) and 
might not recognize industrial capsules reliably.

### How to get a highly accurate Capsule Model:

1. Go to Roboflow Universe: https://universe.roboflow.com/
2. Search for "Pill Detection" or "Capsule"
3. Find a dataset with "YOLOv8" format.
4. Click "Download Model" or use the provided Python snippet to download the 'best.pt' weights file.
5. Move that 'best.pt' file into the 'c:\ECG\models\' directory and rename it to 'capsule_best.pt'.

If you just want to test if the pipeline runs, simply run 'main.py'. 
Ultralytics will automatically download 'yolov8n.pt' to use as a fallback.
"""

print(__doc__)
