"""
Capsule AI Trainer Script
-------------------------
Run this script on your WINDOWS LAPTOP to train your custom capsule model!

Prerequisites:
1. Make sure you have installed ultralytics on your laptop:
   pip install ultralytics

2. Extract the dataset ZIP you downloaded from Roboflow.
3. Find the path to the 'data.yaml' file inside that extracted folder.
4. Update the DATA_YAML_PATH variable below to point to that file.
"""

from ultralytics import YOLO

# ⚠️ CHANGE THIS TO THE PATH OF YOUR EXTRACTED data.yaml FILE ⚠️
# Example: "C:/Users/YourName/Downloads/Capsule-Dataset/data.yaml"
DATA_YAML_PATH = r"C:\Users\ngava\Downloads\SV5.v1-v1_capsule_dataset.yolov8\data.yaml"

def main():
    print("Loading base YOLOv8 model...")
    model = YOLO("yolov8n.pt")  # Loads the lightweight 'nano' model

    print(f"Starting training using dataset: {DATA_YAML_PATH}")
    # Train the model
    # epochs=50 is usually enough for a simple object like a capsule
    model.train(
        data=DATA_YAML_PATH,
        epochs=50,
        imgsz=640,
        device="cpu"  # Change to "0" if you have an NVIDIA GPU on your laptop!
    )

    print("\n" + "="*50)
    print("TRAINING COMPLETE! 🎉")
    print("Your new custom weights file is located at:")
    print("runs/detect/train/weights/best.pt")
    print("="*50)
    print("Next steps:")
    print("1. Copy 'runs/detect/train/weights/best.pt'")
    print("2. Paste it into 'C:\\ECG\\models\\'")
    print("3. Rename it to 'capsule_best.pt'")
    print("4. git add . && git commit -m 'added trained model' && git push")

if __name__ == "__main__":
    main()
