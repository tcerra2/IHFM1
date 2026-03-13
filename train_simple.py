#!/usr/bin/env python
"""
Simple YOLO Training Script - Chicago Police Cars
"""

from ultralytics import YOLO
import sys

def main():
    print("\n" + "=" * 70)
    print("TRAINING - Chicago Police Car Detection")
    print("=" * 70)
    
    print("\nLoading YOLOv8m model...")
    model = YOLO('yolov8m.pt')
    
    print("Starting training...\n")
    
    results = model.train(
        data='police_cars_dataset/data.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        patience=20,
        device='cpu',
        verbose=True,
        save=True,
        project='runs/detect',
        name='train4',
        exist_ok=True
    )
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print("Model saved to: runs/detect/train4/weights/best.pt")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
