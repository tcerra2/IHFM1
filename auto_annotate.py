#!/usr/bin/env python
"""
Auto-Annotation Script for Chicago Police Cars Dataset

This script uses the trained police car model to automatically generate
YOLO format annotations for all images in the dataset. 

The annotations are saved as .txt files in the labels/ folders.
You can then refine them in LabelImg if needed.

Usage:
    python auto_annotate.py
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import sys

def normalize_coordinates(x1, y1, x2, y2, img_w, img_h):
    """Convert pixel coordinates to YOLO normalized format."""
    center_x = (x1 + x2) / 2.0 / img_w
    center_y = (y1 + y2) / 2.0 / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    
    # Clamp to [0, 1]
    center_x = max(0, min(1, center_x))
    center_y = max(0, min(1, center_y))
    width = max(0.001, min(1, width))
    height = max(0.001, min(1, height))
    
    return center_x, center_y, width, height

def auto_annotate_dataset(model_path, conf_threshold=0.5):
    """
    Auto-annotate all images in the dataset using the trained model.
    
    Args:
        model_path: Path to the trained YOLO model
        conf_threshold: Confidence threshold for detections
    """
    
    print("\n" + "=" * 70)
    print("AUTO-ANNOTATION - Chicago Police Cars Dataset")
    print("=" * 70)
    
    # Check model exists
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please ensure the model is trained at: runs/detect/train3/weights/best.pt")
        return False
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)
    
    dataset_dir = Path("police_cars_dataset")
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    
    # Process each split
    splits = ["train", "val", "test"]
    total_annotated = 0
    total_detections = 0
    
    for split in splits:
        split_images_dir = images_dir / split
        split_labels_dir = labels_dir / split
        
        if not split_images_dir.exists():
            print(f"Warning: {split} images directory not found")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing {split.upper()} set")
        print(f"{'='*70}")
        
        # Get all images
        image_files = sorted(list(split_images_dir.glob("*.jpg")) + 
                           list(split_images_dir.glob("*.png")))
        
        if not image_files:
            print(f"No images found in {split_images_dir}")
            continue
        
        print(f"Found {len(image_files)} images")
        
        split_total = 0
        split_detections = 0
        
        for idx, img_file in enumerate(image_files, 1):
            try:
                # Load image
                img = cv2.imread(str(img_file))
                if img is None:
                    print(f"  [{idx}/{len(image_files)}] SKIP - Could not load: {img_file.name}")
                    continue
                
                h, w = img.shape[:2]
                
                # Run inference
                results = model.predict(img, conf=conf_threshold, verbose=False)
                
                # Extract detections
                detections = []
                if results and len(results) > 0:
                    result = results[0]
                    if result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(float, box.xyxy[0])
                            conf = float(box.conf)
                            
                            # Convert to YOLO format (class_id, center_x, center_y, width, height)
                            cx, cy, bw, bh = normalize_coordinates(x1, y1, x2, y2, w, h)
                            detections.append((0, cx, cy, bw, bh, conf))
                
                # Save annotations
                label_file = split_labels_dir / img_file.name.replace(img_file.suffix, ".txt")
                
                with open(label_file, 'w') as f:
                    for class_id, cx, cy, bw, bh, conf in detections:
                        f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                
                split_total += 1
                split_detections += len(detections)
                
                status = f"✓ {len(detections)} detection(s)" if detections else "○ No detections"
                print(f"  [{idx:2d}/{len(image_files)}] {img_file.name:30s} - {status}")
                
            except Exception as e:
                print(f"  [{idx}/{len(image_files)}] ERROR - {img_file.name}: {e}")
                continue
        
        print(f"\n{split.upper()} Summary:")
        print(f"  ✓ Annotated: {split_total}/{len(image_files)}")
        print(f"  ✓ Total detections: {split_detections}")
        
        total_annotated += split_total
        total_detections += split_detections
    
    print("\n" + "=" * 70)
    print("ANNOTATION COMPLETE")
    print("=" * 70)
    print(f"Total annotated: {total_annotated}")
    print(f"Total detections: {total_detections}")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("""
1. OPTIONAL: Refine annotations in LabelImg
   python launch_labelimg.py
   
   - Review each image
   - Adjust bounding boxes if needed
   - Ensure boxes are tight around police cars
   - Delete any incorrect detections

2. When satisfied with annotations, retrain the model:
   python train_now.py
   
   - This will use the improved annotations
   - Expect better accuracy!

3. Update the app:
   - Edit police tracker.py around line 783
   - Update model path to new trained model (e.g., train4, train5, etc.)
   - Restart the app to use improved model

""")
    print("=" * 70)
    
    return True

def main():
    # Use the trained police car model
    model_path = "runs/detect/train3/weights/best.pt"
    
    # You can adjust confidence threshold here
    # Higher = more selective, Lower = more detections
    conf_threshold = 0.4
    
    success = auto_annotate_dataset(model_path, conf_threshold)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
