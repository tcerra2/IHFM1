#!/usr/bin/env python
"""
LabelImg Launcher for Chicago Police Cars Dataset

This script launches LabelImg with the correct configuration for annotating 
the police_cars_dataset with specific police car bounding boxes.

Usage:
    python launch_labelimg.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Get the dataset path
    workspace_dir = Path(__file__).parent
    dataset_dir = workspace_dir / "police_cars_dataset"
    images_dir = dataset_dir / "images"
    
    # Check if directories exist
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)
    
    print("=" * 70)
    print("Chicago Police Cars Dataset - LabelImg Annotation Tool")
    print("=" * 70)
    print(f"\nDataset Location: {dataset_dir}")
    print(f"Images Location: {images_dir}")
    print("\n" + "=" * 70)
    print("ANNOTATION GUIDELINES:")
    print("=" * 70)
    print("""
1. CLASS SELECTION:
   - Select class: chicago_police_car (should be automatically set)

2. DRAWING BOUNDING BOXES:
   - Draw boxes ONLY around the police car itself
   - NOT around the entire image as a generic vehicle
   - Focus on the distinctive police vehicle features:
     * Police markings/stripes
     * Emergency lights
     * Police-specific paint schemes
     * License plate area
   
3. BOX PLACEMENT:
   - Start from top-left corner of police car
   - Drag to bottom-right corner of police car
   - Make boxes tight but inclusive (no excessive padding)
   - One box per police vehicle in image

4. QUALITY CHECKS:
   - Verify box covers the entire police car
   - Check that box doesn't include too much background
   - When done with image, press 's' to save
   - Use arrow keys to navigate between images

5. KEYBOARD SHORTCUTS:
   - 'd': Next image
   - 'a': Previous image
   - 's': Save current annotation
   - 'w': Create rectangle (polygon mode)
   - 'r': Move a box
   - 'Delete': Delete selected box
   - Ctrl+s: Save current annotations
   - Esc: Deselect box

6. WHEN COMPLETE:
   - Annotations are automatically saved as .txt files
   - Each image gets a corresponding .txt file
   - Format: class_id center_x center_y width height (YOLO format)
   - Labels will be in: police_cars_dataset/labels/train/
                        police_cars_dataset/labels/val/
                        police_cars_dataset/labels/test/

""")
    print("=" * 70)
    print("Launching LabelImg...\n")
    
    # Launch labelImg with the dataset
    try:
        # Start with train images first, then val, then test
        subprocess.run([
            sys.executable, "-m", "labelImg",
            str(images_dir / "train"),
            str(dataset_dir / "classes.txt")
        ], cwd=str(workspace_dir))
    except Exception as e:
        print(f"Error launching LabelImg: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
