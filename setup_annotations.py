#!/usr/bin/env python
"""
Convert existing generic annotations to LabelImg-compatible YOLO format.

This script:
1. Creates the labels directory structure if it doesn't exist
2. Backs up old annotations
3. Clears labels for re-annotation

Run this BEFORE starting LabelImg annotation.
"""

import os
import shutil
from pathlib import Path

def setup_annotation_directories():
    """Create directory structure for annotations."""
    dataset_dir = Path("police_cars_dataset")
    labels_dir = dataset_dir / "labels"
    
    # Create directory structure
    for split in ["train", "val", "test"]:
        split_labels_dir = labels_dir / split
        split_labels_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {split_labels_dir}")
    
    print(f"\n✓ Label directories ready at: {labels_dir}")
    return labels_dir

def create_classes_file():
    """Create classes.txt file for LabelImg."""
    dataset_dir = Path("police_cars_dataset")
    classes_file = dataset_dir / "classes.txt"
    
    with open(classes_file, 'w') as f:
        f.write("chicago_police_car\n")
    
    print(f"✓ Created classes file: {classes_file}")
    return classes_file

def clear_old_annotations():
    """Back up and clear old annotations."""
    dataset_dir = Path("police_cars_dataset")
    labels_dir = dataset_dir / "labels"
    
    # Backup old annotations
    backup_dir = dataset_dir / "labels_backup"
    if labels_dir.exists():
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(labels_dir, backup_dir)
        print(f"✓ Backed up old annotations to: {backup_dir}")
        
        # Clear labels
        for txt_file in labels_dir.rglob("*.txt"):
            txt_file.unlink()
        print(f"✓ Cleared old annotations for re-annotation")
    
    return backup_dir if backup_dir.exists() else None

def main():
    print("\n" + "=" * 70)
    print("ANNOTATION SETUP - Chicago Police Cars Dataset")
    print("=" * 70 + "\n")
    
    # Check if dataset exists
    if not Path("police_cars_dataset").exists():
        print("Error: police_cars_dataset directory not found!")
        return False
    
    # Create directory structure
    labels_dir = setup_annotation_directories()
    
    # Create classes file
    classes_file = create_classes_file()
    
    # Backup and clear old annotations
    backup_dir = clear_old_annotations()
    
    print("\n" + "=" * 70)
    print("SETUP COMPLETE - Ready for annotation!")
    print("=" * 70 + "\n")
    print("Next steps:")
    print("1. Run: python launch_labelimg.py")
    print("2. Follow the on-screen annotation guidelines")
    print("3. Focus on drawing boxes ONLY around police cars")
    print("4. When finished, run: python train_now.py")
    print("\n" + "=" * 70 + "\n")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
