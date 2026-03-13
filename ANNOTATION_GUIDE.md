# Chicago Police Cars Dataset - Annotation Guide

## Overview
This guide explains how to re-annotate the police car dataset with specific bounding boxes instead of generic full-image annotations. Proper annotation is critical for model accuracy.

## Problem with Current Annotations
The existing dataset was trained with **generic full-image annotations**:
- Each annotation box: `0 0.5 0.5 0.85 0.8`
- This teaches the model: "Anything in this image is a vehicle"
- Result: Model detects ANY vehicle, not specifically police cars

## Solution: Specific Police Car Annotations
We need to:
1. Draw boxes ONLY around the police car itself
2. Ignore non-police vehicles in the image
3. Focus on police-specific features
4. Create tight, accurate bounding boxes

---

## Step-by-Step Setup

### Step 1: Prepare Annotation Directories
```powershell
cd c:\Users\tcerr\Documents\Yolo
.\.venv\Scripts\python.exe setup_annotations.py
```

This script will:
- ✓ Create `police_cars_dataset/labels/` directories
- ✓ Create `police_cars_dataset/classes.txt` file
- ✓ Backup old annotations to `police_cars_dataset/labels_backup/`
- ✓ Clear old annotations for fresh re-annotation

### Step 2: Launch LabelImg
```powershell
cd c:\Users\tcerr\Documents\Yolo
.\.venv\Scripts\python.exe launch_labelimg.py
```

Or directly:
```powershell
cd c:\Users\tcerr\Documents\Yolo
.\.venv\Scripts\python.exe -m labelImg police_cars_dataset/images/train
```

---

## LabelImg Interface Guide

### Main Window
- **Left**: Image display area
- **Left panel**: File list (train images)
- **Right panel**: Box list and labels
- **Bottom**: Progress and status bar

### Starting Annotation
1. First image loads automatically
2. Class `chicago_police_car` should be pre-selected
3. Use **"Create RectBox" button** (or press 'w') to draw boxes

### Drawing a Bounding Box

#### Good Box (Tight Police Car):
```
[Police car with roof lights and markings]
┌─────────────────┐
│ ↙ Roof lights   │
│ 🚔 Police car   │
│ Markings + ID↖  │
└─────────────────┘
```
- Box tightly fits police vehicle
- Includes roof lights, antenna, stripes
- Minimal background on sides

#### Bad Box (Generic Full Image):
```
[Entire street scene]
┌──────────────────────────────────┐
│ [Police car]  [Building]  [Tree] │
│ This teaches generic vehicle      │
│ detection, not police-specific    │
└──────────────────────────────────┘
```
- ❌ Don't do this!
- ❌ Includes unrelated objects
- ❌ Creates generic model

---

## Annotation Workflow

### For Each Image:

1. **Assess the image**
   - Identify police cars (distinctive markings, lights, paint schemes)
   - Ignore other vehicles, people, buildings
   - Count how many police vehicles are present

2. **Draw boxes for EACH police car**
   - Click "Create RectBox" button (or press 'w')
   - Click top-left corner of police car
   - Drag to bottom-right corner
   - Release mouse
   - Box appears with class label

3. **Verify box placement**
   - Police car should be fully enclosed
   - Minimize background/padding
   - If incorrect, click "Edit" and adjust
   - To delete: Select box, press Delete key

4. **Save annotation**
   - Press 's' or click "Save"
   - Creates `.txt` file matching image name
   - Format: YOLO (class_id, center_x, center_y, width, height in normalized coordinates)

5. **Move to next image**
   - Press 'd' or click "Next"
   - Repeat for all images

---

## Police Car Features to Look For

### Chicago Police Distinctive Features:
- **Paint scheme**: Blue & white or solid colors with police markings
- **"POLICE" text**: Often visible on doors/hood
- **Emergency lights**: Red/blue roof lights (even if not activated)
- **Antenna**: Usually visible on roof
- **License plate area**: Police-specific formatting
- **Markings**: Stripes, star logos, unit numbers
- **Light bar**: Roof-mounted emergency light array

### Example Identifications:
| Feature | Police Car | Regular Car |
|---------|-----------|-----------|
| Roof lights | ✓ Large light bar | ✗ None or simple lights |
| Markings | ✓ "POLICE" + star | ✗ Generic branding |
| Paint | ✓ Blue/white scheme | ✗ Various colors |
| Antenna | ✓ Usually taller | ✗ Small or hidden |

---

## Keyboard Shortcuts (LabelImg)

| Key | Action |
|-----|--------|
| `w` | Create rectangle box |
| `d` | Next image |
| `a` | Previous image |
| `s` | Save annotations |
| `Delete` | Delete selected box |
| `Ctrl+s` | Save all |
| `Esc` | Deselect box |
| `r` | Edit/move box |
| `c` | Change class (if needed) |

---

## Annotation Progress

### Three Folders to Annotate:
1. **train/** (92 images) - Primary training data
2. **val/** (11 images) - Validation data
3. **test/** (12 images) - Test data

### Recommended Order:
1. Start with **train/** folder (most important)
2. Then **val/** folder
3. Finally **test/** folder

### Tracking Progress:
- LabelImg shows: `Image [X] / [Total]` at bottom
- You can see remaining images in file list
- Annotated files get a checkmark (✓) in some versions

---

## Quality Assurance Checklist

Before moving to training, verify:

- [ ] All images have been opened in LabelImg
- [ ] Each police car has a bounding box
- [ ] Boxes are tight (minimal background padding)
- [ ] No boxes around non-police vehicles
- [ ] All `.txt` files created in corresponding `labels/` folders
- [ ] `.txt` files match image file names
- [ ] No empty label files (remove if present)

### Verify Files:
```powershell
# Check train labels
Get-ChildItem police_cars_dataset/labels/train/

# Check val labels
Get-ChildItem police_cars_dataset/labels/val/

# Check test labels
Get-ChildItem police_cars_dataset/labels/test/

# Count annotations
(Get-ChildItem police_cars_dataset/labels/train/*.txt).Count
# Should be 92
```

---

## YOLO Label Format

Each image gets a `.txt` file with annotations in this format:

```
<class_id> <center_x> <center_y> <width> <height>
0 0.45 0.52 0.30 0.35
```

### Explanation:
- `class_id`: 0 (chicago_police_car is class 0)
- `center_x`: 0.45 (45% from left edge, normalized 0-1)
- `center_y`: 0.52 (52% from top edge, normalized 0-1)
- `width`: 0.30 (box is 30% of image width)
- `height`: 0.35 (box is 35% of image height)

**LabelImg automatically converts** pixel coordinates → normalized YOLO format.

---

## Common Issues & Solutions

### Issue: LabelImg won't start
```powershell
# Reinstall labelImg
.\.venv\Scripts\python.exe -m pip install --upgrade labelImg
```

### Issue: Classes not showing
- Ensure `police_cars_dataset/classes.txt` exists
- Contains: `chicago_police_car`
- Try restarting LabelImg

### Issue: Can't find images
- Run LabelImg from correct directory
- Use absolute paths if needed:
```powershell
.\.venv\Scripts\python.exe -m labelImg `
  "c:\Users\tcerr\Documents\Yolo\police_cars_dataset\images\train"
```

### Issue: Accidentally deleted annotations
- Restore from backup:
```powershell
Remove-Item police_cars_dataset/labels -Recurse
Copy-Item police_cars_dataset/labels_backup -Destination police_cars_dataset/labels -Recurse
```

---

## After Annotation Complete

### 1. Verify annotations are saved
```powershell
cd c:\Users\tcerr\Documents\Yolo
$train_count = (Get-ChildItem police_cars_dataset/labels/train/*.txt -ErrorAction SilentlyContinue).Count
$val_count = (Get-ChildItem police_cars_dataset/labels/val/*.txt -ErrorAction SilentlyContinue).Count
$test_count = (Get-ChildItem police_cars_dataset/labels/test/*.txt -ErrorAction SilentlyContinue).Count
Write-Output "Train: $train_count, Val: $val_count, Test: $test_count"
```

Should show: `Train: 92, Val: 11, Test: 12`

### 2. Retrain the model
```powershell
cd c:\Users\tcerr\Documents\Yolo
.\.venv\Scripts\python.exe train_now.py
```

This will:
- Load annotations from `labels/` folders
- Train new YOLOv8m model with specific police car boxes
- Save to `runs/detect/train4/` (or next available)
- Expect significantly better accuracy!

### 3. Update app with new model
After training completes:
```python
# In police_tracker.py, update the model path (around line 783):
model_path = 'runs/detect/train4/weights/best.pt'  # New model
```

---

## Expected Improvements

### Before Annotation (Current Model):
- Trained on: Generic full-image boxes
- Result: Detects ANY vehicle
- False positives: High (regular cars, buses, etc.)

### After Annotation (Improved Model):
- Trained on: Specific police car bounding boxes
- Result: Detects only police-marked vehicles
- False positives: Low (specific police features)
- Accuracy: Expected improvement from ~99% → 95-98% (but more reliable)

---

## Questions or Issues?

If you encounter problems:
1. Check this guide's "Common Issues" section
2. Check LabelImg documentation: https://github.com/heartexlabs/labelImg
3. Verify `police_cars_dataset/` structure is intact
4. Run `setup_annotations.py` again if directories missing

---

**Ready to annotate! Start with:**
```powershell
python setup_annotations.py
python launch_labelimg.py
```
