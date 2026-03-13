#!/usr/bin/env python
"""
Merged YOLO Tracking + C-Term Application

Features:
- Left window: YOLO tracking with optional contour overlay
- Right window: C-term heatmap (optional)
- Control panel for parameters and toggles
- Side-by-side real-time comparison

Controls:
- ESC in the OpenCV window: quit
- Tkinter window: sliders/toggles update live
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import sys
from pathlib import Path
import threading
from collections import defaultdict

# Add project to path for YOLO imports
sys.path.insert(0, str(Path(__file__).parent / "yolo_tracking-master" / "yolo_tracking-master"))

from ultralytics import YOLO
from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import WEIGHTS, TRACKER_CONFIGS
import torch

# ==============================================================
# -------------------- LINEAR LIGHT -----------------------------
# ==============================================================

SRGB_TO_LINEAR_LUT = np.array([
    (i / 255.0) / 12.92 if (i / 255.0) <= 0.04045
    else ((i / 255.0 + 0.055) / 1.055) ** 2.4
    for i in range(256)
], dtype=np.float32)

def srgb_to_linear(bgr_u8: np.ndarray) -> np.ndarray:
    """Vectorized sRGB->linear using LUT; input is BGR uint8."""
    return SRGB_TO_LINEAR_LUT[bgr_u8]

def extract_linear_luminance(bgr_linear: np.ndarray) -> np.ndarray:
    """Linear-light luminance from linear BGR."""
    return (
        0.0722 * bgr_linear[:, :, 0] +
        0.7152 * bgr_linear[:, :, 1] +
        0.2126 * bgr_linear[:, :, 2]
    ).astype(np.float32)

def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn > 1e-6:
        return (x - mn) / (mx - mn)
    return np.zeros_like(x, dtype=np.float32)

def adjust_frame_lighting(frame: np.ndarray, brightness: float, contrast: float, saturation: float, exposure: float) -> np.ndarray:
    """Adjust frame brightness, contrast, saturation, and exposure."""
    # Work in float for processing
    img = frame.astype(np.float32) / 255.0
    
    # Apply exposure compensation (multiplicative)
    exposure_factor = 2.0 ** exposure  # -1.0->0.5x, 0.0->1.0x, +1.0->2.0x
    img = img * exposure_factor
    
    # Apply brightness (additive)
    brightness_factor = brightness / 100.0
    img = img + brightness_factor
    
    # Apply contrast (around 0.5)
    img = 0.5 + contrast * (img - 0.5)
    
    # Apply saturation in HSV space
    if saturation != 1.0:
        hsv = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * saturation  # Adjust S channel
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
    
    # Clamp to valid range and convert back to uint8
    img = np.clip(img, 0, 1) * 255
    return img.astype(np.uint8)

# ==============================================================
# -------------------- VIS UTILS --------------------------------
# ==============================================================

def cmap_turbo01(x01: np.ndarray) -> np.ndarray:
    """Apply TURBO colormap to a [0,1] float field."""
    x_u8 = np.clip(x01 * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(x_u8, cv2.COLORMAP_TURBO)

def apply_colormap(x01: np.ndarray, colormap_name: str) -> np.ndarray:
    """Apply selected colormap to a [0,1] float field."""
    x_u8 = np.clip(x01 * 255, 0, 255).astype(np.uint8)
    
    colormap_dict = {
        'TURBO': cv2.COLORMAP_TURBO,
        'VIRIDIS': cv2.COLORMAP_VIRIDIS,
        'HOT': cv2.COLORMAP_HOT,
        'COOL': cv2.COLORMAP_COOL,
        'JET': cv2.COLORMAP_JET,
        'TWILIGHT': cv2.COLORMAP_TWILIGHT,
        'PARULA': cv2.COLORMAP_PARULA,
        'MAGMA': cv2.COLORMAP_MAGMA,
    }
    
    colormap = colormap_dict.get(colormap_name, cv2.COLORMAP_TURBO)
    return cv2.applyColorMap(x_u8, colormap)

def get_contour_color_bgr(color_name: str) -> tuple:
    """Get BGR color tuple from color name."""
    colors = {
        'Orange': (0, 165, 255),
        'Green': (0, 255, 0),
        'Red': (0, 0, 255),
        'Blue': (255, 0, 0),
        'Yellow': (0, 255, 255),
        'Cyan': (255, 255, 0),
        'Magenta': (255, 0, 255),
        'White': (255, 255, 255),
    }
    return colors.get(color_name, (0, 165, 255))

def get_class_color(class_id: int) -> tuple:
    """Get a unique BGR color for each class ID."""
    # Define a diverse palette of colors
    colors = [
        (255, 0, 0),      # Blue
        (0, 255, 0),      # Green
        (0, 0, 255),      # Red
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (128, 0, 255),    # Orange-Purple
        (0, 128, 255),    # Orange
        (255, 128, 0),    # Light Blue
        (128, 255, 0),    # Light Green
        (0, 255, 128),    # Green-Cyan
        (255, 0, 128),    # Purple
        (128, 128, 255),  # Light Purple
        (128, 255, 128),  # Light Green
        (255, 128, 128),  # Light Red
        (192, 0, 192),    # Dark Purple
        (192, 192, 0),    # Olive
        (0, 192, 192),    # Teal
        (192, 0, 0),      # Dark Blue
        (0, 192, 0),      # Dark Green
    ]
    return colors[class_id % len(colors)]

def extract_contours_from_region(region: np.ndarray) -> list:
    """Extract contours from a region using Canny edge detection."""
    if region.size == 0:
        return []
    
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def reconstruct_face_with_contours(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, 
                                   edge_contours: list = None, thickness: int = 2) -> np.ndarray:
    """
    Reconstruct a face image with contour outlines.
    
    Args:
        frame: Original frame
        x1, y1, x2, y2: Bounding box coordinates
        edge_contours: List of contours to draw
        thickness: Contour line thickness
    
    Returns:
        Reconstructed face image or None if invalid
    """
    try:
        # Validate bounds
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Extract the region
        face_region = frame[y1:y2, x1:x2].copy()
        
        if face_region.size == 0 or face_region.shape[0] <= 0 or face_region.shape[1] <= 0:
            return None
        
        # Create a mask for contours
        mask = np.zeros(face_region.shape[:2], dtype=np.uint8)
        
        # Draw extracted edge contours on mask with error handling
        if edge_contours and len(edge_contours) > 0:
            for contour in edge_contours:
                try:
                    if contour is not None and len(contour) > 0:
                        cv2.drawContours(mask, [contour], 0, 255, thickness)
                except Exception:
                    continue
        
        # Create outlined version
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Create output with original image + contour overlay
        output = face_region.copy()
        
        # Highlight contour areas with a blend
        contour_mask = mask > 0
        if np.any(contour_mask):
            output[contour_mask] = cv2.addWeighted(output[contour_mask], 0.6, 
                                                   np.array([0, 255, 255], dtype=np.uint8), 0.4, 0)
        
        return output
    except Exception as e:
        print(f"Error in reconstruct_face_with_contours: {e}")
        return None

def create_face_gallery(frame: np.ndarray, detections: list, max_height: int = 120,
                        G_up: np.ndarray = None, C_up: np.ndarray = None, F_up: np.ndarray = None,
                        ui = None) -> np.ndarray:
    """
    Create a gallery of face thumbnails at the bottom of the screen.
    
    Args:
        frame: Original frame
        detections: List of tuples (x1, y1, x2, y2, track_id, edge_contours)
        max_height: Maximum height for each thumbnail
        G_up: G-term heatmap (optional)
        C_up: C-term heatmap (optional)
        F_up: F-term heatmap (optional)
        ui: UI object with toggle states (optional)
    
    Returns:
        Gallery image
    """
    try:
        if frame is None or frame.size == 0:
            return np.zeros((max_height, 640, 3), dtype=np.uint8)
        
        if not detections:
            return np.zeros((max_height, frame.shape[1], 3), dtype=np.uint8)
        
        gallery_width = frame.shape[1]
        num_detections = len(detections)
        thumb_width = max(1, gallery_width // max(1, num_detections))
        
        gallery = np.zeros((max_height, gallery_width, 3), dtype=np.uint8)
        
        for idx, detection_data in enumerate(detections):
            try:
                x1, y1, x2, y2, track_id, edge_contours = detection_data
                
                # Validate coordinates
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Extract face region directly from frame
                h, w = frame.shape[:2]
                x1_safe = max(0, x1)
                y1_safe = max(0, y1)
                x2_safe = min(w, x2)
                y2_safe = min(h, y2)
                
                if x2_safe <= x1_safe or y2_safe <= y1_safe:
                    continue
                
                face_region = frame[y1_safe:y2_safe, x1_safe:x2_safe].copy()
                
                if face_region.size == 0 or face_region.shape[0] <= 0 or face_region.shape[1] <= 0:
                    continue
                
                # If facial_generate is enabled, use black background with only contours
                if ui is not None and ui.facial_generate.get():
                    face_region = np.zeros_like(face_region)
                
                # Draw edge contours on face region if they exist
                if edge_contours and len(edge_contours) > 0:
                    for contour in edge_contours:
                        try:
                            if contour is not None and len(contour) > 0:
                                cv2.drawContours(face_region, [contour], 0, (255, 255, 255), 1)
                        except Exception:
                            pass
                
                # Draw G-term contours if enabled
                if ui is not None and ui.enable_gterm.get() and ui.show_contours_g.get() and G_up is not None:
                    try:
                        g_region = G_up[y1_safe:y2_safe, x1_safe:x2_safe]
                        if g_region.size > 0:
                            g_field = np.clip(g_region, 0, 1)
                            for p in [75, 90]:
                                thresh = np.percentile(g_field, p)
                                mask = (g_field >= thresh).astype(np.uint8)
                                frac = float(mask.mean())
                                if 0.01 < frac < 0.99:
                                    mask_u8 = (mask * 255).astype(np.uint8)
                                    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    cv2.drawContours(face_region, contours, -1, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                    except Exception:
                        pass
                
                # Draw C-term contours if enabled
                if ui is not None and ui.enable_cterm.get() and ui.show_contours_c.get() and C_up is not None:
                    try:
                        c_region = C_up[y1_safe:y2_safe, x1_safe:x2_safe]
                        if c_region.size > 0:
                            c_field = np.clip(c_region, 0, 1)
                            for p in [75, 90]:
                                thresh = np.percentile(c_field, p)
                                mask = (c_field >= thresh).astype(np.uint8)
                                frac = float(mask.mean())
                                if 0.01 < frac < 0.99:
                                    mask_u8 = (mask * 255).astype(np.uint8)
                                    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    cv2.drawContours(face_region, contours, -1, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                    except Exception:
                        pass
                
                # Draw F-term contours if enabled
                if ui is not None and ui.enable_fterm.get() and ui.show_contours_f.get() and F_up is not None:
                    try:
                        f_region = F_up[y1_safe:y2_safe, x1_safe:x2_safe]
                        if f_region.size > 0:
                            f_field = np.clip(f_region, 0, 1)
                            for p in [75, 90]:
                                thresh = np.percentile(f_field, p)
                                mask = (f_field >= thresh).astype(np.uint8)
                                frac = float(mask.mean())
                                if 0.01 < frac < 0.99:
                                    mask_u8 = (mask * 255).astype(np.uint8)
                                    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    cv2.drawContours(face_region, contours, -1, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                    except Exception:
                        pass
                
                # Resize to thumbnail
                thumb_height = max(20, max_height - 10)
                height_diff = y2_safe - y1_safe
                width_diff = x2_safe - x1_safe
                
                if height_diff <= 0:
                    continue
                
                aspect = width_diff / float(height_diff)
                thumb_w = max(1, int(thumb_height * aspect))
                thumb_w = min(thumb_w, thumb_width - 5)
                
                if thumb_w <= 0:
                    continue
                
                thumb = cv2.resize(face_region, (thumb_w, thumb_height), interpolation=cv2.INTER_LINEAR)
                
                if thumb.size == 0:
                    continue
                
                # Position in gallery
                start_x = idx * thumb_width
                end_x = min(start_x + thumb_w, gallery_width)
                start_y = 5
                end_y = min(start_y + thumb_height, max_height)
                
                # Place thumbnail
                actual_w = end_x - start_x
                actual_h = end_y - start_y
                
                if actual_w > 0 and actual_h > 0 and actual_w <= thumb.shape[1] and actual_h <= thumb.shape[0]:
                    gallery[start_y:end_y, start_x:end_x] = thumb[:actual_h, :actual_w]
                
                # Draw ID label safely
                label_x = max(0, min(start_x + 5, gallery_width - 50))
                label_y = max(15, min(start_y + 20, max_height - 5))
                cv2.putText(gallery, f"ID:{int(track_id)}", (label_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            except Exception as e:
                continue
        
        return gallery
    
    except Exception as e:
        return np.zeros((max_height, 640, 3), dtype=np.uint8)

def draw_contours(
    image_bgr: np.ndarray,
    field01: np.ndarray,
    color=(0, 165, 255),  # orange
    percentiles=(75, 90),
    thickness=1
):
    """Draw percentile-based contours of a [0,1] field onto image_bgr."""
    field = np.clip(field01, 0, 1)
    for p in percentiles:
        try:
            thresh = np.percentile(field, p)
        except Exception:
            continue
        mask = (field >= thresh).astype(np.uint8)
        frac = float(mask.mean())
        if frac < 0.01 or frac > 0.99:
            continue
        mask_u8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_bgr, contours, -1, color, thickness, lineType=cv2.LINE_AA)

# ==============================================================
# -------------------- C TERM -----------------------------------
# ==============================================================

def C_term(I: np.ndarray, blur: float = 10.0, gamma: float = 1.0, sharp: float = 1.0) -> np.ndarray:
    """
    C-term (edge/contrast) pipeline consistent with your app's structure:
      local mean/std normalization -> Sobel magnitude -> normalize01
      -> bilateral -> CLAHE -> unsharp -> gamma
    """
    m = cv2.GaussianBlur(I, (31, 31), blur)
    s = np.sqrt(cv2.GaussianBlur((I - m) ** 2, (31, 31), blur) + 1e-6)
    In = (I - m) / (s + 1e-6)

    gx = cv2.Sobel(In, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(In, cv2.CV_32F, 0, 1)
    C = cv2.magnitude(gx, gy)
    C = normalize01(C)

    C_bilat = cv2.bilateralFilter(C, d=5, sigmaColor=0.1, sigmaSpace=3)

    C_u8 = np.clip(C_bilat * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    C_clahe = clahe.apply(C_u8).astype(np.float32) / 255.0

    C_blur = cv2.GaussianBlur(C_clahe, (0, 0), 1.0)
    C_sharp = np.clip(C_clahe + sharp * (C_clahe - C_blur), 0, 1)

    C_gamma = np.power(np.clip(C_sharp, 0, 1), gamma)
    return C_gamma.astype(np.float32)

def G_term(I: np.ndarray, sig: float = 2.0, gamma: float = 1.0, sharp: float = 1.0) -> np.ndarray:
    """
    G-term (Gaussian coherence):
      GaussianBlur -> normalize01 -> bilateral -> CLAHE -> unsharp -> gamma
    """
    G = cv2.GaussianBlur(I, (0, 0), sig)
    G = normalize01(G)

    G_bilat = cv2.bilateralFilter(G, d=5, sigmaColor=0.1, sigmaSpace=3)

    G_u8 = np.clip(G_bilat * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    G_clahe = clahe.apply(G_u8).astype(np.float32) / 255.0

    G_blur = cv2.GaussianBlur(G_clahe, (0, 0), 1.0)
    G_sharp = np.clip(G_clahe + sharp * (G_clahe - G_blur), 0, 1)

    G_gamma = np.power(np.clip(G_sharp, 0, 1), gamma)
    return G_gamma.astype(np.float32)

def F_term(I: np.ndarray, scale: float = 2.0, gamma: float = 1.0, sharp: float = 1.0) -> np.ndarray:
    """
    F-term (gradient energy):
      GaussianBlur -> Sobel gradients -> magnitude -> normalize01
      -> bilateral -> CLAHE -> unsharp -> gamma
    """
    blur = cv2.GaussianBlur(I, (0, 0), scale)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1)
    F = cv2.magnitude(gx, gy)
    F = normalize01(F)

    F_bilat = cv2.bilateralFilter(F, d=5, sigmaColor=0.1, sigmaSpace=3)

    F_u8 = np.clip(F_bilat * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    F_clahe = clahe.apply(F_u8).astype(np.float32) / 255.0

    F_blur = cv2.GaussianBlur(F_clahe, (0, 0), 1.0)
    F_sharp = np.clip(F_clahe + sharp * (F_clahe - F_blur), 0, 1)

    F_gamma = np.power(np.clip(F_sharp, 0, 1), gamma)
    return F_gamma.astype(np.float32)

# ==============================================================
# -------------------- YOLO SETUP --------------------------------
# ==============================================================

def convert_boxes_to_numpy(boxes):
    """Convert Ultralytics Boxes object to numpy array format."""
    if hasattr(boxes, 'xyxy') and hasattr(boxes, 'conf'):
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else np.asarray(boxes.xyxy)
        conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else np.asarray(boxes.conf)
        cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else np.asarray(boxes.cls)
        
        dets = np.concatenate([xyxy, conf.reshape(-1, 1), cls.reshape(-1, 1)], axis=1)
        return dets
    elif isinstance(boxes, np.ndarray):
        return boxes
    else:
        return np.asarray(boxes)

# ==============================================================
# -------------------- UI ----------------------------------------
# ==============================================================

class MergedUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YOLO + GCF Terms Controls")
        self.root.geometry("420x700")

        # C-Term parameters
        self.blur = tk.DoubleVar(value=10.0)
        self.c_gamma = tk.DoubleVar(value=1.0)
        self.c_sharp = tk.DoubleVar(value=1.0)

        # G-Term parameters
        self.g_sig = tk.DoubleVar(value=2.0)
        self.g_gamma = tk.DoubleVar(value=1.0)
        self.g_sharp = tk.DoubleVar(value=1.0)

        # F-Term parameters
        self.f_scale = tk.DoubleVar(value=2.0)
        self.f_gamma = tk.DoubleVar(value=1.0)
        self.f_sharp = tk.DoubleVar(value=1.0)

        # Toggles
        self.show_heatmap_c = tk.BooleanVar(value=False)
        self.show_heatmap_g = tk.BooleanVar(value=False)
        self.show_heatmap_f = tk.BooleanVar(value=False)
        self.show_contours_g = tk.BooleanVar(value=False)
        self.show_contours_c = tk.BooleanVar(value=False)
        self.show_contours_f = tk.BooleanVar(value=False)
        self.enable_cterm = tk.BooleanVar(value=True)
        self.enable_gterm = tk.BooleanVar(value=False)
        self.enable_fterm = tk.BooleanVar(value=False)
        self.enable_yolo = tk.BooleanVar(value=True)
        self.show_edge_detection = tk.BooleanVar(value=False)
        self.facial_generate = tk.BooleanVar(value=False)

        # YOLO parameters
        self.confidence = tk.DoubleVar(value=0.4)
        self.iou = tk.DoubleVar(value=0.7)
        self.model_type = tk.StringVar(value='general')  # 'general', 'face', or 'police'
        
        # Camera adjustment parameters
        self.camera_brightness = tk.DoubleVar(value=0.0)
        self.camera_contrast = tk.DoubleVar(value=1.0)
        self.camera_saturation = tk.DoubleVar(value=1.0)
        self.camera_exposure = tk.DoubleVar(value=0.0)
        
        # Heatmap and Contour options
        self.heatmap_colormap = tk.StringVar(value='TURBO')
        self.contour_color = tk.StringVar(value='Orange')
        
        self.heatmap_options = ['TURBO', 'VIRIDIS', 'HOT', 'COOL', 'JET', 'TWILIGHT', 'PARULA', 'MAGMA']
        self.color_options = ['Orange', 'Green', 'Red', 'Blue', 'Yellow', 'Cyan', 'Magenta', 'White']

        # Create a canvas with scrollbar for the settings
        canvas = tk.Canvas(self.root, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Allow mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Title
        ttk.Label(scrollable_frame, text="YOLO + GCF Terms Tracking", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=10, pady=(10, 5))

        # Status indicator
        self.model_status_label = ttk.Label(scrollable_frame, text="", font=("Segoe UI", 10, "bold"), foreground="blue")
        self.model_status_label.pack(anchor="w", padx=10, pady=(0, 10))

        # YOLO section
        ttk.LabelFrame(scrollable_frame, text="YOLO Settings", padding=10).pack(fill="x", padx=10, pady=5)
        
        ttk.Label(scrollable_frame, text="Confidence threshold").pack(anchor="w", padx=10)
        ttk.Scale(scrollable_frame, from_=0.0, to=1.0, variable=self.confidence, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(scrollable_frame, textvariable=self.confidence).pack(anchor="w", padx=10)

        ttk.Label(scrollable_frame, text="IoU threshold").pack(anchor="w", padx=10, pady=(6, 0))
        ttk.Scale(scrollable_frame, from_=0.0, to=1.0, variable=self.iou, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(scrollable_frame, textvariable=self.iou).pack(anchor="w", padx=10)

        ttk.Checkbutton(scrollable_frame, text="Enable YOLO tracking", variable=self.enable_yolo).pack(anchor="w", padx=10, pady=(4, 0))
        ttk.Checkbutton(scrollable_frame, text="Show edge detection (yellow)", variable=self.show_edge_detection).pack(anchor="w", padx=10)
        ttk.Checkbutton(scrollable_frame, text="Facial generate (contours only)", variable=self.facial_generate).pack(anchor="w", padx=10)

        ttk.Label(scrollable_frame, text="Model Type").pack(anchor="w", padx=10, pady=(6, 0))
        ttk.Combobox(scrollable_frame, textvariable=self.model_type, values=['general', 'face', 'police'], state="readonly", width=20).pack(fill="x", padx=10)
        ttk.Label(scrollable_frame, text="Restart app to apply change", font=("Segoe UI", 8, "italic")).pack(anchor="w", padx=10, pady=(0, 10))

        # Camera Settings section
        ttk.LabelFrame(scrollable_frame, text="Camera Adjustments", padding=10).pack(fill="x", padx=10, pady=5)
        
        ttk.Label(scrollable_frame, text="Brightness (-100 to +100)").pack(anchor="w", padx=10)
        ttk.Scale(scrollable_frame, from_=-100.0, to=100.0, variable=self.camera_brightness, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(scrollable_frame, textvariable=self.camera_brightness).pack(anchor="w", padx=10)

        ttk.Label(scrollable_frame, text="Contrast (0.5 to 3.0)").pack(anchor="w", padx=10, pady=(6, 0))
        ttk.Scale(scrollable_frame, from_=0.5, to=3.0, variable=self.camera_contrast, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(scrollable_frame, textvariable=self.camera_contrast).pack(anchor="w", padx=10)

        ttk.Label(scrollable_frame, text="Saturation (0.0 to 2.0)").pack(anchor="w", padx=10, pady=(6, 0))
        ttk.Scale(scrollable_frame, from_=0.0, to=2.0, variable=self.camera_saturation, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(scrollable_frame, textvariable=self.camera_saturation).pack(anchor="w", padx=10)

        ttk.Label(scrollable_frame, text="Exposure (-1.0 to +1.0)").pack(anchor="w", padx=10, pady=(6, 0))
        ttk.Scale(scrollable_frame, from_=-1.0, to=1.0, variable=self.camera_exposure, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(scrollable_frame, textvariable=self.camera_exposure).pack(anchor="w", padx=10)

        # G-Term section
        ttk.LabelFrame(scrollable_frame, text="G-Term (Gaussian Coherence)", padding=10).pack(fill="x", padx=10, pady=5)

        ttk.Label(scrollable_frame, text="Sigma").pack(anchor="w", padx=10)
        ttk.Scale(scrollable_frame, from_=0.5, to=10.0, variable=self.g_sig, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(scrollable_frame, textvariable=self.g_sig).pack(anchor="w", padx=10)

        ttk.Label(scrollable_frame, text="Gamma").pack(anchor="w", padx=10, pady=(6, 0))
        ttk.Scale(scrollable_frame, from_=0.5, to=2.5, variable=self.g_gamma, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(scrollable_frame, textvariable=self.g_gamma).pack(anchor="w", padx=10)

        ttk.Label(scrollable_frame, text="Sharpness").pack(anchor="w", padx=10, pady=(6, 0))
        ttk.Scale(scrollable_frame, from_=0.0, to=2.0, variable=self.g_sharp, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(scrollable_frame, textvariable=self.g_sharp).pack(anchor="w", padx=10)

        ttk.Checkbutton(scrollable_frame, text="Enable G-Term", variable=self.enable_gterm).pack(anchor="w", padx=10, pady=(4, 0))
        ttk.Checkbutton(scrollable_frame, text="Show G heatmap", variable=self.show_heatmap_g).pack(anchor="w", padx=10)
        ttk.Checkbutton(scrollable_frame, text="Show G contours on YOLO", variable=self.show_contours_g).pack(anchor="w", padx=10)

        # C-Term section
        ttk.LabelFrame(scrollable_frame, text="C-Term (Contrast/Edges)", padding=10).pack(fill="x", padx=10, pady=5)

        ttk.Label(scrollable_frame, text="Blur (Gaussian sigma)").pack(anchor="w", padx=10)
        ttk.Scale(scrollable_frame, from_=2.0, to=20.0, variable=self.blur, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(scrollable_frame, textvariable=self.blur).pack(anchor="w", padx=10)

        ttk.Label(scrollable_frame, text="Gamma").pack(anchor="w", padx=10, pady=(6, 0))
        ttk.Scale(scrollable_frame, from_=0.5, to=2.5, variable=self.c_gamma, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(scrollable_frame, textvariable=self.c_gamma).pack(anchor="w", padx=10)

        ttk.Label(scrollable_frame, text="Sharpness").pack(anchor="w", padx=10, pady=(6, 0))
        ttk.Scale(scrollable_frame, from_=0.0, to=2.0, variable=self.c_sharp, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(scrollable_frame, textvariable=self.c_sharp).pack(anchor="w", padx=10)

        ttk.Checkbutton(scrollable_frame, text="Enable C-Term", variable=self.enable_cterm).pack(anchor="w", padx=10, pady=(4, 0))
        ttk.Checkbutton(scrollable_frame, text="Show C heatmap", variable=self.show_heatmap_c).pack(anchor="w", padx=10)
        ttk.Checkbutton(scrollable_frame, text="Show C contours on YOLO", variable=self.show_contours_c).pack(anchor="w", padx=10)

        # F-Term section
        ttk.LabelFrame(scrollable_frame, text="F-Term (Gradient Energy)", padding=10).pack(fill="x", padx=10, pady=5)

        ttk.Label(scrollable_frame, text="Scale").pack(anchor="w", padx=10)
        ttk.Scale(scrollable_frame, from_=0.5, to=10.0, variable=self.f_scale, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(scrollable_frame, textvariable=self.f_scale).pack(anchor="w", padx=10)

        ttk.Label(scrollable_frame, text="Gamma").pack(anchor="w", padx=10, pady=(6, 0))
        ttk.Scale(scrollable_frame, from_=0.5, to=2.5, variable=self.f_gamma, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(scrollable_frame, textvariable=self.f_gamma).pack(anchor="w", padx=10)

        ttk.Label(scrollable_frame, text="Sharpness").pack(anchor="w", padx=10, pady=(6, 0))
        ttk.Scale(scrollable_frame, from_=0.0, to=2.0, variable=self.f_sharp, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(scrollable_frame, textvariable=self.f_sharp).pack(anchor="w", padx=10)

        ttk.Checkbutton(scrollable_frame, text="Enable F-Term", variable=self.enable_fterm).pack(anchor="w", padx=10, pady=(4, 0))
        ttk.Checkbutton(scrollable_frame, text="Show F heatmap", variable=self.show_heatmap_f).pack(anchor="w", padx=10)
        ttk.Checkbutton(scrollable_frame, text="Show F contours on YOLO", variable=self.show_contours_f).pack(anchor="w", padx=10)

        # Visualization options
        ttk.LabelFrame(scrollable_frame, text="Visualization", padding=10).pack(fill="x", padx=10, pady=5)

        ttk.Label(scrollable_frame, text="Heatmap Colormap").pack(anchor="w", padx=10, pady=(6, 0))
        ttk.Combobox(scrollable_frame, textvariable=self.heatmap_colormap, values=self.heatmap_options, state="readonly", width=20).pack(fill="x", padx=10)

        ttk.Label(scrollable_frame, text="Contour Color").pack(anchor="w", padx=10, pady=(6, 0))
        ttk.Combobox(scrollable_frame, textvariable=self.contour_color, values=self.color_options, state="readonly", width=20).pack(fill="x", padx=10, pady=(0, 10))

    def update(self):
        try:
            self.root.update_idletasks()
            self.root.update()
        except:
            pass

# ==============================================================
# -------------------- MAIN ------------------------------------
# ==============================================================

def main():
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera (VideoCapture(0)).")

    print("Initializing YOLO model...")
    ui = MergedUI()
    current_model_type = ui.model_type.get()
    
    if current_model_type == 'face':
        model_path = 'yolov8n-face.pt'
        status_text = "🔍 MODEL: FACE DETECTION"
        print("Loading YOLOv8-face model (face detection)...")
    elif current_model_type == 'police':
        model_path = 'runs/detect/train3/weights/best.pt'
        status_text = "🚔 MODEL: CHICAGO POLICE DETECTION"
        print("Loading YOLOv8m police detection model...")
    else:
        model_path = 'yolov8n.pt'
        status_text = "🔍 MODEL: GENERAL OBJECTS"
        print("Loading YOLOv8 general object detection model...")
    
    ui.model_status_label.config(text=status_text)
    yolo = YOLO(model_path)
    
    # Enable tracking (requires botsort or deepocsort)
    try:
        from boxmot.tracker_zoo import create_tracker
        from boxmot.utils import WEIGHTS, TRACKER_CONFIGS
        print("BoxMOT tracking enabled")
    except:
        print("Warning: BoxMOT not fully available, using basic tracking")

    print("Initializing UI...")


    # Small compute grid for C-term (keeps it fast)
    ihfm_width, ihfm_height = 320, 180

    print("\n" + "=" * 60)
    print("YOLO + GCF Terms Tracking Started")
    print("=" * 60)
    print("Left window:  YOLO tracking with optional term contours")
    print("Right windows: G/C/F heatmaps (optional)")
    print("Press ESC to quit")
    print("=" * 60 + "\n")

    quit_event = False

    try:
        while not quit_event:
            ret, frame = cap.read()
            if not ret:
                break

            # Check if model type has changed
            new_model_type = ui.model_type.get()
            if new_model_type != current_model_type:
                print(f"\n>>> Model switched from '{current_model_type}' to '{new_model_type}'")
                current_model_type = new_model_type
                
                # Unload old model
                del yolo
                torch.cuda.empty_cache()
                
                # Load new model
                if new_model_type == 'face':
                    model_path = 'yolov8n-face.pt'
                    status_text = "🔍 MODEL: FACE DETECTION"
                    print("Loading YOLOv8-face model (face detection)...")
                elif new_model_type == 'police':
                    model_path = 'runs/detect/train3/weights/best.pt'
                    status_text = "🚔 MODEL: CHICAGO POLICE DETECTION"
                    print("Loading YOLOv8m police detection model...")
                else:
                    model_path = 'yolov8n.pt'
                    status_text = "🔍 MODEL: GENERAL OBJECTS"
                    print("Loading YOLOv8 general object detection model...")
                
                yolo = YOLO(model_path)
                ui.model_status_label.config(text=status_text)
                print(f"Model switched successfully!\n")

            # Apply camera adjustments
            if (ui.camera_brightness.get() != 0.0 or 
                ui.camera_contrast.get() != 1.0 or 
                ui.camera_saturation.get() != 1.0 or 
                ui.camera_exposure.get() != 0.0):
                frame = adjust_frame_lighting(
                    frame,
                    float(ui.camera_brightness.get()),
                    float(ui.camera_contrast.get()),
                    float(ui.camera_saturation.get()),
                    float(ui.camera_exposure.get())
                )

            ui.update()
            h0, w0 = frame.shape[:2]

            # YOLO Tracking
            yolo_img = frame.copy()
            detection_boxes = []  # Store detection boxes for contour masking
            face_gallery_data = []  # Store face data for gallery reconstruction
            if ui.enable_yolo.get():
                try:
                    results = yolo.track(
                        frame,
                        conf=float(ui.confidence.get()),
                        iou=float(ui.iou.get()),
                        verbose=False
                    )
                    
                    if results and len(results) > 0:
                        result = results[0]
                        # Draw bounding boxes and tracks
                        if result.boxes is not None and len(result.boxes) > 0:
                            print(f"\n--- Frame Detections ({len(result.boxes)} objects) ---")
                            for box in result.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                track_id = int(box.id) if box.id is not None else -1
                                conf = float(box.conf)
                                cls = int(box.cls) if box.cls is not None else -1
                                
                                # Get class name and color
                                class_name = yolo.names.get(cls, f"Unknown({cls})") if hasattr(yolo, 'names') else f"Class{cls}"
                                class_color = get_class_color(cls)
                                
                                # Print detection info
                                print(f"  [{track_id}] {class_name:15s} - Conf: {conf:.2f}")
                                
                                # Store bounding box for contour masking
                                detection_boxes.append((x1, y1, x2, y2))
                                
                                # Draw bounding box with class color
                                cv2.rectangle(yolo_img, (x1, y1), (x2, y2), class_color, 2)
                                
                                # Draw track ID and class name with class color
                                if track_id >= 0:
                                    cv2.putText(yolo_img, f"ID:{track_id}", (x1, y1 - 10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color, 2)
                                
                                # Draw class name and confidence with class color
                                cv2.putText(yolo_img, f"{class_name} {conf:.2f}", (x1, y2 + 20),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color, 2)
                                
                                # Collect face data for gallery (always, for reconstruction)
                                face_gallery_data.append((x1, y1, x2, y2, track_id, []))
                                print(f"Added face to gallery: ID={track_id}, coords=({x1},{y1})-({x2},{y2})")
                
                        # Apply edge detection inside boxes if enabled
                        if ui.show_edge_detection.get():
                            for box_idx, (x1, y1, x2, y2) in enumerate(detection_boxes):
                                try:
                                    # Extract region inside box
                                    box_region = frame[y1:y2, x1:x2]
                                    if box_region.size == 0 or box_region.shape[0] <= 0 or box_region.shape[1] <= 0:
                                        continue
                                    
                                    # Convert to grayscale
                                    gray = cv2.cvtColor(box_region, cv2.COLOR_BGR2GRAY)
                                    # Apply Canny edge detection
                                    edges = cv2.Canny(gray, 50, 150)
                                    # Find contours on edges
                                    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                                    
                                    # Draw contours in yellow (0, 255, 255 in BGR) with proper offset
                                    for contour in contours:
                                        if contour is not None and len(contour) > 0:
                                            # Offset contour coordinates to match full image position
                                            offset_contour = contour + np.array([x1, y1])
                                            cv2.drawContours(yolo_img, [offset_contour], 0, (0, 255, 255), 1)
                                    
                                    # Update gallery data with contours
                                    if box_idx < len(face_gallery_data):
                                        x1_g, y1_g, x2_g, y2_g, track_id, _ = face_gallery_data[box_idx]
                                        face_gallery_data[box_idx] = (x1_g, y1_g, x2_g, y2_g, track_id, contours)
                                
                                except Exception as e:
                                    print(f"Error processing edge detection for box {box_idx}: {e}")
                                    continue
                except Exception as e:
                    print(f"YOLO error: {e}")

            # GCF Terms Computation
            gterm_heatmap = None
            cterm_heatmap = None
            fterm_heatmap = None
            
            try:
                # Linearize and compute luminance
                lin = srgb_to_linear(frame)
                I_full = extract_linear_luminance(lin)

                # Downsample to compute grid
                I = cv2.resize(I_full, (ihfm_width, ihfm_height), interpolation=cv2.INTER_AREA)

                # Compute terms on small grid
                if ui.enable_gterm.get():
                    G = G_term(
                        I,
                        sig=float(ui.g_sig.get()),
                        gamma=float(ui.g_gamma.get()),
                        sharp=float(ui.g_sharp.get()),
                    )
                    G_up = cv2.resize(G, (w0, h0), interpolation=cv2.INTER_LINEAR)
                    if ui.show_heatmap_g.get():
                        gterm_heatmap = apply_colormap(G_up, ui.heatmap_colormap.get())

                if ui.enable_cterm.get():
                    C = C_term(
                        I,
                        blur=float(ui.blur.get()),
                        gamma=float(ui.c_gamma.get()),
                        sharp=float(ui.c_sharp.get()),
                    )
                    C_up = cv2.resize(C, (w0, h0), interpolation=cv2.INTER_LINEAR)
                    if ui.show_heatmap_c.get():
                        cterm_heatmap = apply_colormap(C_up, ui.heatmap_colormap.get())

                if ui.enable_fterm.get():
                    F = F_term(
                        I,
                        scale=float(ui.f_scale.get()),
                        gamma=float(ui.f_gamma.get()),
                        sharp=float(ui.f_sharp.get()),
                    )
                    F_up = cv2.resize(F, (w0, h0), interpolation=cv2.INTER_LINEAR)
                    if ui.show_heatmap_f.get():
                        fterm_heatmap = apply_colormap(F_up, ui.heatmap_colormap.get())

                # Helper function to draw contours within detection boxes
                def draw_term_contours(yolo_img, term_field, detection_boxes, enabled, color):
                    """Draw contours for a term within detection boxes."""
                    if not enabled or len(detection_boxes) == 0:
                        return
                    
                    for (x1, y1, x2, y2) in detection_boxes:
                        # Extract the term region for this detection box
                        term_region = term_field[y1:y2, x1:x2]
                        
                        # Draw contours only within this region
                        field = np.clip(term_region, 0, 1)
                        for p in [75, 90]:
                            try:
                                thresh = np.percentile(field, p)
                            except Exception:
                                continue
                            mask = (field >= thresh).astype(np.uint8)
                            frac = float(mask.mean())
                            if frac < 0.01 or frac > 0.99:
                                continue
                            mask_u8 = (mask * 255).astype(np.uint8)
                            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            # Offset contours to the correct position in the full image
                            for contour in contours:
                                contour[:, :, 0] += x1
                                contour[:, :, 1] += y1
                            cv2.drawContours(yolo_img, contours, -1, color, 1, lineType=cv2.LINE_AA)

                # Draw contours on YOLO image if enabled, only within detection boxes
                contour_color = get_contour_color_bgr(ui.contour_color.get())
                
                if ui.enable_gterm.get():
                    draw_term_contours(yolo_img, G_up, detection_boxes, ui.show_contours_g.get(), contour_color)
                
                if ui.enable_cterm.get():
                    draw_term_contours(yolo_img, C_up, detection_boxes, ui.show_contours_c.get(), contour_color)
                
                if ui.enable_fterm.get():
                    draw_term_contours(yolo_img, F_up, detection_boxes, ui.show_contours_f.get(), contour_color)

            except Exception as e:
                print(f"GCF Terms error: {e}")

            # Create display with tiles
            tiles = [yolo_img]
            if gterm_heatmap is not None:
                tiles.append(gterm_heatmap)
            if cterm_heatmap is not None:
                tiles.append(cterm_heatmap)
            if fterm_heatmap is not None:
                tiles.append(fterm_heatmap)

            if len(tiles) == 1:
                display = tiles[0]
            else:
                display = np.hstack(tiles)

            # Create and add face gallery at bottom
            if face_gallery_data and len(face_gallery_data) > 0:
                try:
                    # Prepare GCF term parameters for gallery
                    G_up_gallery = G_up if 'G_up' in locals() else None
                    C_up_gallery = C_up if 'C_up' in locals() else None
                    F_up_gallery = F_up if 'F_up' in locals() else None
                    
                    gallery = create_face_gallery(frame, face_gallery_data, max_height=140,
                                                  G_up=G_up_gallery, C_up=C_up_gallery, F_up=F_up_gallery,
                                                  ui=ui)
                    if gallery is not None and gallery.size > 0:
                        display = np.vstack([display, gallery])
                except Exception as e:
                    print(f"Error creating gallery: {e}")

            cv2.imshow("YOLO + C-Term (ESC to quit)", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    except KeyboardInterrupt:
        print("\n\nTracking stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            ui.root.destroy()
        except:
            pass

if __name__ == "__main__":
    main()
