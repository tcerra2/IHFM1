"""
IHFM C-Term Standalone App (Single File)

Extracted from the user's IHFM single-file application.
- Webcam capture (OpenCV)
- Linear-light luminance conversion
- C-term computation (local contrast/edges) with the same processing stages:
  normalize -> bilateral -> CLAHE -> unsharp -> gamma
- Optional contour overlay
- Minimal Tkinter control panel (blur/gamma/sharp + toggles)

Controls:
- ESC in the OpenCV window: quit
- Tkinter window: sliders/toggles update live
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

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

# ==============================================================
# -------------------- VIS UTILS --------------------------------
# ==============================================================

def cmap_turbo01(x01: np.ndarray) -> np.ndarray:
    """Apply TURBO colormap to a [0,1] float field."""
    x_u8 = np.clip(x01 * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(x_u8, cv2.COLORMAP_TURBO)

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
    C-term extracted from the IHFM app.

    Steps:
    1) Local mean/std normalization (Gaussian)
    2) Gradient magnitude on normalized image
    3) normalize01
    4) bilateral
    5) CLAHE
    6) unsharp mask
    7) gamma correction
    """
    # Local normalization
    m = cv2.GaussianBlur(I, (31, 31), blur)
    s = np.sqrt(cv2.GaussianBlur((I - m) ** 2, (31, 31), blur) + 1e-6)
    In = (I - m) / (s + 1e-6)

    # Gradients
    gx = cv2.Sobel(In, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(In, cv2.CV_32F, 0, 1)
    C = cv2.magnitude(gx, gy)
    C = normalize01(C)

    # Bilateral filter
    C_bilat = cv2.bilateralFilter(C, d=5, sigmaColor=0.1, sigmaSpace=3)

    # CLAHE
    C_u8 = np.clip(C_bilat * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    C_clahe = clahe.apply(C_u8).astype(np.float32) / 255.0

    # Unsharp mask
    C_blur = cv2.GaussianBlur(C_clahe, (0, 0), 1.0)
    C_sharp = np.clip(C_clahe + sharp * (C_clahe - C_blur), 0, 1)

    # Gamma correction
    C_gamma = np.power(np.clip(C_sharp, 0, 1), gamma)
    return C_gamma.astype(np.float32)

# ==============================================================
# -------------------- UI ---------------------------------------
# ==============================================================

class CTermUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("C-Term Controls")
        self.root.geometry("320x260")

        self.blur = tk.DoubleVar(value=10.0)
        self.gamma = tk.DoubleVar(value=1.0)
        self.sharp = tk.DoubleVar(value=1.0)

        self.show_heatmap = tk.BooleanVar(value=True)
        self.show_contours = tk.BooleanVar(value=False)

        ttk.Label(self.root, text="C-Term Parameters", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=10, pady=(10, 2))

        ttk.Label(self.root, text="blur (Gaussian sigma)").pack(anchor="w", padx=10)
        ttk.Scale(self.root, from_=2.0, to=20.0, variable=self.blur, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(self.root, textvariable=self.blur).pack(anchor="w", padx=10)

        ttk.Label(self.root, text="gamma").pack(anchor="w", padx=10, pady=(6, 0))
        ttk.Scale(self.root, from_=0.5, to=2.5, variable=self.gamma, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(self.root, textvariable=self.gamma).pack(anchor="w", padx=10)

        ttk.Label(self.root, text="sharp").pack(anchor="w", padx=10, pady=(6, 0))
        ttk.Scale(self.root, from_=0.0, to=2.0, variable=self.sharp, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(self.root, textvariable=self.sharp).pack(anchor="w", padx=10)

        ttk.Checkbutton(self.root, text="Show C heatmap tile", variable=self.show_heatmap).pack(anchor="w", padx=10, pady=(8, 0))
        ttk.Checkbutton(self.root, text="Overlay C contours on camera", variable=self.show_contours).pack(anchor="w", padx=10)

    def update(self):
        self.root.update_idletasks()
        self.root.update()

# ==============================================================
# -------------------- MAIN -------------------------------------
# ==============================================================

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera (VideoCapture(0)).")

    ui = CTermUI()

    # Small compute grid (keeps it fast like the original app)
    ihfm_width, ihfm_height = 320, 180

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ui.update()

        # Linearize and compute luminance
        lin = srgb_to_linear(frame)
        I_full = extract_linear_luminance(lin)

        # Downsample to compute grid
        I = cv2.resize(I_full, (ihfm_width, ihfm_height), interpolation=cv2.INTER_AREA)

        # Compute C on small grid
        C = C_term(
            I,
            blur=float(ui.blur.get()),
            gamma=float(ui.gamma.get()),
            sharp=float(ui.sharp.get()),
        )

        # Prepare display
        h0, w0 = frame.shape[:2]
        C_up = cv2.resize(C, (w0, h0), interpolation=cv2.INTER_NEAREST)

        # Tile view: camera + optional heatmap
        tiles = [frame.copy()]
        if ui.show_contours.get():
            draw_contours(tiles[0], C_up, color=(0, 165, 255), percentiles=(75, 90), thickness=1)

        if ui.show_heatmap.get():
            tiles.append(cmap_turbo01(C_up))

        if len(tiles) == 1:
            dash = tiles[0]
        else:
            dash = np.hstack(tiles)

        cv2.imshow("C-Term Standalone (ESC to quit)", dash)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
