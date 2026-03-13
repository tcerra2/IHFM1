"""
IHFM G/C/F Terms Standalone App (Single File)

Includes:
- Webcam capture (OpenCV)
- Linear-light luminance conversion (sRGB -> linear -> Y)
- G term: Gaussian coherence-style smoothing pipeline
- C term: local contrast/edge-style pipeline
- F term: gradient/fractal-ish edge energy pipeline
- Optional contour overlay from any selected term
- Minimal Tkinter controls for term parameters + view toggles

Run:
    pip install opencv-python numpy
    python gcf_terms_standalone_app.py

Keys:
- ESC in OpenCV window: quit
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
        thresh = float(np.percentile(field, p))
        mask = (field >= thresh).astype(np.uint8)
        frac = float(mask.mean())
        if frac < 0.01 or frac > 0.99:
            continue
        mask_u8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_bgr, contours, -1, color, thickness, lineType=cv2.LINE_AA)

# ==============================================================
# -------------------- TERMS ------------------------------------
# ==============================================================

def G_term(I: np.ndarray, sig: float = 2.0, gamma: float = 1.0, sharp: float = 1.0) -> np.ndarray:
    """
    G-term from your IHFM app:
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
    F-term from your IHFM app:
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

def C_term(I: np.ndarray, blur: float = 10.0, gamma: float = 1.0, sharp: float = 1.0) -> np.ndarray:
    """
    C-term (edge/contrast) pipeline consistent with your app’s structure:
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

# ==============================================================
# -------------------- UI ---------------------------------------
# ==============================================================

class GCFUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("IHFM G/C/F Controls")
        self.root.geometry("360x520")

        # G params
        self.g_sig = tk.DoubleVar(value=2.0)
        self.g_gamma = tk.DoubleVar(value=1.0)
        self.g_sharp = tk.DoubleVar(value=1.0)

        # C params
        self.c_blur = tk.DoubleVar(value=10.0)
        self.c_gamma = tk.DoubleVar(value=1.0)
        self.c_sharp = tk.DoubleVar(value=1.0)

        # F params
        self.f_scale = tk.DoubleVar(value=2.0)
        self.f_gamma = tk.DoubleVar(value=1.0)
        self.f_sharp = tk.DoubleVar(value=1.0)

        # view toggles
        self.show_G = tk.BooleanVar(value=True)
        self.show_C = tk.BooleanVar(value=True)
        self.show_F = tk.BooleanVar(value=True)

        self.show_contours = tk.BooleanVar(value=False)
        self.contour_source = tk.StringVar(value="C")  # "G"|"C"|"F"

        self._build()

    def _section(self, title: str):
        ttk.Separator(self.root).pack(fill="x", padx=10, pady=(10, 6))
        ttk.Label(self.root, text=title, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=10)

    def _build(self):
        ttk.Label(self.root, text="IHFM Terms (G / C / F)", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=10, pady=(10, 4))
        ttk.Label(self.root, text="Heatmaps will tile to the right of the camera feed.").pack(anchor="w", padx=10)

        self._section("G term (Gaussian coherence)")
        ttk.Label(self.root, text="sigma").pack(anchor="w", padx=10)
        ttk.Scale(self.root, from_=0.5, to=10.0, variable=self.g_sig, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(self.root, textvariable=self.g_sig).pack(anchor="w", padx=10)
        ttk.Label(self.root, text="gamma").pack(anchor="w", padx=10, pady=(4, 0))
        ttk.Scale(self.root, from_=0.5, to=2.5, variable=self.g_gamma, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(self.root, textvariable=self.g_gamma).pack(anchor="w", padx=10)
        ttk.Label(self.root, text="sharp").pack(anchor="w", padx=10, pady=(4, 0))
        ttk.Scale(self.root, from_=0.0, to=2.0, variable=self.g_sharp, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(self.root, textvariable=self.g_sharp).pack(anchor="w", padx=10)

        self._section("C term (contrast / edges)")
        ttk.Label(self.root, text="blur (local norm sigma)").pack(anchor="w", padx=10)
        ttk.Scale(self.root, from_=2.0, to=20.0, variable=self.c_blur, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(self.root, textvariable=self.c_blur).pack(anchor="w", padx=10)
        ttk.Label(self.root, text="gamma").pack(anchor="w", padx=10, pady=(4, 0))
        ttk.Scale(self.root, from_=0.5, to=2.5, variable=self.c_gamma, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(self.root, textvariable=self.c_gamma).pack(anchor="w", padx=10)
        ttk.Label(self.root, text="sharp").pack(anchor="w", padx=10, pady=(4, 0))
        ttk.Scale(self.root, from_=0.0, to=2.0, variable=self.c_sharp, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(self.root, textvariable=self.c_sharp).pack(anchor="w", padx=10)

        self._section("F term (gradient energy)")
        ttk.Label(self.root, text="scale").pack(anchor="w", padx=10)
        ttk.Scale(self.root, from_=0.5, to=10.0, variable=self.f_scale, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(self.root, textvariable=self.f_scale).pack(anchor="w", padx=10)
        ttk.Label(self.root, text="gamma").pack(anchor="w", padx=10, pady=(4, 0))
        ttk.Scale(self.root, from_=0.5, to=2.5, variable=self.f_gamma, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(self.root, textvariable=self.f_gamma).pack(anchor="w", padx=10)
        ttk.Label(self.root, text="sharp").pack(anchor="w", padx=10, pady=(4, 0))
        ttk.Scale(self.root, from_=0.0, to=2.0, variable=self.f_sharp, orient="horizontal").pack(fill="x", padx=10)
        ttk.Label(self.root, textvariable=self.f_sharp).pack(anchor="w", padx=10)

        self._section("View")
        ttk.Checkbutton(self.root, text="Show G heatmap", variable=self.show_G).pack(anchor="w", padx=10)
        ttk.Checkbutton(self.root, text="Show C heatmap", variable=self.show_C).pack(anchor="w", padx=10)
        ttk.Checkbutton(self.root, text="Show F heatmap", variable=self.show_F).pack(anchor="w", padx=10)

        ttk.Checkbutton(self.root, text="Overlay contours on camera", variable=self.show_contours).pack(anchor="w", padx=10, pady=(6, 0))
        ttk.Label(self.root, text="Contour source").pack(anchor="w", padx=10, pady=(2, 0))
        ttk.Combobox(self.root, textvariable=self.contour_source, values=["G", "C", "F"], state="readonly").pack(fill="x", padx=10)

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

    ui = GCFUI()

    # small compute grid
    ihfm_width, ihfm_height = 320, 180

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ui.update()

        # Linearize & luminance
        lin = srgb_to_linear(frame)
        I_full = extract_linear_luminance(lin)

        I = cv2.resize(I_full, (ihfm_width, ihfm_height), interpolation=cv2.INTER_AREA)

        # Compute terms on small grid
        G = G_term(I, sig=float(ui.g_sig.get()), gamma=float(ui.g_gamma.get()), sharp=float(ui.g_sharp.get()))
        C = C_term(I, blur=float(ui.c_blur.get()), gamma=float(ui.c_gamma.get()), sharp=float(ui.c_sharp.get()))
        F = F_term(I, scale=float(ui.f_scale.get()), gamma=float(ui.f_gamma.get()), sharp=float(ui.f_sharp.get()))

        # Upsample to camera size
        h0, w0 = frame.shape[:2]
        G_up = cv2.resize(G, (w0, h0), interpolation=cv2.INTER_NEAREST)
        C_up = cv2.resize(C, (w0, h0), interpolation=cv2.INTER_NEAREST)
        F_up = cv2.resize(F, (w0, h0), interpolation=cv2.INTER_NEAREST)

        # Base camera tile
        cam = frame.copy()

        # Optional contour overlay
        if ui.show_contours.get():
            src = ui.contour_source.get().upper()
            field = C_up if src == "C" else (G_up if src == "G" else F_up)
            draw_contours(cam, field, color=(0, 165, 255), percentiles=(75, 90), thickness=1)

        tiles = [cam]
        if ui.show_G.get():
            tiles.append(cmap_turbo01(G_up))
        if ui.show_C.get():
            tiles.append(cmap_turbo01(C_up))
        if ui.show_F.get():
            tiles.append(cmap_turbo01(F_up))

        dash = np.hstack(tiles) if len(tiles) > 1 else tiles[0]

        cv2.imshow("IHFM G/C/F Standalone (ESC to quit)", dash)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
