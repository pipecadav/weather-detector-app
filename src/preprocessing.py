"""Normalización y regiones visuales: no detecta objetos meteorológicos reales."""
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class ImageConfig:
    """Parámetros fijos compartidos entre entrenamiento e inferencia."""
    max_side: int = 640
    roi_size: int = 128
    filter_name: str = "bilateral"
    segmentation: str = "otsu"
    min_area_fraction: float = 0.005
    max_area_fraction: float = 0.90
    max_regions: int = 3

    def to_dict(self):
        return asdict(self)


def read_rgb(path):
    """Leer con Pillow, respetar EXIF y devolver RGB uint8 (no BGR)."""
    with Image.open(Path(path)) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()


def resize_max(rgb, max_side=640):
    """Reducir imágenes grandes sin deformarlas ni ampliar imágenes pequeñas."""
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    return cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA) if scale < 1 else rgb.copy()


def normalize_roi(rgb, mask, size=128):
    """Ajustar proporción y rellenar bordes por reflexión; máscara de relleno=0.

    La reflexión evita un borde negro artificial en HOG. Color/textura usarán
    solo píxeles indicados por la máscara, excluyendo el relleno.
    """
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    small = cv2.resize(rgb, (nw, nh), interpolation=interpolation)
    small_mask = cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
    top, left = (size - nh) // 2, (size - nw) // 2
    bottom, right = size - nh - top, size - nw - left
    return (cv2.copyMakeBorder(small, top, bottom, left, right, cv2.BORDER_REFLECT_101),
            cv2.copyMakeBorder(small_mask, top, bottom, left, right, cv2.BORDER_CONSTANT))


def preprocess(rgb, config=ImageConfig()):
    """Separar rama visual de rama de segmentación; devolver evidencia y ROI.

    La imagen global siempre se conserva para no perder contexto (niebla, cielo).
    Las regiones adicionales son componentes claros/oscuros con área limitada.
    El filtro solo afecta la segmentación: no borramos textura en los descriptores.
    """
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Se espera una imagen RGB uint8 de tres canales.")
    image = resize_max(rgb, config.max_side)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if config.filter_name == "bilateral":
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    elif config.filter_name == "gaussian":
        filtered = cv2.GaussianBlur(gray, (3, 3), 0)
    elif config.filter_name == "median":
        filtered = cv2.medianBlur(gray, 3)
    elif config.filter_name == "none":
        filtered = gray.copy()
    else:
        raise ValueError("Filtro desconocido.")
    if config.segmentation == "otsu":
        threshold, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif config.segmentation == "adaptive":
        threshold = None
        binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 5)
    else:
        raise ValueError("Segmentación desconocida.")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    candidates = []
    total_area = gray.size
    # Dos polaridades: no asumimos que el fenómeno siempre es más claro.
    cleaned_masks = []
    for polarity, base in (("bright", binary), ("dark", 255 - binary)):
        clean = cv2.morphologyEx(base, cv2.MORPH_OPEN, kernel)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
        cleaned_masks.append(clean)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
        for k in range(1, count):
            x, y, w, h, area = stats[k].tolist()
            if config.min_area_fraction <= area / total_area <= config.max_area_fraction:
                component = (labels[y:y+h, x:x+w] == k).astype(np.uint8)
                candidates.append({"bbox": (x, y, w, h), "area": area,
                                   "mask_original": component, "polarity": polarity})
    candidates.sort(key=lambda r: (-r["area"], r["bbox"]))
    regions = candidates[:config.max_regions]
    for region in regions:
        x, y, w, h = region["bbox"]
        region["rgb"], region["mask"] = normalize_roi(
            image[y:y+h, x:x+w], region["mask_original"], config.roi_size)
    global_rgb, global_mask = normalize_roi(image, np.ones(gray.shape, np.uint8), config.roi_size)
    global_roi = {"rgb": global_rgb, "mask": global_mask, "kind": "global"}
    return {"rgb": image, "gray": gray, "filtered": filtered, "binary": binary,
            "cleaned_masks": cleaned_masks, "threshold": threshold,
            "regions": regions, "global_roi": global_roi, "config": config.to_dict()}
