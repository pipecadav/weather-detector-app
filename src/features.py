"""Descriptores por ROI y vector fijo por fotografía, sin aprender de etiquetas."""
from functools import lru_cache

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

from src.preprocessing import ImageConfig, preprocess, read_rgb

GABOR_ORIENTATIONS = (0, 45, 90, 135)
GABOR_WAVELENGTHS = (5, 10, 20)  # Valores usados en el código de la Parte 1.
SHAPE_NAMES = ["area_fraction", "perimeter_normalized", "circularity", "aspect_ratio",
               "extent", "solidity", *[f"hu_log_{i}" for i in range(1, 8)],
               "region_count", "selected_coverage", "no_region"]


@lru_cache(maxsize=1)
def gabor_bank():
    """Reutilizar 12 kernels 31×31 de la Parte 1 sin recalcular por imagen."""
    return [cv2.getGaborKernel((31, 31), 0.56 * wavelength, np.deg2rad(angle),
                              wavelength, 0.5, 0, ktype=cv2.CV_32F)
            for angle in GABOR_ORIENTATIONS for wavelength in GABOR_WAVELENGTHS]


def roi_features(roi):
    """HOG rectangular; color y estadísticas de textura dentro de la máscara.

    HOG incorpora contexto del rectángulo y relleno reflejado: no se multiplica
    por la máscara porque esto crearía gradientes artificiales en su frontera.
    Gabor usa energía respuesta², como el código de E.2 de la Parte 1.
    """
    rgb, mask = roi["rgb"], roi["mask"].astype(bool)
    if not mask.any():
        raise ValueError("ROI sin píxeles válidos.")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
            block_norm="L2-Hys", feature_vector=True).astype(np.float32)
    gab = []
    for kernel in gabor_bank():
        response = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        energy = response[mask].astype(np.float64) ** 2
        gab.extend((energy.mean(), energy.std()))
    lbp_features = []
    for points, radius in ((8, 1), (16, 2)):
        # Excluir centros cuya vecindad atraviesa la máscara siempre que sea posible.
        inner = cv2.erode(mask.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8)).astype(bool)
        codes = local_binary_pattern(gray, points, radius, method="uniform")
        hist = np.bincount(codes[inner if inner.any() else mask].astype(int), minlength=points+2)
        lbp_features.extend(hist / hist.sum())
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float64)
    hsv /= np.array([179.0, 255.0, 255.0])
    color = []
    for channel in range(3):
        values = hsv[..., channel][mask]
        mean, std = values.mean(), values.std()
        skew = np.mean(((values - mean) / std) ** 3) if std > 1e-8 else 0.0
        color.extend((mean, std, skew))
    return {"hog": h, "gabor": np.asarray(gab), "lbp": np.asarray(lbp_features),
            "color": np.asarray(color)}


def shape_features(region, image_area):
    """Medir la máscara original, antes del relleno, no la forma del rectángulo."""
    mask = region["mask_original"].astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    area = float(region["area"])
    perimeter = cv2.arcLength(contour, True)
    contour_area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    h, w = mask.shape
    hu = cv2.HuMoments(cv2.moments(mask, binaryImage=True)).ravel()
    signed_log = -np.sign(hu) * np.log10(np.maximum(np.abs(hu), 1e-30))
    return np.array([area / image_area, perimeter / np.sqrt(image_area),
                     4*np.pi*contour_area / max(perimeter**2, 1e-12), w/h,
                     area/(w*h), contour_area/max(hull_area, 1e-12), *signed_log])


def extract_image_features(rgb, config=ImageConfig()):
    """Media de descriptores global/local y forma solo de regiones válidas.

    Siempre incluir la imagen completa evita perder información de escenas.
    Gabor conserva 24 valores al promediar ROI. Forma tiene 13 medias más
    cantidad, cobertura y ausencia de regiones (16 valores).
    """
    processed = preprocess(rgb, config)
    rois = [processed["global_roi"], *processed["regions"]]
    descriptors = [roi_features(roi) for roi in rois]
    blocks = {name: np.mean([d[name] for d in descriptors], axis=0).astype(np.float32)
              for name in ("gabor", "hog", "lbp", "color")}
    regions = processed["regions"]
    area = processed["gray"].size
    form = np.mean([shape_features(r, area) for r in regions], axis=0) if regions else np.zeros(13)
    blocks["shape"] = np.array([*form, len(regions),
                                sum(r["area"] for r in regions)/area,
                                int(not regions)], dtype=np.float32)
    if not all(np.isfinite(v).all() for v in blocks.values()):
        raise ValueError("El extractor produjo características no finitas.")
    return blocks


def feature_names():
    """Orden explícito de columnas, indispensable para reproducir inferencia."""
    return {
        "gabor": [f"gabor_{angle}_{wave}_{stat}" for angle in GABOR_ORIENTATIONS
                  for wave in GABOR_WAVELENGTHS for stat in ("energy_mean", "energy_std")],
        "hog": [f"hog_{i}" for i in range(8100)],
        "lbp": [f"lbp_{points}_{radius}_bin_{i}" for points, radius in ((8,1),(16,2))
                for i in range(points+2)],
        "color": [f"hsv_{c}_{s}" for c in ("h","s","v") for s in ("mean","std","skew")],
        "shape": SHAPE_NAMES,
    }
