"""API mínima de predicción para entrega; no contiene aplicación ni interfaz."""
import time
from pathlib import Path

import joblib
import numpy as np

from src.experiments import combine, transform_blocks
from src.features import extract_image_features
from src.preprocessing import ImageConfig, read_rgb


def predict_image(image_path, model_path="models/weather_model.joblib"):
    """Clasificar una imagen con el extractor y transformaciones guardados.

    Cargar solo archivos joblib de confianza. Este clasificador cerrado siempre
    elige una clase conocida: no reconoce automáticamente imágenes fuera del tema.
    """
    bundle = joblib.load(Path(model_path))
    config = ImageConfig(**bundle["extraction"]["image_config"])
    start = time.perf_counter()
    features = extract_image_features(read_rgb(image_path), config)
    blocks = {name: features[name][None, :] for name in bundle["families"]}
    x = combine(transform_blocks(blocks, bundle["transforms"]), bundle["families"])
    model = bundle["model"]
    prediction = model.predict(x)[0]
    probabilities = model.predict_proba(x)[0]
    index = np.flatnonzero(model.classes_ == prediction)[0]
    return {"class": str(prediction), "confidence": float(probabilities[index]),
            "probabilities": dict(zip(model.classes_.tolist(), probabilities.tolist())),
            "processing_seconds": time.perf_counter()-start}
