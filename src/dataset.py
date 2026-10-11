"""Auditoría de imágenes sin transformar ni modificar el dataset original."""

from hashlib import sha256
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_class_root(download_path):
    """Localiza la carpeta que contiene las carpetas de clases.

    Kaggle puede devolver una carpeta envolvente. Se buscan carpetas cuyos
    hijos directos contengan imágenes, sin fijar de antemano once nombres.
    Si hay varias candidatas, se exige elegir explícitamente una de ellas.
    """
    root = Path(download_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"No existe la carpeta: {root}")
    candidates = []
    for parent in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        children = [p for p in parent.iterdir() if p.is_dir() and not p.name.startswith(".")]
        image_classes = [p for p in children if any(
            f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS for f in p.iterdir()
        )]
        if len(image_classes) >= 2:
            candidates.append(parent)
    if len(candidates) != 1:
        raise ValueError(
            "No se encontró una raíz única de clases. Define CLASS_ROOT manualmente. "
            f"Candidatas: {[str(p) for p in candidates]}"
        )
    return candidates[0]


def audit_dataset(class_root):
    """Devuelve un registro por imagen, incluidas las que no pueden leerse.

    La etiqueta procede del directorio, no de una inspección semántica.
    Se calculan dos huellas: del archivo y de los píxeles RGB orientados.
    La segunda detecta contenido idéntico con metadatos distintos; ninguna
    de ellas detecta automáticamente imágenes solamente parecidas.
    """
    root = Path(class_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"No existe la carpeta: {root}")
    records = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative = path.relative_to(root).as_posix()
            row = {
                "image_id": sha256(relative.encode("utf-8")).hexdigest(),
                "relative_path": relative, "label": folder.name,
                "valid": False, "width": None, "height": None,
                "mode": None, "file_sha256": None, "pixel_sha256": None,
                "error": "",
            }
            try:
                row["file_sha256"] = sha256(path.read_bytes()).hexdigest()
                # verify comprueba integridad; hay que reabrir después.
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    row["mode"] = image.mode
                    rgb = ImageOps.exif_transpose(image).convert("RGB")
                    rgb.load()
                    row["width"], row["height"] = rgb.size
                    payload = f"{rgb.width}x{rgb.height}:RGB:".encode() + rgb.tobytes()
                    row["pixel_sha256"] = sha256(payload).hexdigest()
                row["valid"] = True
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            records.append(row)
    if not records:
        raise ValueError("No hay archivos con extensiones de imagen reconocidas.")
    return pd.DataFrame.from_records(records)


def curate_and_split(manifest, seed=42):
    """Depura registros y asigna train/val/test sin tocar archivos originales.

    Excluye contenido con etiquetas contradictorias y conserva la primera ruta
    alfabética de cada grupo restante. Divide cada clase por separado: 15 % para
    val, 15 % para test y el resto para train. El redondeo implica proporciones
    aproximadas. La semilla permite repetir la asignación con el mismo manifiesto.
    """
    records = manifest.sort_values("relative_path").copy().reset_index(drop=True)
    if records["image_id"].duplicated().any():
        raise ValueError("Los identificadores de imagen deben ser únicos.")
    valid = records.loc[records["valid"]].copy()
    if valid["pixel_sha256"].isna().any():
        raise ValueError("Una imagen válida debe tener huella de píxeles.")
    counts = valid.groupby("pixel_sha256")["label"].nunique()
    conflicts = counts.index[counts > 1]
    records["exclusion_reason"] = ""
    records.loc[~records["valid"], "exclusion_reason"] = "archivo_invalido"
    records.loc[records["valid"] & records["pixel_sha256"].isin(conflicts),
                "exclusion_reason"] = "etiquetas_contradictorias"
    eligible = records.loc[records["exclusion_reason"] == ""]
    duplicate_indices = eligible.index[eligible.duplicated("pixel_sha256", keep="first")]
    records.loc[duplicate_indices, "exclusion_reason"] = "copia_repetida"
    records["included"] = records["exclusion_reason"] == ""
    records["split"] = "excluded"
    rng = np.random.default_rng(seed)
    for label, group in records.loc[records["included"]].groupby("label", sort=True):
        n = len(group)
        if n < 3:
            raise ValueError(f"La clase {label} necesita al menos tres imágenes únicas.")
        order = rng.permutation(group.index.to_numpy())
        n_val = max(1, int(round(0.15 * n)))
        n_test = max(1, int(round(0.15 * n)))
        records.loc[order[:n_val], "split"] = "val"
        records.loc[order[n_val:n_val + n_test], "split"] = "test"
        records.loc[order[n_val + n_test:], "split"] = "train"
    if records.loc[records["included"]].empty:
        raise ValueError("No quedan imágenes después de la depuración.")
    # Comprobación contra fuga: un contenido solo puede pertenecer a una partición.
    selected = records.loc[records["included"]]
    if selected.groupby("pixel_sha256")["split"].nunique().max() > 1:
        raise RuntimeError("Hay contenido compartido entre particiones.")
    return records
