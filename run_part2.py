"""Ejecutar etapas largas sin mantener el notebook ocupado: python run_part2.py ETAPA."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
import json
from pathlib import Path
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import extract_image_features, feature_names
from src.preprocessing import ImageConfig, preprocess, read_rgb

ROOT = Path(__file__).resolve().parent


def load_source():
    config = json.loads((ROOT / "data/manifests/audit_config.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(ROOT / "data/manifests/curated_split.csv")
    return Path(config["class_root"]), manifest.loc[manifest["included"]].reset_index(drop=True)


def preview():
    """Mostrar solo ejemplos de train; no inspeccionar prueba para diseñar máscaras."""
    source, rows = load_source()
    samples = rows.loc[rows["split"] == "train"].groupby("label", sort=True).sample(n=1, random_state=42)
    output = ROOT / "reports/preprocessing"
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    for method in ("otsu", "adaptive"):
        fig, axes = plt.subplots(len(samples), 4, figsize=(12, len(samples)*2.7))
        for i, row in enumerate(samples.itertuples()):
            processed = preprocess(read_rgb(source / row.relative_path), ImageConfig(segmentation=method))
            overlay = processed["rgb"].copy()
            for region in processed["regions"]:
                x, y, w, h = region["bbox"]
                cv2.rectangle(overlay, (x, y), (x+w-1, y+h-1), (255, 0, 0), 2)
            panels = [processed["rgb"], processed["filtered"], processed["binary"], overlay]
            for ax, image, title in zip(axes[i], panels, (row.label, "Bilateral", method, "Regiones visuales")):
                ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
                ax.set_title(title, fontsize=9)
                ax.axis("off")
            diagnostics.append({"relative_path": row.relative_path, "label": row.label,
                                "method": method, "regions": len(processed["regions"]),
                                "threshold": processed["threshold"]})
        fig.tight_layout()
        fig.savefig(output / f"{method}_train.png", dpi=100)
        plt.close(fig)
    pd.DataFrame(diagnostics).to_csv(output / "diagnostics.csv", index=False)
    print("Galerías de entrenamiento guardadas:", output, flush=True)


def extract():
    """Procesar el dataset con checkpoints por lote para poder reanudar."""
    source, rows = load_source()
    output = ROOT / "data/features"
    output.mkdir(parents=True, exist_ok=True)
    config = ImageConfig()
    names = feature_names()
    versions = {name: version(name) for name in
                ("numpy", "opencv-python-headless", "scikit-image", "scikit-learn", "joblib")}
    metadata = {"image_config": config.to_dict(), "feature_names": names,
                "versions": versions, "aggregation": "equal_mean_global_and_up_to_3_regions",
                "source_root": str(source)}
    signature = json.dumps(metadata, sort_keys=True)
    frames = rows.to_dict("records")
    cv2.setNumThreads(1)  # Evitar que cada trabajador abra más hilos.

    def process(row):
        start = time.perf_counter()
        try:
            features = extract_image_features(read_rgb(source / row["relative_path"]), config)
        except Exception as exc:
            raise RuntimeError(f"No se pudo procesar {row['relative_path']}: {exc}") from exc
        return features, time.perf_counter()-start

    batches = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, len(frames), 128):
            batch_rows = frames[start:start+128]
            checkpoint = output / f"batch_{start:05d}.npz"
            ids = np.array([row["image_id"] for row in batch_rows])
            if checkpoint.exists():
                with np.load(checkpoint, allow_pickle=False) as old:
                    reusable = str(old["signature"]) == signature and np.array_equal(old["image_ids"], ids)
                if not reusable:
                    raise RuntimeError(f"Checkpoint incompatible: {checkpoint}. Usa otra carpeta o revisa configuración.")
            else:
                results = list(pool.map(process, batch_rows))
                arrays = {name: np.stack([f[name] for f, _ in results]) for name in names}
                arrays.update(image_ids=ids, signature=np.array(signature),
                              seconds=np.array([elapsed for _, elapsed in results]))
                np.savez_compressed(checkpoint, **arrays)
            batches.append(checkpoint)
            print(f"Características: {min(start+128,len(frames))}/{len(frames)}", flush=True)
    assembled = {name: [] for name in names}
    seconds = []
    for batch in batches:
        with np.load(batch, allow_pickle=False) as data:
            for name in names:
                assembled[name].append(data[name])
            seconds.append(data["seconds"])
    arrays = {name: np.concatenate(parts) for name, parts in assembled.items()}
    arrays.update(labels=rows["label"].to_numpy(dtype=str), splits=rows["split"].to_numpy(dtype=str),
                  image_ids=rows["image_id"].to_numpy(dtype=str),
                  relative_paths=rows["relative_path"].to_numpy(dtype=str), seconds=np.concatenate(seconds))
    np.savez_compressed(output / "weather_features.npz", **arrays)
    (output / "feature_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Características completas:", {k: v.shape for k,v in arrays.items()}, flush=True)


def train():
    from src.experiments import train_experiments
    with np.load(ROOT / "data/features/weather_features.npz", allow_pickle=False) as data:
        blocks = {name: data[name].copy() for name in feature_names()}
        labels, splits = data["labels"].copy(), data["splits"].copy()
        paths, seconds = data["relative_paths"].copy(), data["seconds"].copy()
    metadata = json.loads((ROOT / "data/features/feature_config.json").read_text(encoding="utf-8"))
    bundle, table = train_experiments(blocks, labels, splits, ROOT / "reports/experiments", ROOT / "models", metadata)
    report = ROOT / "reports/experiments"
    predictions = pd.read_csv(report / "test_predictions.csv")
    predictions["relative_path"] = paths[splits == "test"]
    predictions["extraction_seconds"] = seconds[splits == "test"]
    predictions.to_csv(report / "test_predictions.csv", index=False)
    print("Ganador por validación:", bundle["experiment"], bundle["classifier"], flush=True)
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preview", "extract", "train"))
    args = parser.parse_args()
    {"preview": preview, "extract": extract, "train": train}[args.stage]()
