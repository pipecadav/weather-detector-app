"""Experimentos reproducibles con selección en validación, nunca en prueba."""
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

EXPERIMENTS = {
    "E1_Gabor": ("gabor",),
    "E2_HOG": ("hog",),
    "E3_LBP": ("lbp",),
    "E4_Color_Forma": ("color", "shape"),
    "E5_Combinadas": ("gabor", "hog", "lbp", "color", "shape"),
}


def fit_transforms(blocks, train_mask, seed=42, hog_components=128):
    """Ajustar un escalador por familia y PCA solo para HOG, usando train.

    128 componentes es un presupuesto inicial explícito, no una promesa de 95 %
    de varianza. El gráfico reportará la varianza realmente retenida. Compartir
    estas transformaciones fijas entre experimentos ahorra trabajo sin usar val/test.
    """
    transforms, transformed = {}, {}
    with threadpool_limits(limits=4):
        for name, values in blocks.items():
            scaler = StandardScaler().fit(values[train_mask])
            scaled = scaler.transform(values).astype(np.float32)
            pca = None
            if name == "hog":
                components = min(hog_components, train_mask.sum()-1, values.shape[1])
                # Whitening deja los componentes con varianza unitaria en train:
                # evita que los primeros dominen la escala del hiperparámetro C.
                pca = PCA(n_components=components, svd_solver="randomized", whiten=True, random_state=seed)
                pca.fit(scaled[train_mask])
                scaled = pca.transform(scaled).astype(np.float32)
            transforms[name] = {"scaler": scaler, "pca": pca}
            transformed[name] = scaled
    return transforms, transformed


def combine(blocks, families):
    """Concatenar en un orden fijo: cambiarlo altera lo que recibe el modelo."""
    return np.concatenate([blocks[name] for name in families], axis=1)


def transform_blocks(blocks, transforms):
    """Aplicar parámetros guardados; nunca ejecutar fit sobre imágenes nuevas."""
    out = {}
    for name, transform in transforms.items():
        scaled = transform["scaler"].transform(blocks[name])
        if transform["pca"] is not None:
            scaled = transform["pca"].transform(scaled)
        out[name] = scaled
    return out


def metrics(y, pred):
    return {"accuracy": accuracy_score(y, pred), "f1_macro": f1_score(y, pred, average="macro"),
            "f1_weighted": f1_score(y, pred, average="weighted")}


def train_experiments(blocks, labels, splits, report_dir, model_dir, extraction_metadata, seed=42):
    """Comparar modelos en validación, fijar ganador y luego evaluar prueba.

    Ajuste inicial moderado: SVM lineal/RBF y dos configuraciones RF. No hacemos
    CV porque es opcional y ya existe val independiente. Cada familia tiene una
    única transformación ajustada con train. El conjunto de prueba se consulta
    después de fijar configuraciones por F1 macro en val.
    """
    report_dir, model_dir = Path(report_dir), Path(model_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = (splits == name for name in ("train", "val", "test"))
    transforms, transformed = fit_transforms(blocks, train, seed)
    joblib.dump(transforms, model_dir / "transforms.joblib")
    search_rows, chosen = [], {}
    for experiment, families in EXPERIMENTS.items():
        x = combine(transformed, families)
        candidates = [
            ("SVM", {"kernel": "linear", "C": c}, SVC(kernel="linear", C=c,
                class_weight="balanced", random_state=seed)) for c in (0.1, 1.0)]
        candidates += [("SVM", {"kernel": "rbf", "C": c, "gamma": gamma},
                        SVC(kernel="rbf", C=c, gamma=gamma, class_weight="balanced",
                            random_state=seed)) for c in (1.0, 10.0) for gamma in ("scale", 0.01)]
        candidates += [("RandomForest", {"n_estimators": 200, "max_features": fraction},
                        RandomForestClassifier(n_estimators=200, max_features=fraction,
                            class_weight="balanced", random_state=seed, n_jobs=4))
                       for fraction in ("sqrt", 0.5)]
        for classifier, params, model in candidates:
            start = time.perf_counter()
            with threadpool_limits(limits=4):
                model.fit(x[train], labels[train])
            fit_seconds = time.perf_counter() - start
            score = metrics(labels[val], model.predict(x[val]))
            row = {"experiment": experiment, "classifier": classifier,
                   "parameters": str(params), "fit_seconds": fit_seconds, **score}
            search_rows.append(row)
            key = (experiment, classifier)
            if key not in chosen or score["f1_macro"] > chosen[key]["row"]["f1_macro"]:
                chosen[key] = {"model": model, "row": row, "families": families}
            print(experiment, classifier, params, "val F1=", round(score["f1_macro"], 4), flush=True)
        pd.DataFrame(search_rows).to_csv(report_dir / "validation_search.csv", index=False)
    # Elegir antes de consultar prueba. Desempate determinista por orden de búsqueda.
    winner_key = max(chosen, key=lambda key: chosen[key]["row"]["f1_macro"])
    winner = chosen[winner_key]
    model = winner["model"]
    if isinstance(model, SVC):
        # Calibración interna usando solo train. La selección no usa probabilidades.
        model.set_params(probability=True)
        with threadpool_limits(limits=4):
            model.fit(combine(transformed, winner["families"])[train], labels[train])
    bundle = {"model": model, "transforms": {n: transforms[n] for n in winner["families"]},
              "families": winner["families"], "experiment": winner_key[0],
              "classifier": winner_key[1], "validation_metrics": winner["row"],
              "classes": model.classes_.tolist(), "extraction": extraction_metadata,
              "seed": seed}
    joblib.dump(bundle, model_dir / "weather_model.joblib")
    results = []
    for (experiment, classifier), candidate in chosen.items():
        x_test = combine(transformed, candidate["families"])[test]
        start = time.perf_counter()
        pred = candidate["model"].predict(x_test)
        elapsed = time.perf_counter() - start
        results.append({"experiment": experiment, "classifier": classifier,
                        "validation_f1_macro": candidate["row"]["f1_macro"],
                        "prediction_ms_per_image": 1000*elapsed/test.sum(),
                        "selected_for_delivery": (experiment, classifier) == winner_key,
                        **metrics(labels[test], pred)})
    table = pd.DataFrame(results)
    table.to_csv(report_dir / "test_comparison.csv", index=False)
    x_test = combine(transformed, winner["families"])[test]
    pred, proba = model.predict(x_test), model.predict_proba(x_test)
    classes = model.classes_
    pd.DataFrame(classification_report(labels[test], pred, labels=classes,
        output_dict=True, zero_division=0)).T.to_csv(report_dir / "classification_report.csv")
    for normalize, filename in ((None, "confusion_counts.csv"), ("true", "confusion_normalized.csv")):
        matrix = confusion_matrix(labels[test], pred, labels=classes, normalize=normalize)
        pd.DataFrame(matrix, index=classes, columns=classes).to_csv(report_dir / filename)
    predicted_indices = np.array([np.flatnonzero(classes == p)[0] for p in pred])
    predictions = pd.DataFrame({"true_label": labels[test], "predicted_label": pred,
                                "predicted_probability": proba[np.arange(len(pred)), predicted_indices],
                                "correct": labels[test] == pred})
    predictions.to_csv(report_dir / "test_predictions.csv", index=False)
    np.savez_compressed(report_dir / "training_projection.npz",
                       xy=PCA(2, random_state=seed).fit_transform(combine(transformed, winner["families"])[train]),
                       labels=labels[train])
    pca = transforms["hog"]["pca"]
    pd.DataFrame({"component": np.arange(1, pca.n_components_+1),
                  "cumulative_variance": np.cumsum(pca.explained_variance_ratio_)}).to_csv(
                      report_dir / "hog_pca_variance.csv", index=False)
    # Importancia por familias sobre val: permutar juntas todas las columnas de
    # una familia preserva relaciones internas y revela su aporte al modelo.
    val_blocks = {n: values[val].copy() for n, values in transformed.items()}
    baseline = f1_score(labels[val], model.predict(combine(val_blocks, winner["families"])), average="macro")
    rng = np.random.default_rng(seed)
    importance = []
    for name in winner["families"]:
        drops = []
        for _ in range(5):
            altered = val_blocks.copy()
            altered[name] = val_blocks[name][rng.permutation(val.sum())]
            drops.append(baseline-f1_score(labels[val], model.predict(combine(altered, winner["families"])), average="macro"))
        importance.append({"family": name, "f1_drop_mean": np.mean(drops), "f1_drop_std": np.std(drops)})
    pd.DataFrame(importance).to_csv(report_dir / "family_importance.csv", index=False)
    return bundle, table
