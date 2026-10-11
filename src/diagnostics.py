"""Diagnósticos con validación: curvas de aprendizaje y cambios de iluminación."""
from pathlib import Path
import time

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from src.experiments import combine, transform_blocks
from src.features import extract_image_features
from src.preprocessing import ImageConfig, read_rgb


def diagnose(project_root):
    """Analizar el ganador fijado sin utilizar prueba ni elegir nuevos parámetros."""
    root=Path(project_root)
    bundle=joblib.load(root/'models/weather_model.joblib')
    output=root/'reports/experiments'
    with np.load(root/'data/features/weather_features.npz',allow_pickle=False) as data:
        blocks={name:data[name].copy() for name in bundle['families']}
        labels,splits=data['labels'].copy(),data['splits'].copy()
    transformed=transform_blocks(blocks,bundle['transforms'])
    x=combine(transformed,bundle['families'])
    train_indices=np.flatnonzero(splits=='train')
    val=splits=='val'
    learning=[]
    for fraction in (.25,.5,1.0):
        indices=train_indices
        if fraction<1:
            indices,_=train_test_split(train_indices,train_size=fraction,
                                      stratify=labels[train_indices],random_state=42)
        model=clone(bundle['model'])
        if 'probability' in model.get_params():
            model.set_params(probability=False)  # Solo predicción: no recalibrar.
        with threadpool_limits(limits=4):
            model.fit(x[indices],labels[indices])
        learning.append({'train_size':len(indices),
                         'train_f1_macro':f1_score(labels[indices],model.predict(x[indices]),average='macro'),
                         'validation_f1_macro':f1_score(labels[val],model.predict(x[val]),average='macro')})
    curve=pd.DataFrame(learning)
    curve.to_csv(output/'learning_curve.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(curve.train_size,curve.train_f1_macro,'o-',label='Entrenamiento')
    ax.plot(curve.train_size,curve.validation_f1_macro,'o-',label='Validación fija')
    ax.set(xlabel='Imágenes de entrenamiento',ylabel='F1 macro',ylim=(0,1),
           title='Curva con transformaciones ajustadas sobre train completo')
    ax.legend();fig.tight_layout();fig.savefig(output/'learning_curve.png',dpi=150);plt.close(fig)
    # Esta curva estudia el clasificador, no reajusta PCA por cada tamaño.
    # Por tanto no equivale a una curva end-to-end de todo el pipeline.
    manifest=pd.read_csv(root/'data/manifests/curated_split.csv')
    sample=manifest.loc[manifest.included & (manifest.split=='val')].groupby('label',sort=True).sample(n=3,random_state=42)
    config=ImageConfig(**bundle['extraction']['image_config'])
    source=Path(bundle['extraction']['source_root'])
    conditions={'original':lambda rgb:rgb,
                'brillo_0.7':lambda rgb:np.clip(rgb.astype(float)*.7,0,255).astype(np.uint8),
                'brillo_1.3':lambda rgb:np.clip(rgb.astype(float)*1.3,0,255).astype(np.uint8),
                'gamma_0.6':lambda rgb:np.clip(255*(rgb.astype(float)/255)**.6,0,255).astype(np.uint8),
                'gamma_1.6':lambda rgb:np.clip(255*(rgb.astype(float)/255)**1.6,0,255).astype(np.uint8)}
    results=[]
    for condition,change in conditions.items():
        predictions,times=[],[]
        for row in sample.itertuples():
            image=change(read_rgb(source/row.relative_path))
            start=time.perf_counter()
            features=extract_image_features(image,config)
            inputs={name:features[name][None,:] for name in bundle['families']}
            vector=combine(transform_blocks(inputs,bundle['transforms']),bundle['families'])
            predictions.append(bundle['model'].predict(vector)[0])
            times.append(time.perf_counter()-start)
        results.append({'condition':condition,'sample_count':len(sample),
                        'accuracy':accuracy_score(sample.label,predictions),
                        'f1_macro':f1_score(sample.label,predictions,average='macro'),
                        'mean_end_to_end_seconds':np.mean(times)})
        print('Robustez:',condition,results[-1],flush=True)
    robustness=pd.DataFrame(results)
    robustness.to_csv(output/'illumination_robustness.csv',index=False)
    sample[['image_id','relative_path','label']].to_csv(output/'robustness_samples.csv',index=False)
    return curve,robustness
