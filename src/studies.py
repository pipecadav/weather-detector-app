"""Estudio de tamaño/costo de HOG sobre entrenamiento, sin selección con test."""
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import hog
import cv2

from src.preprocessing import ImageConfig, preprocess, read_rgb


def study_hog(class_root, selected, output_dir):
    """Comparar celdas 4/8/16 y reportar dimensión/tiempo, no accuracy.

    Usamos dos imágenes por clase en entrenamiento y la ROI global. Esto
    ilustra el compromiso entre detalle y costo; no demuestra qué tamaño clasifica
    mejor. La referencia de experimentos conserva 8×8.
    """
    samples=selected.loc[selected['split']=='train'].groupby('label',sort=True).sample(n=2,random_state=42)
    grays=[]
    for row in samples.itertuples():
        global_roi=preprocess(read_rgb(Path(class_root)/row.relative_path),ImageConfig())['global_roi']
        grays.append(cv2.cvtColor(global_roi['rgb'],cv2.COLOR_RGB2GRAY))
    rows=[]
    for cell_size in (4,8,16):
        timings=[]
        for gray in grays:
            start=time.perf_counter()
            vector=hog(gray,orientations=9,pixels_per_cell=(cell_size,cell_size),
                       cells_per_block=(2,2),block_norm='L2-Hys')
            timings.append(time.perf_counter()-start)
        rows.append({'pixels_per_cell':cell_size,'dimension':len(vector),
                     'mean_ms':1000*np.mean(timings),'sample_count':len(grays)})
    result=pd.DataFrame(rows)
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    result.to_csv(output/'hog_parameters.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(result.pixels_per_cell.astype(str),result.dimension)
    axes[0].set(xlabel='Lado de celda (px)',ylabel='Dimensión HOG')
    axes[1].bar(result.pixels_per_cell.astype(str),result.mean_ms)
    axes[1].set(xlabel='Lado de celda (px)',ylabel='Tiempo medio HOG (ms)')
    fig.suptitle('Estudio de costo sobre 22 imágenes de entrenamiento')
    fig.tight_layout();fig.savefig(output/'hog_parameters.png',dpi=150);plt.close(fig)
    return result
