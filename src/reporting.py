"""Figuras y lectura de resultados reales; no modifica ni selecciona modelos."""
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from src.preprocessing import read_rgb


def build_reports(project_root):
    """Crear figuras exportables a partir de la evaluación ya fijada."""
    root=Path(project_root)
    output=root/'reports/experiments'
    bundle=joblib.load(root/'models/weather_model.joblib')
    table=pd.read_csv(output/'test_comparison.csv')
    for filename,title in [('confusion_counts','Matriz de confusión: conteos'),
                           ('confusion_normalized','Matriz de confusión: proporción por clase')]:
        matrix=pd.read_csv(output/f'{filename}.csv',index_col=0)
        fig,ax=plt.subplots(figsize=(11,9))
        ConfusionMatrixDisplay(matrix.to_numpy(),display_labels=matrix.columns).plot(
            ax=ax,xticks_rotation=45,values_format='.2f' if 'normalized' in filename else 'd',
            cmap='Blues',colorbar=False)
        ax.set_title(title)
        ax.set_xlabel('Clase predicha')
        ax.set_ylabel('Clase real')
        fig.tight_layout(); fig.savefig(output/f'{filename}.png',dpi=150); plt.close(fig)
    variance=pd.read_csv(output/'hog_pca_variance.csv')
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(variance.component,variance.cumulative_variance)
    ax.set(xlabel='Componentes PCA de HOG',ylabel='Varianza acumulada explicada',ylim=(0,1),
           title='PCA ajustado únicamente con entrenamiento')
    fig.tight_layout(); fig.savefig(output/'hog_pca_variance.png',dpi=150); plt.close(fig)
    with np.load(output/'training_projection.npz',allow_pickle=False) as data:
        fig,ax=plt.subplots(figsize=(10,6))
        for label in np.unique(data['labels']):
            keep=data['labels']==label
            ax.scatter(data['xy'][keep,0],data['xy'][keep,1],s=8,alpha=.45,label=label)
        ax.set(xlabel='PCA 1',ylabel='PCA 2',title='Proyección 2D de entrenamiento (modelo seleccionado)')
        ax.legend(bbox_to_anchor=(1.02,1),loc='upper left'); fig.tight_layout()
        fig.savefig(output/'training_projection.png',dpi=150); plt.close(fig)
    importance=pd.read_csv(output/'family_importance.csv').sort_values('f1_drop_mean')
    fig,ax=plt.subplots(figsize=(8,4))
    ax.barh(importance.family,importance.f1_drop_mean,xerr=importance.f1_drop_std)
    ax.set(xlabel='Caída de F1 macro al permutar',title='Importancia por familias en validación')
    fig.tight_layout(); fig.savefig(output/'family_importance.png',dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    table.pivot(index='experiment',columns='classifier',values='f1_macro').plot.bar(ax=ax)
    ax.set(ylabel='F1 macro en prueba',ylim=(0,1),title='Experimentos fijados antes de evaluar prueba')
    plt.xticks(rotation=20); fig.tight_layout(); fig.savefig(output/'test_comparison.png',dpi=150); plt.close(fig)
    predictions=pd.read_csv(output/'test_predictions.csv')
    errors=predictions.loc[~predictions.correct].sort_values('predicted_probability',ascending=False).head(12)
    source=Path(bundle['extraction']['source_root'])
    if not errors.empty:
        fig,axes=plt.subplots(3,4,figsize=(14,10))
        for ax in axes.flat:
            ax.axis('off')
        for ax,(_,row) in zip(axes.flat,errors.iterrows()):
            image=read_rgb(source/row.relative_path)
            ax.imshow(image)
            ax.set_title(f'Real: {row.true_label}\nPred: {row.predicted_label} ({row.predicted_probability:.2f})',fontsize=9)
        fig.tight_layout(); fig.savefig(output/'errors.png',dpi=120); plt.close(fig)
    winner=table.loc[table.selected_for_delivery].iloc[0]
    retained=float(variance.cumulative_variance.iloc[-1])
    summary=f'''# Resultados iniciales de la Parte 2

Se procesaron 6671 imágenes únicas: 4671 de entrenamiento, 1000 de validación
y 1000 de prueba. Se compararon cinco representaciones con SVM y Random Forest.
Gabor (24 estadísticas de energía por ROI) fue la referencia principal.

## Selección y resultado
El ganador se eligió por F1 macro en validación **antes** de evaluar prueba:
**{bundle['experiment']} / {bundle['classifier']}**.
- F1 macro en validación: {winner.validation_f1_macro:.4f}.
- Accuracy en prueba: {winner.accuracy:.4f}.
- F1 macro en prueba: {winner.f1_macro:.4f}.
- F1 ponderado en prueba: {winner.f1_weighted:.4f}.
- Tiempo medio de extracción en prueba: {predictions.extraction_seconds.mean():.3f} s
  (ejecución concurrente; no es un benchmark aislado de una app).

HOG pasó de 8100 valores a {len(variance)} componentes PCA con whitening, reteniendo
{retained:.1%} de varianza. Es un presupuesto inicial de dimensión, no 95 % garantizado.
Escalado y PCA se ajustaron solo con entrenamiento. No se hizo validación cruzada;
se usó el conjunto de validación independiente para una búsqueda moderada.

## Interpretación y límites
Las máscaras representan regiones de intensidad, no detecciones confirmadas del
fenómeno. Incluimos siempre una ROI global para conservar escena y contexto.
Los descriptores por ROI se promedian; esa agregación pierde ubicación espacial.
Se eliminaron duplicados exactos del experimento, pero no se verificaron imágenes
casi iguales, procedencia por cámara/día ni generalización a fotografías propias.
El modelo siempre elige una de las once clases, incluso fuera del dominio.
Las probabilidades estimadas no garantizan acierto. Los momentos HSV de matiz no
resuelven su periodicidad. El filtro bilateral procede de la propuesta de la Parte 1;
no se demostró aquí que sea el mejor filtro para clasificación.

La comparación en prueba es descriptiva: no debe usarse para volver a seleccionar
hiperparámetros. Nuevas mejoras requerirán mantener separado un conjunto final.
La galería de errores contiene los doce errores de mayor probabilidad estimada:
hay que inspeccionarlos antes de atribuir causas (textura, iluminación o etiqueta).

## Qué estudiar ahora
1. Leer la tabla: ¿gana el mismo modelo en todas las familias?
2. Comparar diagonales y confusiones fuera de la diagonal.
3. ¿Qué familias reducen más el F1 al permutarlas?
4. ¿Por qué una proyección 2D mezclada no demuestra que el clasificador falle?
5. Identificar tres errores y proponer causas como hipótesis, no como certezas.

La aplicación e interfaz quedan pendientes por decisión del equipo.
'''
    (output/'RESULTADOS.md').write_text(summary,encoding='utf-8')
    return summary
