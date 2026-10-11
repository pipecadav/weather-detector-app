# Resultados iniciales de la Parte 2

Se procesaron 6671 imágenes únicas: 4671 de entrenamiento, 1000 de validación
y 1000 de prueba. Se compararon cinco representaciones con SVM y Random Forest.
Gabor (24 estadísticas de energía por ROI) fue la referencia principal.

## Selección y resultado
El ganador se eligió por F1 macro en validación **antes** de evaluar prueba:
**E5_Combinadas / SVM**.
- F1 macro en validación: 0.7406.
- Accuracy en prueba: 0.7040.
- F1 macro en prueba: 0.6765.
- F1 ponderado en prueba: 0.7025.
- Tiempo medio de extracción en prueba: 0.113 s
  (ejecución concurrente; no es un benchmark aislado de una app).

HOG pasó de 8100 valores a 128 componentes PCA con whitening, reteniendo
55.2% de varianza. Es un presupuesto inicial de dimensión, no 95 % garantizado.
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

## Diagnósticos adicionales (validación)
La curva del clasificador con representaciones fijas pasó de F1 macro de validación
0,604 (1167 imágenes) a 0,677 (2335) y 0,741 (4671). El F1 de entrenamiento
se mantuvo cerca de 1: hay una brecha compatible con sobreajuste. La curva mantiene
PCA/escalado de train completo, por lo que no es una curva de todo el pipeline.

En 33 imágenes de validación, tres por clase, el brillo multiplicado por 1,3
redujo accuracy de 0,879 a 0,606. Esto evidencia sensibilidad en esa muestra, no
estima la tasa de acierto en nuevas cámaras. La precisión aparente del grupo
original no sustituye la evaluación de las 1000 imágenes de prueba.

Ver `learning_curve.csv`, `illumination_robustness.csv` y `robustness_samples.csv`.
La estimación de tiempo de robustez excluye lectura y carga del modelo.
