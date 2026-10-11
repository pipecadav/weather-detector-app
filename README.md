# Clasificación de fenómenos meteorológicos — Parte 2

El proyecto clasifica fotografías del dataset `jehanbhathena/weather-dataset`.
Cada imagen es una muestra; las regiones segmentadas serán entradas intermedias
para los descriptores. Gabor (24 características de energía) es el descriptor
principal y HOG se incluirá como comparación/complemento.

## Estado actual

Implementado: auditoría, depuración, particiones 70/15/15, preprocesamiento,
galerías Otsu/adaptativo, descriptores Gabor/HOG/LBP/color/forma, comparación
SVM/Random Forest, evaluación y API de inferencia sin interfaz. Las métricas
solo se presentan cuando termina la ejecución real. La aplicación y la interfaz
siguen pendientes para el final, junto con el informe y la sustentación.

La auditoría ejecutada del dataset registró 6862 imágenes válidas, 372 involucradas
en duplicación de píxeles y un grupo con etiquetas contradictorias (fogsmog y
sandstorm). Conservar una copia por contenido y excluir el grupo contradictorio
deja 6671 imágenes: 189 copias y 2 registros contradictorios se excluyen, sin borrar
archivos. Los manifiestos registran la decisión por imagen. Las huellas no detectan
automáticamente imágenes casi iguales. Las cifras se recalculan al ejecutar.

## Ejecución local

1. Instalar las dependencias: `python -m pip install -r requirements.txt`.
2. Abrir `Parte_2.ipynb` con un kernel Python del mismo entorno.
3. Ejecutar desde la raíz del proyecto, en orden. La primera celda comprueba rutas.
4. Dejar `LOCAL_DATASET_PATH = None` para descargar mediante KaggleHub, o indicar
   una carpeta ya descargada. No colocar credenciales en el notebook.
5. Inspeccionar los conteos, errores y duplicados antes de crear particiones.

En Colab, cargar también la carpeta `src/` en el directorio de trabajo o clonar
este repositorio. La instalación indicada en el notebook usa el kernel activo.
KaggleHub mantiene los archivos descargados en su propia caché: no se copian al
repositorio. El registro guarda rutas relativas y la ubicación de la fuente en
un archivo de configuración local.

## Estudio

Cada bloque del notebook explica objetivo, decisiones, interpretación y preguntas.
Las etiquetas de las carpetas son etiquetas proporcionadas por la fuente; una
auditoría de archivos no verifica que el fenómeno visible esté bien etiquetado.
La adaptación de detección múltiple a clasificación global se documentará en el
informe. La Parte 1 permanece independiente.

## Etapas largas y resultados

Desde la raíz, con el entorno del proyecto activo:

```powershell
python run_part2.py preview
python run_part2.py extract
python run_part2.py train
```

`extract` guarda checkpoints por lote de 128; reanuda si configuración e
identificadores coinciden. Si cambias código o archivos fuente, usa una nueva
carpeta de caché para evitar reutilizar descriptores obsoletos. No se detectan
automáticamente cambios de implementación en checkpoints ya escritos.

Resultados: `reports/experiments/`, modelo: `models/weather_model.joblib`,
características: `data/features/weather_features.npz`. No se versionan datos
pesados ni modelos en Git por defecto; conservarlos para la entrega.

La comparación usa cinco experimentos: Gabor, HOG, LBP, color+forma y todas las
familias. Se elige por F1 macro de validación; después se evalúa prueba. PCA de
HOG tiene un presupuesto inicial de 128 componentes; se reporta su varianza real.
El conjunto de prueba no se usa para reoptimizar. Las regiones son de intensidad,
no objetos meteorológicos confirmados. La ROI global preserva el contexto.

```python
from src.inference import predict_image
result = predict_image("mi_imagen.jpg", "models/weather_model.joblib")
```

El modelo debe viajar junto con `src/` y `requirements-lock.txt`. Cargar solo
archivos joblib propios y de confianza. Para respetar versiones del experimento:
`python -m pip install -r requirements-lock.txt`. La configuración exportada
registra además las versiones principales usadas en extracción.

Pruebas de invariantes: `python -m unittest discover -s tests -v`.
Si Jupyter conserva módulos antiguos, reinicia kernel o usa `importlib.reload`.

## Ejecución inicial completada

Ganador fijado con validación: SVM RBF C=10, gamma=scale, todas las familias.
En prueba obtuvo accuracy de 70,4 % y F1 macro de 0,6765. Son resultados iniciales,
no evidencia suficiente de robustez con nuevas cámaras. El conjunto de prueba
se evaluó tras fijar parámetros y no se usó para volver a optimizar.

PCA conservó 128 componentes con whitening y 55,2 % de varianza HOG. En validación,
la combinación mejoró respecto a Gabor solo. Hay errores de confianza alta y
sensibilidad a cambios de iluminación en una muestra pequeña.

Para regenerar figuras de evaluaciones ya existentes:

```python
from src.reporting import build_reports
build_reports(".")
```

Diagnósticos adicionales (se ejecutan con validación, sin cambiar el modelo):

```python
from src.diagnostics import diagnose
diagnose(".")
```

La interfaz sigue pendiente. También quedan interpretación propia del equipo,
pruebas con fotografías nuevas y redacción de los entregables académicos.
