# Taller Redes Neuronales V — Reconocimiento facial mediante arquitecturas siamesas

**Universidad de Cundinamarca · Especialización en Inteligencia Artificial · Deep Learning – Conceptos (2026)**
Autor: Diego Yanzir Gutiérrez Camelo

Continuación de [Taller Redes Neuronales III](https://github.com/DiegoYanzir/Taller-Redes-Neuronales-III). Este
proyecto implementa un sistema de **verificación y reconocimiento facial** con una **red neuronal siamesa** entrenada
con **Triplet Loss** en **TensorFlow + Keras**. La red proyecta cada rostro en un *embedding* de 128 dimensiones con
norma L2 y decide si dos fotos son de la misma persona comparando su distancia con un umbral. Se evalúa en escenarios
**one-shot** (una foto de referencia) y **few-shot** (k = 3 y k = 5 fotos) sobre identidades que el modelo nunca vio.

## Tecnologías

| Herramienta | Uso |
|---|---|
| Python 3.12 | Lenguaje |
| TensorFlow 2.x / Keras 3 | Red de embeddings (MobileNetV2), modelo siamés y bucle de entrenamiento con `GradientTape` |
| MTCNN | Detección del rostro y de 5 puntos faciales (ojos, nariz, boca) |
| OpenCV | Lectura, conversión BGR→RGB, alineación, recorte, CLAHE; Haar Cascades como detector de respaldo |
| scikit-learn | Dataset complementario LFW, curva ROC/AUC y t-SNE |
| NumPy + Matplotlib | Minería de tripletas y gráficas |
| Google Colab + Google Drive | Entorno de ejecución y almacenamiento del dataset propio |

## Notebook

| # | Notebook | Contenido | Colab |
|---|---|---|---|
| 01 | `01_reconocimiento_facial_red_siamesa.ipynb` | Estructura del dataset en Drive (y captura con webcam), preprocesamiento facial (MTCNN, alineación, recorte 160×160, CLAHE, Data Augmentation), tripletas A/P/N con minería *semi-hard*, red siamesa con MobileNetV2 + embedding L2 de 128 D, Triplet Loss explícita, umbral óptimo, ROC, verificación de pares, one-shot y few-shot, t-SNE | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DiegoYanzir/Taller-Redes-Neuronales-V/blob/main/notebooks/01_reconocimiento_facial_red_siamesa.ipynb) |

## Dataset y consideraciones éticas

- **Fotos propias:** fotografías del autor y de participantes que otorgaron **consentimiento informado explícito para
  uso exclusivamente académico**. Se guardan en Google Drive (`Redes_Neuronales_V/dataset_facial/fotos_propias/<persona>/`)
  y **no se publican en este repositorio**: la carpeta `dataset_facial/` está en `.gitignore`. Los rasgos faciales son
  datos biométricos sensibles (Ley 1581 de 2012, art. 5).
- **Complemento:** 62 identidades × 20 fotos de *Labeled Faces in the Wild* (LFW), conjunto público de investigación en
  verificación facial, descargado automáticamente con scikit-learn. La red necesita ver muchas identidades para
  aprender un espacio métrico que generalice.
- **Partición por identidad:** el 25 % de las identidades LFW se reserva por completo para la prueba (one-shot real);
  de cada persona propia, la mitad de sus fotos entrena y la otra mitad se usa como soporte/consulta en la prueba.

## Resumen de resultados

Ejecución de verificación en CPU con semilla 42, **solo con identidades LFW** (sin fotos propias): entrenamiento con
736 imágenes de 46 identidades; prueba con
16 identidades **nunca vistas** (6080 pares). Al agregar fotos propias las cifras cambian.

### Verificación (conjunto de prueba, umbral elegido en validación)

| Métrica | Antes de entrenar | Después de Triplet Loss |
|---|---|---|
| Umbral τ (distancia euclidiana²) | 0,466 | 1,056 |
| Distancia intra-clase media | 0,494 | 0,881 |
| Distancia inter-clase media | 0,595 | 1,586 |
| Exactitud | 63,1 % | **77,4 %** |
| AUC ROC | 0,695 | **0,859** |

Mejores pesos en la época 27 de 27 (15,5 min en CPU).

### One-shot y Few-shot (16 identidades en la galería, 30 episodios)

| Escenario | Identificación (rank-1) | Verificación con τ |
|---|---|---|
| One-shot (k = 1) | 38,2 % ± 4,2 % | 76,7 % ± 2,7 % |
| Few-shot (k = 3) | 53,4 % ± 2,9 % | 82,2 % ± 1,7 % |
| Few-shot (k = 5) | 59,3 % ± 2,8 % | 83,3 % ± 1,3 % |

Identificación por azar: 6,2 %.

## Cómo ejecutar

**Google Colab (recomendado):**

1. Cree en su Google Drive la carpeta `Redes_Neuronales_V/dataset_facial/fotos_propias/` y dentro una subcarpeta por
   persona (`Diego_Gutierrez/`, `Participante_2/`, …) con **12–20 fotos** cada una (un solo rostro, distintos gestos,
   ángulos leves e iluminación). También puede capturarlas con la celda 1.2 (cámara web).
2. Abra el notebook con el badge *Open in Colab*, active *Entorno de ejecución → Cambiar tipo de entorno de
   ejecución → GPU (T4)* y ejecute *Entorno de ejecución → Ejecutar todo*. Autorice el acceso a Drive cuando se pida.
3. Las figuras y `metricas.json` quedan en `Redes_Neuronales_V/results/` de su Drive.

Si `fotos_propias/` está vacía, el notebook corre igual en modo demostración (solo LFW).

**Local:**

```bash
git clone https://github.com/DiegoYanzir/Taller-Redes-Neuronales-V.git
cd Taller-Redes-Neuronales-V
pip install -r requirements.txt
python scripts/01_reconocimiento_facial_red_siamesa.py     # crea dataset_facial/ y results/ en la carpeta actual
```

## Estructura del repositorio

```
Taller-Redes-Neuronales-V/
├── notebooks/                 # Notebook listo para Google Colab
├── scripts/                   # Mismo código en .py (formato "percent": celdas con # %%)
├── results/
│   ├── figuras/               # Preprocesamiento, tripletas, convergencia, distancias/ROC, verificación, one/few-shot, t-SNE
│   ├── metricas.json          # Configuración, particiones, historial, umbral y métricas
│   └── salida_*.txt           # Salida de consola de la ejecución
├── requirements.txt
└── README.md
```

## Referencias

- Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. *CVPR*, 815–823.
- Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese neural networks for one-shot image recognition. *ICML Deep Learning Workshop*.
- Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. *IEEE Signal Processing Letters, 23*(10), 1499–1503.
- Sandler, M. et al. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *CVPR*, 4510–4520.
- Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical networks for few-shot learning. *NeurIPS 30*.
- Huang, G. B. et al. (2007). *Labeled Faces in the Wild* (Tech. Rep. 07-49). University of Massachusetts, Amherst.
- Jaiswal, A. (2022). [*Face recognition system using Python*](https://www.analyticsvidhya.com/blog/2022/04/face-recognition-system-using-python/). Analytics Vidhya.
- [*What are Siamese neural networks in deep learning?*](https://towardsdatascience.com/what-are-siamese-neural-networks-in-deep-learning-bb092f749dcb/) (s.f.). Towards Data Science.
