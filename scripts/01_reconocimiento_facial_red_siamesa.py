# %% [markdown]
# # Notebook 01 — Reconocimiento y verificación facial con una Red Neuronal Siamesa y Triplet Loss
#
# **Curso:** Deep Learning – Conceptos · Especialización en Inteligencia Artificial · Universidad de Cundinamarca
# **Tema (Semana 10):** redes siamesas, preprocesamiento para *Face Recognition*, Triplet Loss y aprendizaje
# *one-shot*, *few-shot* y *zero-shot*.
#
# Flujo del notebook:
#
# 1. **Entorno y dataset:** montaje de Google Drive y estructura de carpetas para el dataset facial propio
#    (fotografías tomadas **con consentimiento informado** y uso exclusivamente académico), captura opcional con
#    la cámara web de Colab y complemento con identidades públicas de *Labeled Faces in the Wild* (LFW).
# 2. **Preprocesamiento facial:** detección del rostro (MTCNN, con Haar Cascades de OpenCV como respaldo),
#    alineación por la línea de los ojos, recorte, redimensionamiento a 160×160, conversión BGR→RGB,
#    ecualización de iluminación (CLAHE), normalización y *Data Augmentation*.
# 3. **Particiones y tripletas:** identidades de entrenamiento / validación / prueba y organización de las
#    tripletas Ancla (A), Positiva (P) y Negativa (N).
# 4. **Arquitectura:** red generadora de *embeddings* (MobileNetV2 preentrenada + capa densa de 128 D con
#    normalización L2) y modelo siamés de tres ramas que **comparten los mismos pesos**.
# 5. **Triplet Loss** implementada explícitamente: $L(A,P,N)=\max(\lVert f(A)-f(P)\rVert^2-\lVert f(A)-f(N)\rVert^2+\alpha,\,0)$,
#    con bucle de entrenamiento propio (`tf.GradientTape`), minería de tripletas *semi-hard* y curva de convergencia.
# 6. **Evaluación:** umbral de decisión óptimo, distancias intra-clase vs. inter-clase, curva ROC, verificación
#    de identidad, **One-shot** (1 imagen de referencia) y **Few-shot** (k = 3 y k = 5 imágenes de soporte).
#
# > En Colab: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU (T4)*. En CPU el notebook
# > también corre completo (≈ 25–35 minutos).

# %%
# ============================================================
# 0. Importaciones, versiones, semillas y estilo de las gráficas
# ============================================================
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import fetch_lfw_people
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, roc_curve

EN_COLAB = "google.colab" in sys.modules
if EN_COLAB:
    # MTCNN no viene instalado en Colab (los demás paquetes sí)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mtcnn"], check=False)

SEED = 42
keras.utils.set_random_seed(SEED)        # fija las semillas de Python, NumPy y TensorFlow (reproducibilidad)
RNG = np.random.default_rng(SEED)

# Paleta fija: azul = misma identidad (intra-clase), naranja = identidades distintas (inter-clase)
COLOR_INTRA, COLOR_INTER = "#2a78d6", "#eb6834"
COLOR_OK, COLOR_NO = "#2e9e44", "#d1343a"
plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e4e0", "grid.linewidth": 0.8,
    "axes.edgecolor": "#8a8984", "axes.titleweight": "bold", "lines.linewidth": 2,
})

print("Python:", sys.version.split()[0], "| TensorFlow:", tf.__version__, "| Keras:", keras.__version__,
      "| OpenCV:", cv2.__version__)
print("GPU disponible:", tf.config.list_physical_devices("GPU"))
print("Ejecutando en Google Colab:", EN_COLAB)

# %% [markdown]
# ## 1. Configuración del entorno y estructura del dataset
#
# En Colab el dataset y los resultados se guardan en **Google Drive** para que sobrevivan al reinicio del entorno.
# La estructura de carpetas es:
#
# ```
# Redes_Neuronales_V/
# ├── dataset_facial/
# │   ├── fotos_propias/              ← SUS fotografías (con consentimiento), una carpeta por persona
# │   │   ├── Diego_Gutierrez/        (12–20 fotos: distintos ángulos, gestos e iluminación)
# │   │   └── Participante_2/
# │   ├── complemento_lfw/            ← identidades públicas de LFW (se descargan automáticamente)
# │   ├── procesado/                  ← rostros detectados, alineados y recortados a 160×160
# │   ├── tripletas_muestra/          ← ejemplos de tripletas (ancla, positiva, negativa)
# │   └── manifiesto_particiones.csv  ← imagen → persona → partición (entrenamiento/validación/prueba)
# └── results/
#     ├── figuras/                    ← todas las gráficas del notebook
#     ├── modelos/red_embedding.keras
#     └── metricas.json
# ```
#
# **Recomendaciones para las fotos propias:** al menos **12 fotos por persona** (ideal 15–20), un solo rostro por
# foto, de frente o con giro leve, con variaciones de expresión, iluminación, fondo, gafas, peinado. Cada persona
# que aparece en el dataset debe haber firmado o aceptado un **consentimiento informado** para uso académico.

# %%
# ============================================================
# 1.1 Rutas de trabajo (Google Drive en Colab, carpeta local fuera de Colab)
# ============================================================
if EN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    DIR_BASE = Path("/content/drive/MyDrive/Redes_Neuronales_V")
else:
    DIR_BASE = Path(os.environ.get("RNV_DIR_BASE", Path.cwd()))

DIR_DATASET = DIR_BASE / "dataset_facial"
DIR_PROPIAS = DIR_DATASET / "fotos_propias"
DIR_LFW = DIR_DATASET / "complemento_lfw"
DIR_PROCESADO = DIR_DATASET / "procesado"
DIR_TRIPLETAS = DIR_DATASET / "tripletas_muestra"
DIR_RESULTADOS = DIR_BASE / "results"
DIR_FIG = DIR_RESULTADOS / "figuras"
DIR_MODELOS = DIR_RESULTADOS / "modelos"

for carpeta in (DIR_PROPIAS, DIR_LFW, DIR_PROCESADO, DIR_TRIPLETAS, DIR_FIG, DIR_MODELOS):
    carpeta.mkdir(parents=True, exist_ok=True)

# ---------------- Hiperparámetros y parámetros del experimento ----------------
TAM_ROSTRO = 160                 # tamaño estándar de entrada (FaceNet usa 160×160)
MARGEN_RECORTE = 0.20            # margen alrededor de la caja del rostro (20 %)
ECUALIZAR_ILUMINACION = True     # CLAHE sobre el canal L (espacio LAB)
LFW_MAX_PERSONAS = 62            # identidades públicas de complemento (todas las de LFW con ≥ 20 fotos)
LFW_FOTOS_POR_PERSONA = 20
FRACCION_IDS_PRUEBA_LFW = 0.25   # identidades LFW que el modelo NUNCA ve en entrenamiento (one-shot real)
FRACCION_PRUEBA_PROPIAS = 0.5    # de cada persona propia: 50 % entrena, 50 % se reserva para prueba
FRACCION_VALIDACION = 0.2        # de las imágenes de entrenamiento, 20 % para validación y elección del umbral
DIM_EMBEDDING = 128
ALPHA = 0.2                      # margen α de la Triplet Loss (valor de FaceNet)
TAM_LOTE = 32
TRIPLETAS_POR_EPOCA = 960
EPOCAS_FASE1 = 12                # backbone congelado: solo se entrena la cabeza de embedding
EPOCAS_FASE2 = 15                # ajuste fino de los últimos bloques de MobileNetV2
LR_FASE1, LR_FASE2 = 1e-3, 5e-5
CAPAS_AJUSTE_FINO = 40
MINERIA_DESDE_EPOCA = 3          # a partir de esta época se eligen negativos semi-hard
K_FEW_SHOT = (1, 3, 5)
EPISODIOS = 30

print("Carpeta base:", DIR_BASE.resolve())

# %% [markdown]
# ### 1.2 (Opcional) Captura de fotos con la cámara web en Colab
#
# Si no tiene las fotos en el computador, puede capturarlas directamente desde el navegador. Cambie
# `CAPTURAR_CON_WEBCAM = True`, escriba el nombre de la persona y ejecute la celda: se toma una foto por cada clic en
# **Capturar**. Varíe el gesto, el ángulo y la luz entre fotos. Solo funciona en Google Colab.

# %%
# ============================================================
# 1.2 Captura opcional con cámara web (solo Colab)
# ============================================================
CAPTURAR_CON_WEBCAM = False
PERSONA_A_CAPTURAR = "Diego_Gutierrez"
NUM_FOTOS_A_CAPTURAR = 15


def capturar_foto_webcam(ruta_salida, calidad=0.9):
    """Abre la cámara web del navegador, espera el clic en 'Capturar' y guarda la foto en `ruta_salida`."""
    from base64 import b64decode
    from IPython.display import Javascript, display
    from google.colab.output import eval_js

    display(Javascript("""
    async function tomarFoto(calidad) {
      const div = document.createElement('div');
      const boton = document.createElement('button');
      boton.textContent = 'Capturar';
      div.appendChild(boton);
      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});
      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
      await new Promise((resolve) => boton.onclick = resolve);
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', calidad);
    }"""))
    datos = eval_js(f"tomarFoto({calidad})")
    Path(ruta_salida).write_bytes(b64decode(datos.split(",")[1]))
    return ruta_salida


if CAPTURAR_CON_WEBCAM and EN_COLAB:
    carpeta_persona = DIR_PROPIAS / PERSONA_A_CAPTURAR
    carpeta_persona.mkdir(parents=True, exist_ok=True)
    inicio = len(list(carpeta_persona.glob("*.jpg")))
    for i in range(NUM_FOTOS_A_CAPTURAR):
        print(f"Foto {i + 1}/{NUM_FOTOS_A_CAPTURAR}: cambie levemente la pose y pulse 'Capturar'")
        capturar_foto_webcam(carpeta_persona / f"{PERSONA_A_CAPTURAR}_{inicio + i + 1:03d}.jpg")
    print("Fotos guardadas en", carpeta_persona)

# %% [markdown]
# ### 1.3 Inventario del dataset propio y complemento con LFW
#
# Una red siamesa aprende a **separar identidades**, de modo que necesita ver muchas personas distintas durante el
# entrenamiento. Con las fotos de 2–5 personas no basta para aprender un espacio métrico general, así que el dataset
# propio se complementa con 62 identidades públicas de **Labeled Faces in the Wild** (Huang et al., 2007), un
# conjunto de referencia creado para investigar la verificación facial. Las fotos LFW se exportan a disco como
# archivos JPG para que pasen por **exactamente el mismo pipeline** que las fotos propias.
#
# Si la carpeta `fotos_propias/` está vacía, el notebook se ejecuta igual (modo demostración con LFW).

# %%
# ============================================================
# 1.3 Inventario del dataset
# ============================================================
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def listar_imagenes(carpeta):
    return sorted(p for p in Path(carpeta).rglob("*") if p.suffix.lower() in EXTENSIONES)


def exportar_complemento_lfw(destino, max_personas, fotos_por_persona):
    """Descarga LFW (scikit-learn) y guarda `fotos_por_persona` JPG de las `max_personas` identidades más frecuentes."""
    if len(listar_imagenes(destino)) >= max_personas * fotos_por_persona:
        return
    lfw = fetch_lfw_people(min_faces_per_person=fotos_por_persona, resize=1.0, color=True,
                           funneled=True, slice_=None, download_if_missing=True)
    imagenes = lfw.images if lfw.images.max() > 1.5 else lfw.images * 255.0
    conteo = np.bincount(lfw.target)
    for id_persona in np.argsort(-conteo)[:max_personas]:
        nombre = lfw.target_names[id_persona].replace(" ", "_")
        carpeta = destino / nombre
        carpeta.mkdir(parents=True, exist_ok=True)
        indices = np.flatnonzero(lfw.target == id_persona)[:fotos_por_persona]
        for k, idx in enumerate(indices):
            bgr = cv2.cvtColor(imagenes[idx].astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(carpeta / f"{nombre}_{k + 1:03d}.jpg"), bgr)


exportar_complemento_lfw(DIR_LFW, LFW_MAX_PERSONAS, LFW_FOTOS_POR_PERSONA)

inventario = []   # (ruta, persona, origen)
for origen, carpeta_origen in (("propia", DIR_PROPIAS), ("lfw", DIR_LFW)):
    for carpeta_persona in sorted(p for p in carpeta_origen.iterdir() if p.is_dir()):
        for ruta in listar_imagenes(carpeta_persona):
            inventario.append((ruta, carpeta_persona.name, origen))

PERSONAS_PROPIAS = sorted({p for _, p, o in inventario if o == "propia"})
PERSONAS_LFW = sorted({p for _, p, o in inventario if o == "lfw"})
print(f"Imágenes encontradas: {len(inventario)}")
print(f"  Personas propias ({len(PERSONAS_PROPIAS)}):", PERSONAS_PROPIAS or "— ninguna (modo demostración con LFW)")
for persona in PERSONAS_PROPIAS:
    n = sum(1 for _, p, _ in inventario if p == persona)
    aviso = "" if n >= 12 else "  ← se recomiendan al menos 12 fotos"
    print(f"    {persona}: {n} fotos{aviso}")
print(f"  Identidades LFW de complemento: {len(PERSONAS_LFW)} × {LFW_FOTOS_POR_PERSONA} fotos")

# Muestra de fotografías originales (sin procesar)
muestra_personas = (PERSONAS_PROPIAS + PERSONAS_LFW)[:6]
fig, ejes = plt.subplots(len(muestra_personas), 5, figsize=(10, 2.1 * len(muestra_personas)))
for fila, persona in enumerate(muestra_personas):
    rutas = [r for r, p, _ in inventario if p == persona][:5]
    for col in range(5):
        eje = ejes[fila, col]
        eje.axis("off")
        if col < len(rutas):
            eje.imshow(cv2.cvtColor(cv2.imread(str(rutas[col])), cv2.COLOR_BGR2RGB))
    ejes[fila, 0].set_title(persona.replace("_", " "), loc="left", fontsize=9)
fig.suptitle("Fotografías originales del dataset (sin preprocesar)", fontweight="bold")
plt.tight_layout()
plt.savefig(DIR_FIG / "01_dataset_original.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Pipeline de preprocesamiento facial
#
# Cada imagen pasa por los siguientes pasos:
#
# | Paso | Operación | Motivo |
# |---|---|---|
# | 1 | Lectura con OpenCV y conversión **BGR → RGB** | OpenCV carga en BGR; MTCNN y la red esperan RGB |
# | 2 | **Detección** del rostro con MTCNN (caja + 5 puntos faciales); respaldo con Haar Cascades | La red debe ver solo la cara, no el fondo |
# | 3 | Selección del rostro de mayor área y confianza ≥ 0,90 | Descarta caras de fondo y falsos positivos |
# | 4 | **Alineación**: rotación que deja horizontal la línea de los ojos | Elimina la variación por inclinación de la cabeza |
# | 5 | **Recorte** cuadrado con 20 % de margen y **redimensionamiento** a 160×160 (`INTER_AREA`) | Tamaño fijo para la CNN |
# | 6 | **Ecualización de iluminación** CLAHE sobre el canal L (espacio LAB) | Reduce el efecto de sombras y luz desigual |
# | 7 | **Normalización** de píxeles a [-1, 1] (dentro del modelo) | Escala que espera MobileNetV2 |
# | 8 | **Data Augmentation** en el entrenamiento: flip horizontal, rotación ±9°, brillo y contraste ±15 % | Robustece un dataset pequeño |

# %%
# ============================================================
# 2.1 Detectores de rostro: MTCNN (principal) y Haar Cascades (respaldo)
# ============================================================
try:
    from mtcnn import MTCNN
    DETECTOR_MTCNN = MTCNN(device="GPU:0" if tf.config.list_physical_devices("GPU") else "CPU:0")
    DETECTOR_NOMBRE = "MTCNN"
except ImportError:
    DETECTOR_MTCNN = None
    DETECTOR_NOMBRE = "Haar Cascades (OpenCV)"

# Haar Cascades viene en OpenCV 4.x (el de Colab); OpenCV 5 lo trasladó a opencv-contrib
if hasattr(cv2, "CascadeClassifier"):
    HAAR_ROSTRO = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    HAAR_OJOS = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
elif DETECTOR_MTCNN is None:
    raise ImportError("Instale MTCNN (pip install mtcnn) u OpenCV 4.x para disponer de un detector de rostros.")
else:
    HAAR_ROSTRO = HAAR_OJOS = None
print("Detector principal:", DETECTOR_NOMBRE, "| respaldo Haar:", HAAR_ROSTRO is not None)


def detectar_rostro(rgb, confianza_min=0.90):
    """Devuelve (caja [x, y, w, h], ojo_izq, ojo_der) del rostro más grande, o None si no hay rostro.

    Los ojos se expresan como (x, y) en píxeles; ojo_izq es el que aparece a la izquierda de la imagen.
    """
    if DETECTOR_MTCNN is not None:
        caras = [c for c in DETECTOR_MTCNN.detect_faces(rgb) if c["confidence"] >= confianza_min]
        if caras:
            c = max(caras, key=lambda c: c["box"][2] * c["box"][3])
            ojos = sorted([c["keypoints"]["left_eye"], c["keypoints"]["right_eye"]])
            return np.array(c["box"], dtype=float), np.array(ojos[0], float), np.array(ojos[1], float)
    # Respaldo: Haar Cascades (rostro) + Haar de ojos para alinear
    if HAAR_ROSTRO is None:
        return None
    gris = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    caras = HAAR_ROSTRO.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(caras) == 0:
        return None
    x, y, w, h = max(caras, key=lambda c: c[2] * c[3])
    ojos = HAAR_OJOS.detectMultiScale(gris[y:y + h // 2, x:x + w], scaleFactor=1.1, minNeighbors=5)
    if len(ojos) >= 2:
        ojos = sorted(ojos, key=lambda o: -o[2] * o[3])[:2]
        centros = sorted((x + ex + ew / 2, y + ey + eh / 2) for ex, ey, ew, eh in ojos)
        ojo_izq, ojo_der = np.array(centros[0]), np.array(centros[1])
    else:   # sin ojos detectados: posición típica en un rostro frontal (sin rotación)
        ojo_izq = np.array([x + 0.3 * w, y + 0.4 * h])
        ojo_der = np.array([x + 0.7 * w, y + 0.4 * h])
    return np.array([x, y, w, h], dtype=float), ojo_izq, ojo_der


# %%
# ============================================================
# 2.2 Alineación, recorte, redimensionamiento y ecualización
# ============================================================
def ecualizar_iluminacion(rgb):
    """CLAHE sobre la luminancia (canal L de LAB): mejora el contraste local sin alterar los colores."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def alinear_y_recortar(rgb, caja, ojo_izq, ojo_der, tam=TAM_ROSTRO, margen=MARGEN_RECORTE):
    """Rota la imagen para nivelar los ojos y recorta un cuadrado centrado en el rostro."""
    dy, dx = ojo_der[1] - ojo_izq[1], ojo_der[0] - ojo_izq[0]
    angulo = np.degrees(np.arctan2(dy, dx))
    centro_ojos = tuple(((ojo_izq + ojo_der) / 2).tolist())
    M = cv2.getRotationMatrix2D(centro_ojos, angulo, 1.0)
    rotada = cv2.warpAffine(rgb, M, (rgb.shape[1], rgb.shape[0]), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
    x, y, w, h = caja
    cx, cy = M @ np.array([x + w / 2, y + h / 2, 1.0])        # centro de la caja tras la rotación
    lado = max(w, h) * (1 + margen)
    x1, y1 = int(round(cx - lado / 2)), int(round(cy - lado / 2))
    x2, y2 = int(round(cx + lado / 2)), int(round(cy + lado / 2))
    # Relleno por reflexión si el recorte se sale de la imagen
    pad = max(0, -x1, -y1, x2 - rotada.shape[1], y2 - rotada.shape[0])
    if pad:
        rotada = cv2.copyMakeBorder(rotada, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        x1, y1, x2, y2 = x1 + pad, y1 + pad, x2 + pad, y2 + pad
    recorte = rotada[y1:y2, x1:x2]
    return cv2.resize(recorte, (tam, tam), interpolation=cv2.INTER_AREA), angulo


def preprocesar_imagen(ruta_o_bgr, devolver_detalle=False):
    """Pipeline completo: lectura → RGB → detección → alineación → recorte 160×160 → CLAHE.

    Devuelve un arreglo uint8 (160, 160, 3) en RGB, o None si no se detecta un rostro.
    """
    bgr = cv2.imread(str(ruta_o_bgr)) if isinstance(ruta_o_bgr, (str, Path)) else ruta_o_bgr
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lado_max = max(rgb.shape[:2])
    if lado_max > 1024:                      # fotos de celular: se reducen para acelerar la detección
        escala = 1024 / lado_max
        rgb = cv2.resize(rgb, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    deteccion = detectar_rostro(rgb)
    if deteccion is None:
        return (None, None) if devolver_detalle else None
    caja, ojo_izq, ojo_der = deteccion
    rostro, angulo = alinear_y_recortar(rgb, caja, ojo_izq, ojo_der)
    rostro_sin_clahe = rostro.copy()
    if ECUALIZAR_ILUMINACION:
        rostro = ecualizar_iluminacion(rostro)
    if devolver_detalle:
        return rostro, {"rgb": rgb, "caja": caja, "ojos": (ojo_izq, ojo_der), "angulo": angulo,
                        "sin_clahe": rostro_sin_clahe}
    return rostro


# %%
# ============================================================
# 2.3 Procesamiento de todo el dataset (con caché en disco)
# ============================================================
t0 = time.time()
rostros, personas, origenes, rutas_procesadas = [], [], [], []
descartadas = []
for ruta, persona, origen in inventario:
    destino = DIR_PROCESADO / persona / (ruta.stem + ".png")
    if destino.exists():                                  # caché: no se vuelve a detectar
        rostro = cv2.cvtColor(cv2.imread(str(destino)), cv2.COLOR_BGR2RGB)
    else:
        rostro = preprocesar_imagen(ruta)
        if rostro is None:
            descartadas.append(str(ruta))
            continue
        destino.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destino), cv2.cvtColor(rostro, cv2.COLOR_RGB2BGR))
    rostros.append(rostro)
    personas.append(persona)
    origenes.append(origen)
    rutas_procesadas.append(destino)

X = np.stack(rostros).astype(np.uint8)                   # (N, 160, 160, 3) uint8 RGB
personas = np.array(personas)
origenes = np.array(origenes)
NOMBRES = sorted(set(personas))
y = np.array([NOMBRES.index(p) for p in personas])
print(f"Rostros procesados: {len(X)} | descartados (sin rostro detectado): {len(descartadas)} "
      f"| tiempo: {time.time() - t0:.1f} s")
print("Tensor del dataset:", X.shape, X.dtype, "| rango de píxeles:", X.min(), "-", X.max())
for r in descartadas[:10]:
    print("  sin rostro:", r)

# %% [markdown]
# ### 2.4 Visualización del antes y después del preprocesamiento
#
# Para cada ejemplo se muestra: la foto original con la caja del rostro y los ojos detectados, el rostro alineado y
# recortado a 160×160, el resultado tras la ecualización CLAHE y dos versiones con *Data Augmentation*.

# %%
# ============================================================
# 2.4 Antes / después del preprocesamiento + Data Augmentation
# ============================================================
aumentar = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.025, fill_mode="reflect"),          # ±0,025 × 360° ≈ ±9°
    layers.RandomBrightness(0.15, value_range=(0, 255)),
    layers.RandomContrast(0.15),
], name="data_augmentation")

ejemplos_vis = []
for persona in (PERSONAS_PROPIAS + PERSONAS_LFW)[:4]:
    ruta = next(r for r, p, _ in inventario if p == persona and str(r) not in descartadas)
    ejemplos_vis.append((persona, ruta))

fig, ejes = plt.subplots(len(ejemplos_vis), 5, figsize=(12.5, 2.7 * len(ejemplos_vis)))
titulos = ["Original + detección", "Alineado y recortado", "CLAHE (entrada final)", "Aumentado 1", "Aumentado 2"]
for fila, (persona, ruta) in enumerate(ejemplos_vis):
    rostro, det = preprocesar_imagen(ruta, devolver_detalle=True)
    original = det["rgb"].copy()
    x, y0, w, h = det["caja"].astype(int)
    grosor = max(2, original.shape[1] // 120)
    cv2.rectangle(original, (x, y0), (x + w, y0 + h), (42, 120, 214), grosor)
    for ojo in det["ojos"]:
        cv2.circle(original, tuple(ojo.astype(int)), grosor * 2, (235, 104, 52), -1)
    lote = tf.convert_to_tensor(np.repeat(rostro[None], 2, axis=0), tf.float32)
    aumentadas = np.clip(aumentar(lote, training=True).numpy(), 0, 255).astype(np.uint8)
    paneles = [original, det["sin_clahe"], rostro, aumentadas[0], aumentadas[1]]
    for col, panel in enumerate(paneles):
        ejes[fila, col].imshow(panel)
        ejes[fila, col].axis("off")
        if fila == 0:
            ejes[fila, col].set_title(titulos[col], fontsize=10)
    ejes[fila, 0].text(0, -0.04, f"{persona.replace('_', ' ')} · giro corregido {det['angulo']:+.1f}°",
                       transform=ejes[fila, 0].transAxes, fontsize=8, va="top")
fig.suptitle(f"Pipeline de preprocesamiento facial (detector: {DETECTOR_NOMBRE})", fontweight="bold")
plt.tight_layout()
plt.savefig(DIR_FIG / "02_preprocesamiento_antes_despues.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Particiones del dataset y organización de tripletas
#
# La partición se hace **por identidad** para poder evaluar el aprendizaje *one-shot* de verdad:
#
# * **Identidades LFW:** 75 % se usan para entrenar y el 25 % restante (**identidades nunca vistas**) forma el
#   conjunto de prueba. Reconocerlas con una sola foto de referencia es exactamente el problema *one-shot*.
# * **Personas propias:** la mitad de sus fotos entra al entrenamiento (para que la red las vea en tripletas) y la
#   otra mitad se reserva para la prueba (fotos de soporte y de consulta).
# * Del conjunto de entrenamiento se separa un 20 % de imágenes como **validación**: sirve para vigilar la pérdida
#   y para elegir el **umbral de decisión** sin tocar la prueba.

# %%
# ============================================================
# 3.1 Particiones por identidad
# ============================================================
particion = np.empty(len(X), dtype=object)
ids_lfw = RNG.permutation(PERSONAS_LFW)
n_prueba_lfw = int(round(FRACCION_IDS_PRUEBA_LFW * len(ids_lfw)))
IDS_PRUEBA_LFW = {str(p) for p in ids_lfw[:n_prueba_lfw]}

for persona in NOMBRES:
    idx = RNG.permutation(np.flatnonzero(personas == persona))
    if persona in IDS_PRUEBA_LFW:
        particion[idx] = "prueba"
        continue
    if persona in PERSONAS_PROPIAS:
        n_prueba = int(round(FRACCION_PRUEBA_PROPIAS * len(idx)))
        particion[idx[:n_prueba]] = "prueba"
        idx = idx[n_prueba:]
    n_val = max(1, int(round(FRACCION_VALIDACION * len(idx)))) if len(idx) >= 4 else 0
    particion[idx[:n_val]] = "validacion"
    particion[idx[n_val:]] = "entrenamiento"

idx_ent = np.flatnonzero(particion == "entrenamiento")
idx_val = np.flatnonzero(particion == "validacion")
idx_pru = np.flatnonzero(particion == "prueba")

with open(DIR_DATASET / "manifiesto_particiones.csv", "w", newline="", encoding="utf-8") as f:
    escritor = csv.writer(f)
    escritor.writerow(["ruta_procesada", "persona", "origen", "particion"])
    for i in range(len(X)):
        escritor.writerow([rutas_procesadas[i].relative_to(DIR_DATASET).as_posix(), personas[i], origenes[i],
                           particion[i]])

resumen_particiones = {}
for nombre, idx in (("entrenamiento", idx_ent), ("validacion", idx_val), ("prueba", idx_pru)):
    resumen_particiones[nombre] = {"imagenes": int(len(idx)), "identidades": int(len(set(personas[idx])))}
    print(f"{nombre:>13}: {len(idx):4d} imágenes de {len(set(personas[idx])):2d} identidades")
print("Identidades LFW nunca vistas en entrenamiento:", sorted(IDS_PRUEBA_LFW))

# %% [markdown]
# ### 3.2 Generación de tripletas (A, P, N) y minería *semi-hard*
#
# Una tripleta está formada por una imagen **Ancla** (A), otra imagen **Positiva** (P) de la *misma* persona y una
# imagen **Negativa** (N) de una persona *distinta*. Elegir negativos al azar produce muchas tripletas "fáciles"
# (pérdida cero) que no aportan gradiente; por eso, desde la época 3 se usan negativos **semi-hard**
# (Schroff et al., 2015): negativos más lejanos que el positivo pero dentro del margen,
# $d(A,P) < d(A,N) < d(A,P) + \alpha$, calculados con los *embeddings* del modelo al inicio de cada época.

# %%
# ============================================================
# 3.2 Generadores de tripletas
# ============================================================
def indices_por_persona(indices):
    grupos = {}
    for i in indices:
        grupos.setdefault(y[i], []).append(i)
    return {k: np.array(v) for k, v in grupos.items() if len(v) >= 2}


def generar_tripletas(indices, n, embeddings=None, alpha=ALPHA, rng=RNG):
    """Devuelve un arreglo (n, 3) de índices [ancla, positiva, negativa].

    Sin `embeddings`: negativos aleatorios. Con `embeddings` (alineados con `indices`): minería semi-hard.
    """
    grupos = indices_por_persona(indices)
    clases = list(grupos)
    posicion = {g: k for k, g in enumerate(indices)}
    etiquetas = y[indices]
    tripletas = []
    for _ in range(n):
        c = clases[rng.integers(len(clases))]
        a, p = rng.choice(grupos[c], size=2, replace=False)
        candidatos = indices[etiquetas != c]
        if embeddings is None:
            neg = rng.choice(candidatos)
        else:
            ea, ep = embeddings[posicion[a]], embeddings[posicion[p]]
            d_ap = np.sum((ea - ep) ** 2)
            d_an = np.sum((embeddings[[posicion[k] for k in candidatos]] - ea) ** 2, axis=1)
            semi_hard = candidatos[(d_an > d_ap) & (d_an < d_ap + alpha)]
            if len(semi_hard):
                neg = rng.choice(semi_hard)
            else:   # si no hay semi-hard: el negativo más cercano que siga más lejos que el positivo
                mas_lejos = d_an > d_ap
                neg = candidatos[mas_lejos][np.argmin(d_an[mas_lejos])] if mas_lejos.any() \
                    else candidatos[np.argmax(d_an)]
        tripletas.append((a, p, neg))
    return np.array(tripletas)


# Tripletas fijas de validación (para que la pérdida de validación sea comparable entre épocas)
TRIPLETAS_VAL = generar_tripletas(np.concatenate([idx_val, idx_ent]), 1600, rng=np.random.default_rng(SEED))
TRIPLETAS_VAL = TRIPLETAS_VAL[np.isin(TRIPLETAS_VAL[:, 0], idx_val)] if len(idx_val) else TRIPLETAS_VAL

# Organización en disco de una muestra de tripletas: tripletas_muestra/tripleta_XX/{ancla,positiva,negativa}.png
muestra = generar_tripletas(idx_ent, 10, rng=np.random.default_rng(SEED))
shutil.rmtree(DIR_TRIPLETAS, ignore_errors=True)
for k, (a, p, n) in enumerate(muestra, 1):
    carpeta = DIR_TRIPLETAS / f"tripleta_{k:02d}"
    carpeta.mkdir(parents=True, exist_ok=True)
    for rol, i in (("ancla", a), ("positiva", p), ("negativa", n)):
        cv2.imwrite(str(carpeta / f"{rol}__{personas[i]}.png"), cv2.cvtColor(X[i], cv2.COLOR_RGB2BGR))

fig, ejes = plt.subplots(3, 5, figsize=(10, 6.6))
for col, (a, p, n) in enumerate(muestra[:5]):
    for fila, (rol, i) in enumerate((("Ancla (A)", a), ("Positiva (P)", p), ("Negativa (N)", n))):
        ejes[fila, col].imshow(X[i])
        ejes[fila, col].set_title(f"{rol}\n{personas[i].replace('_', ' ')}", fontsize=8,
                                  color=COLOR_INTER if fila == 2 else "black")
        ejes[fila, col].axis("off")
fig.suptitle("Ejemplos de tripletas de entrenamiento", fontweight="bold")
plt.tight_layout()
plt.savefig(DIR_FIG / "03_tripletas_ejemplo.png", bbox_inches="tight")
plt.show()
print(f"Tripletas de validación fijas: {len(TRIPLETAS_VAL)} | muestra guardada en {DIR_TRIPLETAS}")

# %% [markdown]
# ## 4. Arquitectura: red de *embeddings* y modelo siamés
#
# **Red generadora de características** $f(x)$ (*Embedding Network*):
#
# | Capa | Salida | Detalle |
# |---|---|---|
# | Entrada | (160, 160, 3) | Rostro RGB en [0, 255] |
# | `Rescaling` | (160, 160, 3) | $x/127{,}5 - 1$ → [-1, 1] |
# | MobileNetV2 (ImageNet, sin la capa de clasificación) | (5, 5, 1280) | Convoluciones *depthwise separable* con bloques residuales invertidos |
# | `GlobalAveragePooling2D` | (1280) | Resume el mapa de características |
# | `Dropout(0,3)` | (1280) | Regularización |
# | `Dense(256, ReLU)` + `BatchNormalization` | (256) | Proyección no lineal |
# | `Dense(128)` lineal | (128) | Embedding sin normalizar |
# | `UnitNormalization` (L2) | (128) | $\lVert f(x)\rVert_2 = 1$: el embedding vive en la hiperesfera unitaria |
#
# El **modelo siamés** tiene tres entradas (A, P, N) y aplica **la misma instancia** de $f$ a cada una: las tres
# ramas comparten exactamente los mismos pesos, de modo que imágenes de la misma persona caen cerca en el espacio
# de 128 dimensiones sin importar por qué rama entren.

# %%
# ============================================================
# 4.1 Red de embeddings (transfer learning con MobileNetV2) y modelo siamés
# ============================================================
FORMA_ENTRADA = (TAM_ROSTRO, TAM_ROSTRO, 3)


def construir_red_embedding(dim=DIM_EMBEDDING):
    backbone = keras.applications.MobileNetV2(input_shape=FORMA_ENTRADA, include_top=False, weights="imagenet")
    backbone.trainable = False                                        # fase 1: extractor congelado
    entrada = keras.Input(shape=FORMA_ENTRADA, name="rostro")
    x = layers.Rescaling(1 / 127.5, offset=-1, name="normalizacion")(entrada)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="pooling_global")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    x = layers.Dense(256, activation="relu", name="densa_256")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dense(dim, name=f"embedding_{dim}d")(x)
    salida = layers.UnitNormalization(axis=1, name="normalizacion_l2")(x)
    return keras.Model(entrada, salida, name="red_embedding"), backbone


red_embedding, backbone = construir_red_embedding()

entrada_a = keras.Input(shape=FORMA_ENTRADA, name="ancla")
entrada_p = keras.Input(shape=FORMA_ENTRADA, name="positiva")
entrada_n = keras.Input(shape=FORMA_ENTRADA, name="negativa")
modelo_siames = keras.Model(
    inputs=[entrada_a, entrada_p, entrada_n],
    outputs=[red_embedding(entrada_a), red_embedding(entrada_p), red_embedding(entrada_n)],
    name="red_siamesa_triplet")

red_embedding.summary(line_length=100)
modelo_siames.summary(line_length=100)
# Prueba de que las tres ramas comparten pesos: el modelo siamés no añade ni un parámetro propio
assert len(modelo_siames.weights) == len(red_embedding.weights)
assert modelo_siames.count_params() == red_embedding.count_params()
print("Pesos compartidos por las 3 ramas:", f"{red_embedding.count_params():,}".replace(",", " "), "parámetros")

# %% [markdown]
# ## 5. Triplet Loss
#
# Para una tripleta $(A, P, N)$ y la red de *embeddings* $f$:
#
# $$L(A,P,N)=\max\left(\lVert f(A)-f(P)\rVert_2^2-\lVert f(A)-f(N)\rVert_2^2+\alpha,\;0\right)$$
#
# * $\lVert f(A)-f(P)\rVert^2$: distancia ancla–positiva (**debe ser pequeña**).
# * $\lVert f(A)-f(N)\rVert^2$: distancia ancla–negativa (**debe ser grande**).
# * $\alpha = 0{,}2$: margen mínimo que debe separar ambas distancias. Sin él, la red podría hacer $f(x)=0$ para
#   todo $x$ y obtener pérdida cero.
#
# La pérdida del lote es el promedio sobre sus tripletas. Como los *embeddings* tienen norma 1, las distancias al
# cuadrado están en $[0, 4]$ y $\lVert u-v\rVert^2 = 2 - 2\cos(u, v)$.

# %%
# ============================================================
# 5.1 Implementación explícita de la Triplet Loss
# ============================================================
def triplet_loss(emb_a, emb_p, emb_n, alpha=ALPHA):
    """L(A,P,N) = max(||f(A)-f(P)||² - ||f(A)-f(N)||² + α, 0), promediada sobre el lote.

    Devuelve (pérdida media, d(A,P), d(A,N)) para poder monitorear las distancias.
    """
    d_ap = tf.reduce_sum(tf.square(emb_a - emb_p), axis=1)
    d_an = tf.reduce_sum(tf.square(emb_a - emb_n), axis=1)
    perdida = tf.maximum(d_ap - d_an + alpha, 0.0)
    return tf.reduce_mean(perdida), d_ap, d_an


# Verificación numérica con vectores 2D hechos a mano
a = tf.constant([[1.0, 0.0]])
casos = {"fácil (N lejos)": ([[0.8, 0.6]], [[-1.0, 0.0]]),
         "semi-hard (N dentro del margen)": ([[0.8, 0.6]], [[0.75, 0.6614]]),
         "difícil (N más cerca que P)": ([[0.6, 0.8]], [[0.8, 0.6]])}
for nombre, (p, n) in casos.items():
    L, dap, dan = triplet_loss(a, tf.constant(p), tf.constant(n))
    print(f"{nombre:>32}: d(A,P)={dap.numpy()[0]:.2f}  d(A,N)={dan.numpy()[0]:.2f}  "
          f"L=max({dap.numpy()[0]:.2f} - {dan.numpy()[0]:.2f} + {ALPHA}, 0) = {L.numpy():.2f}")

# %% [markdown]
# ### 5.2 Bucle de entrenamiento
#
# Se usa un bucle propio con `tf.GradientTape` para que cada paso sea explícito: *Data Augmentation* → *embeddings*
# de A, P y N con los pesos compartidos → Triplet Loss → gradientes → actualización con **Adam**.
#
# * **Fase 1** (12 épocas, $\eta = 10^{-3}$): MobileNetV2 congelada, se entrena solo la cabeza de *embedding*.
# * **Fase 2** (10 épocas, $\eta = 5\cdot10^{-5}$): ajuste fino de las últimas 30 capas de MobileNetV2 (las capas de
#   Batch Normalization siguen congeladas para no dañar sus estadísticas con lotes pequeños).
#
# Además de la pérdida se registra la **fracción de tripletas activas** ($L>0$) y la **exactitud de tripletas**
# ($d(A,P) < d(A,N)$) en validación. Se guardan los pesos de la época con menor pérdida de validación.

# %%
# ============================================================
# 5.2 Entrenamiento con Triplet Loss
# ============================================================
def calcular_embeddings(imagenes, lote=64):
    return red_embedding.predict(imagenes.astype(np.float32), batch_size=lote, verbose=0)


def crear_paso_entrenamiento(optimizador):
    @tf.function
    def paso(a, p, n):
        with tf.GradientTape() as cinta:
            ea, ep, en = modelo_siames([aumentar(a, training=True), aumentar(p, training=True),
                                        aumentar(n, training=True)], training=True)
            perdida, d_ap, d_an = triplet_loss(ea, ep, en)
        variables = red_embedding.trainable_variables
        gradientes = cinta.gradient(perdida, variables)
        optimizador.apply_gradients(zip(gradientes, variables))
        activas = tf.reduce_mean(tf.cast(d_ap - d_an + ALPHA > 0, tf.float32))
        return perdida, activas
    return paso


def evaluar_tripletas(tripletas):
    emb = {i: e for i, e in zip(np.unique(tripletas), calcular_embeddings(X[np.unique(tripletas)]))}
    ea, ep, en = (np.stack([emb[i] for i in tripletas[:, k]]) for k in range(3))
    perdida, d_ap, d_an = triplet_loss(ea, ep, en)
    return float(perdida), float(np.mean(d_ap.numpy() < d_an.numpy()))


historial = {"perdida": [], "perdida_val": [], "activas": [], "exactitud_val": [], "fase": [], "tiempo": []}
mejor = {"perdida_val": np.inf, "pesos": None, "epoca": 0}
perdida_val_inicial, exactitud_val_inicial = evaluar_tripletas(TRIPLETAS_VAL)
print(f"Antes de entrenar → pérdida val {perdida_val_inicial:.4f} | exactitud de tripletas val "
      f"{exactitud_val_inicial:.3f}")


def entrenar_fase(n_epocas, lr, fase, epoca_inicial):
    paso = crear_paso_entrenamiento(keras.optimizers.Adam(learning_rate=lr))
    for epoca in range(epoca_inicial, epoca_inicial + n_epocas):
        t0 = time.time()
        usar_mineria = epoca >= MINERIA_DESDE_EPOCA
        emb_ent = calcular_embeddings(X[idx_ent]) if usar_mineria else None
        tripletas = generar_tripletas(idx_ent, TRIPLETAS_POR_EPOCA, embeddings=emb_ent)
        perdidas, activas = [], []
        for inicio in range(0, len(tripletas), TAM_LOTE):
            lote = tripletas[inicio:inicio + TAM_LOTE]
            a, p, n = (tf.convert_to_tensor(X[lote[:, k]], tf.float32) for k in range(3))
            perdida, frac = paso(a, p, n)
            perdidas.append(float(perdida))
            activas.append(float(frac))
        perdida_val, exactitud_val = evaluar_tripletas(TRIPLETAS_VAL)
        for clave, valor in (("perdida", np.mean(perdidas)), ("perdida_val", perdida_val),
                             ("activas", np.mean(activas)), ("exactitud_val", exactitud_val),
                             ("fase", fase), ("tiempo", time.time() - t0)):
            historial[clave].append(float(valor) if clave != "fase" else valor)
        if perdida_val < mejor["perdida_val"]:
            mejor.update(perdida_val=perdida_val, pesos=red_embedding.get_weights(), epoca=epoca)
        print(f"Época {epoca:2d} [fase {fase}{', semi-hard' if usar_mineria else ', aleatorias'}] "
              f"pérdida {np.mean(perdidas):.4f} | activas {np.mean(activas):.2f} | "
              f"pérdida val {perdida_val:.4f} | exactitud val {exactitud_val:.3f} | {time.time() - t0:.0f} s")
    return epoca_inicial + n_epocas


# Embeddings ANTES de entrenar (cabeza aleatoria sobre MobileNetV2-ImageNet): línea base para comparar
EMB_PRUEBA_INICIAL = calcular_embeddings(X[idx_pru])
EMB_VAL_INICIAL = calcular_embeddings(X[idx_val])

t_inicio = time.time()
siguiente = entrenar_fase(EPOCAS_FASE1, LR_FASE1, fase=1, epoca_inicial=1)

# Fase 2: descongelar las últimas capas del backbone (excepto BatchNormalization)
backbone.trainable = True
for capa in backbone.layers[:-CAPAS_AJUSTE_FINO]:
    capa.trainable = False
for capa in backbone.layers:
    if isinstance(capa, layers.BatchNormalization):
        capa.trainable = False
print("Parámetros entrenables en fase 2:",
      f"{sum(int(np.prod(v.shape)) for v in red_embedding.trainable_variables):,}".replace(",", " "))
entrenar_fase(EPOCAS_FASE2, LR_FASE2, fase=2, epoca_inicial=siguiente)

red_embedding.set_weights(mejor["pesos"])
TIEMPO_ENTRENAMIENTO = time.time() - t_inicio
print(f"\nEntrenamiento: {TIEMPO_ENTRENAMIENTO / 60:.1f} min | mejores pesos: época {mejor['epoca']} "
      f"(pérdida val {mejor['perdida_val']:.4f})")
red_embedding.save(DIR_MODELOS / "red_embedding.keras")

# %%
# ============================================================
# 5.3 Curva de convergencia
# ============================================================
epocas = np.arange(1, len(historial["perdida"]) + 1)
fig, (e1, e2) = plt.subplots(1, 2, figsize=(12, 4.2))
e1.plot(epocas, historial["perdida"], color=COLOR_INTRA, marker="o", ms=4, label="Entrenamiento")
e1.plot(epocas, historial["perdida_val"], color=COLOR_INTER, marker="s", ms=4, label="Validación")
e1.axhline(ALPHA, color="#8a8984", ls=":", lw=1.2)
e1.text(epocas[-1], ALPHA, " α (pérdida con f(x) constante)", va="bottom", ha="right", fontsize=8, color="#555")
e2.plot(epocas, historial["activas"], color=COLOR_INTRA, marker="o", ms=4, label="Tripletas activas (L > 0), entren.")
e2.plot(epocas, historial["exactitud_val"], color=COLOR_INTER, marker="s", ms=4,
        label="Exactitud de tripletas, validación")
for eje in (e1, e2):
    eje.axvspan(EPOCAS_FASE1 + 0.5, epocas[-1] + 0.5, color="#f1efe8", zorder=0)
    eje.axvline(mejor["epoca"], color="#555", ls="--", lw=1)
    eje.set_xlabel("Época")
    eje.legend(fontsize=8)
e1.text(EPOCAS_FASE1 + 0.7, e1.get_ylim()[0], "Fase 2: ajuste fino", fontsize=8, va="bottom", color="#555")
e1.set(title="Triplet Loss por época", ylabel="Pérdida media")
e2.set(title="Tripletas activas y exactitud de tripletas", ylabel="Proporción", ylim=(0, 1.02))
plt.tight_layout()
plt.savefig(DIR_FIG / "04_curva_convergencia.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Evaluación del reconocimiento facial
#
# ### 6.1 Distancias intra-clase vs. inter-clase y umbral de decisión
#
# Se forman todos los **pares genuinos** (misma persona) y el mismo número de **pares impostores** (personas
# distintas). El **umbral óptimo** $\tau$ se elige en **validación** maximizando la exactitud de verificación y
# luego se aplica sin cambios a la **prueba**, que contiene identidades nunca vistas. Se compara el modelo antes
# y después del entrenamiento con Triplet Loss.

# %%
# ============================================================
# 6.1 Pares de verificación, umbral óptimo, histogramas y curva ROC
# ============================================================
def formar_pares(indices, rng):
    """Todos los pares genuinos + el mismo número de pares impostores aleatorios. Devuelve (i, j, misma)."""
    genuinos = [(i, j) for k, i in enumerate(indices) for j in indices[k + 1:] if y[i] == y[j]]
    impostores = set()
    while len(impostores) < len(genuinos):
        i, j = rng.choice(indices, size=2, replace=False)
        if y[i] != y[j]:
            impostores.add((min(i, j), max(i, j)))
    pares = np.array(genuinos + sorted(impostores))
    etiquetas = np.r_[np.ones(len(genuinos)), np.zeros(len(impostores))].astype(int)
    return pares, etiquetas


def distancias_pares(emb, indices, pares):
    pos = {g: k for k, g in enumerate(indices)}
    e1 = emb[[pos[i] for i in pares[:, 0]]]
    e2 = emb[[pos[j] for j in pares[:, 1]]]
    return np.sum((e1 - e2) ** 2, axis=1)                  # distancia euclidiana al cuadrado


def umbral_optimo(dist, etiquetas):
    candidatos = np.unique(dist)
    exactitudes = [np.mean((dist <= t) == etiquetas) for t in candidatos]
    return float(candidatos[int(np.argmax(exactitudes))])


def metricas_verificacion(dist, etiquetas, umbral):
    pred = dist <= umbral
    vp, fp = np.sum(pred & (etiquetas == 1)), np.sum(pred & (etiquetas == 0))
    fn, vn = np.sum(~pred & (etiquetas == 1)), np.sum(~pred & (etiquetas == 0))
    return {"exactitud": float((vp + vn) / len(etiquetas)), "tasa_verdaderos_positivos": float(vp / (vp + fn)),
            "tasa_falsos_positivos": float(fp / (fp + vn)), "precision": float(vp / max(vp + fp, 1)),
            "auc": float(roc_auc_score(etiquetas, -dist)),
            "distancia_intra_media": float(dist[etiquetas == 1].mean()),
            "distancia_inter_media": float(dist[etiquetas == 0].mean()),
            "distancia_intra_std": float(dist[etiquetas == 1].std()),
            "distancia_inter_std": float(dist[etiquetas == 0].std()),
            "vp": int(vp), "fp": int(fp), "fn": int(fn), "vn": int(vn)}


rng_pares = np.random.default_rng(SEED)
pares_val, etq_val = formar_pares(idx_val, rng_pares)
pares_pru, etq_pru = formar_pares(idx_pru, rng_pares)
EMB_VAL = calcular_embeddings(X[idx_val])
EMB_PRUEBA = calcular_embeddings(X[idx_pru])

resultados_verificacion = {}
for etapa, emb_v, emb_p in (("antes", EMB_VAL_INICIAL, EMB_PRUEBA_INICIAL), ("despues", EMB_VAL, EMB_PRUEBA)):
    d_val = distancias_pares(emb_v, idx_val, pares_val)
    d_pru = distancias_pares(emb_p, idx_pru, pares_pru)
    tau = umbral_optimo(d_val, etq_val)
    resultados_verificacion[etapa] = {"umbral": tau, "validacion": metricas_verificacion(d_val, etq_val, tau),
                                      "prueba": metricas_verificacion(d_pru, etq_pru, tau),
                                      "_d_pru": d_pru}
    m = resultados_verificacion[etapa]["prueba"]
    print(f"[{etapa:>7}] umbral τ = {tau:.3f} | prueba: exactitud {m['exactitud']:.3f}, AUC {m['auc']:.3f}, "
          f"TVP {m['tasa_verdaderos_positivos']:.3f}, TFP {m['tasa_falsos_positivos']:.3f} | "
          f"d intra {m['distancia_intra_media']:.3f} vs inter {m['distancia_inter_media']:.3f}")
UMBRAL = resultados_verificacion["despues"]["umbral"]
print(f"Pares de validación: {len(etq_val)} | pares de prueba: {len(etq_pru)} "
      f"({int(etq_pru.sum())} genuinos, {int((1 - etq_pru).sum())} impostores)")

fig, ejes = plt.subplots(1, 3, figsize=(15, 4.2))
bins = np.linspace(0, max(r["_d_pru"].max() for r in resultados_verificacion.values()), 40)
for eje, etapa, titulo in ((ejes[0], "antes", "Antes de entrenar"), (ejes[1], "despues", "Después de Triplet Loss")):
    r = resultados_verificacion[etapa]
    eje.hist(r["_d_pru"][etq_pru == 1], bins=bins, alpha=0.75, color=COLOR_INTRA, label="Misma persona (intra)")
    eje.hist(r["_d_pru"][etq_pru == 0], bins=bins, alpha=0.75, color=COLOR_INTER, label="Personas distintas (inter)")
    eje.axvline(r["umbral"], color="#222", ls="--", lw=1.5, label=f"Umbral τ = {r['umbral']:.2f}")
    eje.set(title=f"{titulo} (prueba)", xlabel="Distancia euclidiana² entre embeddings", ylabel="Número de pares")
    eje.legend(fontsize=8)
for etapa, estilo, nombre in (("antes", ":", "Antes"), ("despues", "-", "Después")):
    fpr, tpr, _ = roc_curve(etq_pru, -resultados_verificacion[etapa]["_d_pru"])
    ejes[2].plot(fpr, tpr, ls=estilo, color=COLOR_INTRA,
                 label=f"{nombre} (AUC = {resultados_verificacion[etapa]['prueba']['auc']:.3f})")
ejes[2].plot([0, 1], [0, 1], color="#8a8984", lw=1)
ejes[2].set(title="Curva ROC de verificación (prueba)", xlabel="Tasa de falsos positivos",
            ylabel="Tasa de verdaderos positivos")
ejes[2].legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(DIR_FIG / "05_distancias_intra_inter_roc.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 6.2 Función de verificación de identidad
#
# `verificar_identidad` recibe **dos fotos crudas** (rutas o arreglos), aplica el mismo preprocesamiento del
# entrenamiento, calcula los *embeddings* y compara la distancia euclidiana al cuadrado contra el umbral $\tau$.
# También informa la **similitud coseno**, que para vectores de norma 1 cumple $d^2 = 2 - 2\cos$.

# %%
# ============================================================
# 6.2 Verificación de identidad entre dos imágenes
# ============================================================
def verificar_identidad(imagen_1, imagen_2, umbral=UMBRAL, ya_preprocesadas=False):
    """Compara dos rostros. Acepta rutas, imágenes BGR crudas o rostros RGB 160×160 ya preprocesados."""
    rostros_par = [img if ya_preprocesadas else preprocesar_imagen(img) for img in (imagen_1, imagen_2)]
    if any(r is None for r in rostros_par):
        return {"veredicto": "No se detectó un rostro en alguna de las imágenes"}
    e1, e2 = calcular_embeddings(np.stack(rostros_par))
    distancia = float(np.sum((e1 - e2) ** 2))
    coseno = float(np.dot(e1, e2))
    misma = distancia <= umbral
    return {"distancia": distancia, "similitud_coseno": coseno, "umbral": umbral, "misma_identidad": misma,
            "veredicto": "Misma Identidad - Verificado" if misma else "Identidad Diferente - Rechazado"}


def mostrar_pares(pares_mostrar, titulo, archivo):
    fig, ejes = plt.subplots(2, len(pares_mostrar), figsize=(2.6 * len(pares_mostrar), 6.9))
    for col, (i, j) in enumerate(pares_mostrar):
        r = verificar_identidad(X[i], X[j], ya_preprocesadas=True)
        correcto = r["misma_identidad"] == (y[i] == y[j])
        for fila, k in enumerate((i, j)):
            ejes[fila, col].imshow(X[k])
            ejes[fila, col].set_xticks([]); ejes[fila, col].set_yticks([]); ejes[fila, col].grid(False)
            ejes[fila, col].set_title(personas[k].replace("_", " "), fontsize=8)
        color = COLOR_OK if r["misma_identidad"] else COLOR_NO
        ejes[1, col].set_xlabel(f"d² = {r['distancia']:.3f}  (τ = {UMBRAL:.3f})\ncos = {r['similitud_coseno']:.3f}\n"
                                f"{r['veredicto'].replace(' - ', chr(10))}\n"
                                f"{'✔ acierto' if correcto else '✘ error'}",
                                color=color, fontsize=8.5, fontweight="bold")
        for s in ejes[1, col].spines.values():
            s.set_edgecolor(color); s.set_linewidth(3)
        for s in ejes[0, col].spines.values():
            s.set_edgecolor(color); s.set_linewidth(3)
    fig.suptitle(titulo, fontweight="bold")
    plt.tight_layout()
    plt.savefig(DIR_FIG / archivo, bbox_inches="tight")
    plt.show()


# Pares de demostración: 4 genuinos y 4 impostores (se priorizan las personas propias si existen)
rng_demo = np.random.default_rng(7)
prioridad = [i for i in idx_pru if origenes[i] == "propia"] or list(idx_pru)
genuinos_demo, impostores_demo = [], []
for i in rng_demo.permutation(prioridad):
    mismos = [j for j in idx_pru if y[j] == y[i] and j != i]
    if mismos and len(genuinos_demo) < 4 and y[i] not in [y[a] for a, _ in genuinos_demo]:
        genuinos_demo.append((i, rng_demo.choice(mismos)))
    otros = [j for j in idx_pru if y[j] != y[i]]
    if len(impostores_demo) < 4 and y[i] not in [y[a] for a, _ in impostores_demo]:
        impostores_demo.append((i, rng_demo.choice(otros)))
mostrar_pares(genuinos_demo + impostores_demo,
              "Verificación facial sobre pares de prueba (identidades no vistas en el entrenamiento)",
              "06_verificacion_pares.png")

# Ejemplo de uso directo con dos archivos de imagen crudos
i, j = genuinos_demo[0]
ruta_1 = next(r for r, p, _ in inventario if Path(r).stem == rutas_procesadas[i].stem)
ruta_2 = next(r for r, p, _ in inventario if Path(r).stem == rutas_procesadas[j].stem)
print("verificar_identidad(", ruta_1.name, ",", ruta_2.name, ") →", verificar_identidad(ruta_1, ruta_2))

# Tripletas de prueba con sus distancias
tripletas_prueba = generar_tripletas(idx_pru, 4, rng=np.random.default_rng(3))
emb_t = calcular_embeddings(X[tripletas_prueba.ravel()]).reshape(len(tripletas_prueba), 3, -1)
fig, ejes = plt.subplots(len(tripletas_prueba), 3, figsize=(6.6, 2.5 * len(tripletas_prueba)))
for fila, ((a, p, n), (ea, ep, en)) in enumerate(zip(tripletas_prueba, emb_t)):
    d_ap, d_an = np.sum((ea - ep) ** 2), np.sum((ea - en) ** 2)
    for col, (k, rol) in enumerate(((a, "Ancla"), (p, f"Positiva · d²={d_ap:.3f}"), (n, f"Negativa · d²={d_an:.3f}"))):
        ejes[fila, col].imshow(X[k]); ejes[fila, col].axis("off")
        color = "black" if col == 0 else (COLOR_OK if (d_ap if col == 1 else d_an) <= UMBRAL else COLOR_NO)
        ejes[fila, col].set_title(f"{rol}\n{personas[k].replace('_', ' ')}", fontsize=8, color=color)
fig.suptitle(f"Tripletas de prueba (verde: d² ≤ τ = {UMBRAL:.3f})", fontweight="bold")
plt.tight_layout()
plt.savefig(DIR_FIG / "07_tripletas_prueba_distancias.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 6.3 One-shot y Few-shot Learning
#
# Se simula el escenario real de un sistema de acceso: la **galería** contiene, para cada persona de prueba, solo
# **k** imágenes de referencia (soporte) y el resto de sus fotos son **consultas**.
#
# * **One-shot (k = 1):** la referencia de cada persona es el *embedding* de una única foto.
# * **Few-shot (k = 3, k = 5):** la referencia es el **prototipo** de la persona: la media de sus k *embeddings*
#   de soporte, renormalizada a norma 1 (como en las *Prototypical Networks*).
#
# Para cada consulta se mide: **identificación** (¿el prototipo más cercano es el de la persona correcta?) y
# **verificación** (¿se acepta contra su propio prototipo y se rechaza contra el de otra persona usando τ?). Se
# promedian 30 episodios con soportes elegidos al azar. Ninguna imagen de prueba se usó para entrenar.

# %%
# ============================================================
# 6.3 Evaluación One-shot (k=1) y Few-shot (k=3, k=5)
# ============================================================
POS_PRUEBA = {g: k for k, g in enumerate(idx_pru)}


def prototipo(embs):
    m = embs.mean(axis=0)
    return m / np.linalg.norm(m)


def episodio(k, rng):
    """Devuelve exactitudes (identificación, verificación) de un episodio con k imágenes de soporte por persona."""
    soportes, consultas = {}, []
    for c in np.unique(y[idx_pru]):
        idx_c = rng.permutation(idx_pru[y[idx_pru] == c])
        if len(idx_c) < k + 1:
            continue
        soportes[c] = prototipo(EMB_PRUEBA[[POS_PRUEBA[i] for i in idx_c[:k]]])
        consultas += [(i, c) for i in idx_c[k:]]
    clases = np.array(list(soportes))
    P = np.stack([soportes[c] for c in clases])
    aciertos_id, aciertos_ver = [], []
    for i, c in consultas:
        d = np.sum((P - EMB_PRUEBA[POS_PRUEBA[i]]) ** 2, axis=1)
        aciertos_id.append(clases[np.argmin(d)] == c)
        otra = rng.choice(np.flatnonzero(clases != c))
        aciertos_ver.append(d[clases == c][0] <= UMBRAL)          # genuino: debe aceptarse
        aciertos_ver.append(d[otra] > UMBRAL)                     # impostor: debe rechazarse
    return np.mean(aciertos_id), np.mean(aciertos_ver), len(clases), len(consultas)


resultados_shot = {}
for k in K_FEW_SHOT:
    rng_ep = np.random.default_rng(SEED + k)
    res = np.array([episodio(k, rng_ep) for _ in range(EPISODIOS)])
    resultados_shot[k] = {"identificacion_media": float(res[:, 0].mean()), "identificacion_std": float(res[:, 0].std()),
                          "verificacion_media": float(res[:, 1].mean()), "verificacion_std": float(res[:, 1].std()),
                          "identidades": int(res[0, 2]), "consultas_por_episodio": int(res[0, 3]),
                          "azar": float(1 / res[0, 2])}
    nombre = "One-shot" if k == 1 else "Few-shot"
    r = resultados_shot[k]
    print(f"{nombre} (k={k}): identificación {r['identificacion_media']:.3f} ± {r['identificacion_std']:.3f} "
          f"(azar {r['azar']:.3f}, {r['identidades']} identidades) | verificación "
          f"{r['verificacion_media']:.3f} ± {r['verificacion_std']:.3f}")

fig, eje = plt.subplots(figsize=(7, 4))
xs = np.arange(len(K_FEW_SHOT))
ident = [resultados_shot[k]["identificacion_media"] for k in K_FEW_SHOT]
verif = [resultados_shot[k]["verificacion_media"] for k in K_FEW_SHOT]
b1 = eje.bar(xs - 0.18, ident, 0.36, yerr=[resultados_shot[k]["identificacion_std"] for k in K_FEW_SHOT],
             color=COLOR_INTRA, label="Identificación (rank-1)", capsize=3)
b2 = eje.bar(xs + 0.18, verif, 0.36, yerr=[resultados_shot[k]["verificacion_std"] for k in K_FEW_SHOT],
             color=COLOR_INTER, label="Verificación con umbral τ", capsize=3)
eje.bar_label(b1, labels=[f"{v:.1%}" for v in ident], padding=8, fontsize=8)
eje.bar_label(b2, labels=[f"{v:.1%}" for v in verif], padding=8, fontsize=8)
eje.axhline(resultados_shot[1]["azar"], color="#8a8984", ls=":", lw=1.2)
eje.text(xs[0] - 0.45, resultados_shot[1]["azar"], "azar (identificación)", fontsize=8, va="bottom", ha="left",
         color="#555")
eje.set_xticks(xs, [f"{'One-shot' if k == 1 else 'Few-shot'}\nk = {k}" for k in K_FEW_SHOT])
eje.set(ylim=(0, 1.1), ylabel="Exactitud media (30 episodios)", title="Desempeño según imágenes de soporte por persona")
eje.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig(DIR_FIG / "08_one_shot_few_shot.png", bbox_inches="tight")
plt.show()

# %%
# ============================================================
# 6.4 Demostración visual One-shot y Few-shot
# ============================================================
def demo_shot(k, n_consultas, rng, archivo):
    clases = [c for c in np.unique(y[idx_pru]) if np.sum(y[idx_pru] == c) >= k + 1]
    propias = [c for c in clases if NOMBRES[c] in PERSONAS_PROPIAS]
    soporte_idx, protos = {}, {}
    for c in clases:
        idx_c = rng.permutation(idx_pru[y[idx_pru] == c])
        soporte_idx[c] = idx_c[:k]
        protos[c] = prototipo(EMB_PRUEBA[[POS_PRUEBA[i] for i in idx_c[:k]]])
    clases_arr = np.array(clases)
    P = np.stack([protos[c] for c in clases])
    elegidas = (propias + list(rng.permutation([c for c in clases if c not in propias])))[:n_consultas]
    fig, ejes = plt.subplots(len(elegidas), k + 2, figsize=(1.9 * (k + 2), 2.35 * len(elegidas)))
    for fila, c in enumerate(elegidas):
        consulta = rng.choice([i for i in idx_pru if y[i] == c and i not in soporte_idx[c]])
        d = np.sum((P - EMB_PRUEBA[POS_PRUEBA[consulta]]) ** 2, axis=1)
        pred = clases_arr[np.argmin(d)]
        d_min = d.min()
        aceptado = d_min <= UMBRAL
        texto = (f"{NOMBRES[pred].replace('_', ' ')}\nd² = {d_min:.3f}\n"
                 f"{'Misma Identidad' if aceptado else 'Identidad Diferente'}\n"
                 f"{'Verificado' if aceptado else 'Rechazado'}")
        color = COLOR_OK if aceptado and pred == c else COLOR_NO
        ejes[fila, 0].imshow(X[consulta])
        ejes[fila, 0].set_title(f"Consulta\n{NOMBRES[c].replace('_', ' ')}", fontsize=8)
        for s, i in enumerate(soporte_idx[c]):
            ejes[fila, s + 1].imshow(X[i])
            ejes[fila, s + 1].set_title(f"Soporte {s + 1}", fontsize=8)
        ejes[fila, -1].text(0.5, 0.5, texto, ha="center", va="center", fontsize=9, color=color, fontweight="bold",
                            transform=ejes[fila, -1].transAxes)
        for eje in ejes[fila]:
            eje.axis("off")
    modo = "One-shot (1 imagen de referencia)" if k == 1 else f"Few-shot (k = {k} imágenes, prototipo promedio)"
    fig.suptitle(f"{modo} · galería de {len(clases)} identidades · τ = {UMBRAL:.3f}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(DIR_FIG / archivo, bbox_inches="tight")
    plt.show()


demo_shot(1, 5, np.random.default_rng(11), "09_demo_one_shot.png")
demo_shot(3, 4, np.random.default_rng(12), "10_demo_few_shot_k3.png")
demo_shot(5, 4, np.random.default_rng(13), "11_demo_few_shot_k5.png")

# %%
# ============================================================
# 6.5 Espacio de embeddings (t-SNE de los vectores de 128 D de la prueba)
# ============================================================
fig, ejes = plt.subplots(1, 2, figsize=(12, 5))
paleta = plt.get_cmap("tab20")
for eje, emb, titulo in ((ejes[0], EMB_PRUEBA_INICIAL, "Antes de entrenar"),
                         (ejes[1], EMB_PRUEBA, "Después de Triplet Loss")):
    proy = TSNE(n_components=2, perplexity=15, random_state=SEED, init="pca").fit_transform(emb)
    for k, c in enumerate(np.unique(y[idx_pru])):
        m = y[idx_pru] == c
        eje.scatter(proy[m, 0], proy[m, 1], s=22, color=paleta(k % 20), label=NOMBRES[c].replace("_", " "))
    eje.set(title=f"t-SNE de los embeddings de prueba — {titulo}", xticks=[], yticks=[])
ejes[1].legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig(DIR_FIG / "12_tsne_embeddings.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 6.6 Guardado de métricas
#
# Todas las cifras se guardan en `results/metricas.json`: configuración, tamaño de las particiones, historial de
# entrenamiento, umbral, métricas de verificación antes/después y resultados *one-shot* / *few-shot*.

# %%
# ============================================================
# 6.6 Exportación de resultados
# ============================================================
for r in resultados_verificacion.values():
    r.pop("_d_pru", None)

metricas = {
    "entorno": {"tensorflow": tf.__version__, "keras": keras.__version__, "opencv": cv2.__version__,
                "gpu": bool(tf.config.list_physical_devices("GPU")), "colab": EN_COLAB, "detector": DETECTOR_NOMBRE},
    "configuracion": {"tam_rostro": TAM_ROSTRO, "margen_recorte": MARGEN_RECORTE, "clahe": ECUALIZAR_ILUMINACION,
                      "dim_embedding": DIM_EMBEDDING, "alpha": ALPHA, "tam_lote": TAM_LOTE,
                      "tripletas_por_epoca": TRIPLETAS_POR_EPOCA, "epocas_fase1": EPOCAS_FASE1,
                      "epocas_fase2": EPOCAS_FASE2, "lr_fase1": LR_FASE1, "lr_fase2": LR_FASE2,
                      "capas_ajuste_fino": CAPAS_AJUSTE_FINO, "mineria_desde_epoca": MINERIA_DESDE_EPOCA,
                      "episodios": EPISODIOS, "semilla": SEED},
    "dataset": {"personas_propias": PERSONAS_PROPIAS, "identidades_lfw": len(PERSONAS_LFW),
                "imagenes_totales": len(inventario), "rostros_procesados": int(len(X)),
                "descartadas": len(descartadas), "particiones": resumen_particiones,
                "ids_prueba_lfw": sorted(IDS_PRUEBA_LFW), "pares_validacion": int(len(etq_val)),
                "pares_prueba": int(len(etq_pru)), "pares_prueba_genuinos": int(etq_pru.sum())},
    "modelo": {"parametros_totales": int(red_embedding.count_params()),
               "parametros_entrenables_fase2": int(sum(int(np.prod(v.shape))
                                                       for v in red_embedding.trainable_variables))},
    "entrenamiento": {"historial": historial, "mejor_epoca": int(mejor["epoca"]),
                      "mejor_perdida_val": float(mejor["perdida_val"]), "perdida_val_inicial": perdida_val_inicial,
                      "exactitud_val_inicial": exactitud_val_inicial, "minutos": TIEMPO_ENTRENAMIENTO / 60},
    "verificacion": resultados_verificacion,
    "one_few_shot": {str(k): v for k, v in resultados_shot.items()},
}
(DIR_RESULTADOS / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=1), encoding="utf-8")
print("Métricas guardadas en", DIR_RESULTADOS / "metricas.json")
print("Figuras:", sorted(p.name for p in DIR_FIG.glob("*.png")))

# %% [markdown]
# ## 7. Conclusiones
#
# * La red siamesa no aprende a clasificar personas fijas, sino una **función de distancia**: dos fotos de la misma
#   persona producen *embeddings* cercanos y dos personas distintas, *embeddings* lejanos. Por eso puede reconocer a
#   personas que **nunca vio** durante el entrenamiento, con una sola foto de referencia.
# * La **Triplet Loss** separa las distribuciones de distancias intra-clase e inter-clase; el umbral $\tau$ elegido en
#   validación convierte esa distancia en la decisión *Verificado* / *Rechazado*.
# * El **preprocesamiento** (detección, alineación, recorte e iluminación) es tan importante como la red: si el
#   recorte incluye fondo o la cara está girada, la distancia mide esas diferencias en lugar de la identidad.
# * Pasar de **one-shot** a **few-shot** mejora la exactitud porque el prototipo promedio de k fotos es más estable
#   frente a cambios de pose, gesto e iluminación que una sola imagen.
