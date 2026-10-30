"""Genera docs/Informe_Redes_Neuronales_V.docx (formato APA 7.ª edición, trabajo de estudiante).

Todas las cifras del informe se leen de /results (metricas.json) y las figuras de /results/figuras, de modo que el
documento siempre refleja la última ejecución del notebook. Si el notebook se ejecutó en Colab, copie la carpeta
results/ de Google Drive al repositorio antes de regenerar el informe.

Requisitos: python-docx, lxml, latex2mathml y el archivo MML2OMML.XSL que instala Microsoft Office
(convierte LaTeX -> MathML -> ecuaciones nativas de Word).

Uso (desde la raíz del repositorio):
    python herramientas/construir_informe.py
"""
import json
import re
from pathlib import Path

import latex2mathml.converter as l2m
import numpy as np
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_COLOR_INDEX, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from lxml import etree

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "results"
FIG = RES / "figuras"
SCRIPTS = RAIZ / "scripts"
SALIDA = RAIZ / "docs" / "Informe_Redes_Neuronales_V.docx"
XSL_OMML = Path(r"C:/Program Files/Microsoft Office/root/Office16/MML2OMML.XSL")

AUTOR = "Diego Yanzir Gutiérrez Camelo"
REPO = "https://github.com/DiegoYanzir/Taller-Redes-Neuronales-V"
FUENTE = "Times New Roman"
MONO = "Consolas"

# ======================================================================================
# 1. Lectura de resultados
# ======================================================================================
M = json.loads((RES / "metricas.json").read_text(encoding="utf-8"))


def pct(x, d=2):
    return f"{x * 100:.{d}f}".replace(".", ",") + " %"


def pp(x, d=2):
    return f"{x * 100:.{d}f}".replace(".", ",") + " pp"


def dec(x, d=4):
    return f"{x:.{d}f}".replace(".", ",")


def miles(n):
    return f"{int(n):,}".replace(",", " ")

# ======================================================================================
# 2. Utilidades de formato APA
# ======================================================================================
doc = Document()
XSLT = etree.XSLT(etree.parse(str(XSL_OMML)))
contador = {"tabla": 0, "figura": 0, "codigo": 0}


def configurar_documento():
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)       # carta
    for lado in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, lado, Inches(1))                               # márgenes de 2,54 cm
    estilos = doc.styles
    normal = estilos["Normal"]
    normal.font.name = FUENTE
    normal.font.size = Pt(12)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FUENTE)
    pf = normal.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    # Niveles de título APA 7: 1 centrado negrita; 2 izquierda negrita; 3 izquierda negrita cursiva
    for nivel, centrado, cursiva in ((1, True, False), (2, False, False), (3, False, True)):
        st = estilos[f"Heading {nivel}"]
        st.font.name = FUENTE
        st.font.size = Pt(12)
        st.font.bold = True
        st.font.italic = cursiva
        st.font.color.rgb = RGBColor(0, 0, 0)
        rpr = st.element.get_or_add_rPr()
        fuentes = rpr.find(qn("w:rFonts"))
        if fuentes is None:
            fuentes = OxmlElement("w:rFonts"); rpr.append(fuentes)
        for att in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            fuentes.set(qn(att), FUENTE)
        for att in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            if fuentes.get(qn(att)) is not None:
                del fuentes.attrib[qn(att)]
        spf = st.paragraph_format
        spf.alignment = WD_ALIGN_PARAGRAPH.CENTER if centrado else WD_ALIGN_PARAGRAPH.LEFT
        spf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
        spf.space_before = Pt(0)
        spf.space_after = Pt(0)
        spf.first_line_indent = Inches(0)
        spf.keep_with_next = True
    # Número de página en el encabezado (esquina superior derecha)
    p = sec.header.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    campo(p, "PAGE")


def campo(parrafo, instruccion):
    r = parrafo.add_run()
    for tipo, texto in (("begin", None), (None, instruccion), ("separate", None), (None, None), ("end", None)):
        if tipo:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), tipo)
            if tipo == "begin":
                el.set(qn("w:dirty"), "true")
            r._r.append(el)
        elif texto:
            it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = f" {texto} "
            r._r.append(it)
        else:
            t = OxmlElement("w:t"); t.text = "1"; r._r.append(t)
    r.font.name = FUENTE


def texto_con_formato(p, texto):
    """Soporta **negrita**, *cursiva*, `código` y $LaTeX$ (ecuación nativa de Word) en línea."""
    for trozo in re.split(r"(\$[^$]+\$|\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", texto):
        if not trozo:
            continue
        if trozo.startswith("$"):
            mml = etree.fromstring(l2m.convert(trozo[1:-1], display="inline"))
            p._p.append(XSLT(mml).getroot())
        elif trozo.startswith("**"):
            p.add_run(trozo[2:-2]).bold = True
        elif trozo.startswith("*"):
            p.add_run(trozo[1:-1]).italic = True
        elif trozo.startswith("`"):
            r = p.add_run(trozo[1:-1]); r.font.name = MONO; r.font.size = Pt(10.5)
        else:
            p.add_run(trozo)
    return p


def parrafo(texto, sangria=True, alineacion=None):
    p = doc.add_paragraph()
    if sangria:
        p.paragraph_format.first_line_indent = Inches(0.5)
    if alineacion is not None:
        p.alignment = alineacion
    return texto_con_formato(p, texto)


def titulo(texto, nivel):
    return doc.add_heading(texto, level=nivel)


def vinetas(items, numeradas=False):
    for i, it in enumerate(items, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.75)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        p.add_run(f"{i}.\t" if numeradas else "•\t")
        texto_con_formato(p, it)
        p.paragraph_format.tab_stops.add_tab_stop(Inches(0.75))


def ecuacion(tex, numero=None):
    mml = etree.fromstring(l2m.convert(tex, display="block"))
    omml = XSLT(mml).getroot()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.5
    p._p.append(omml)
    if numero is not None:
        p.add_run(f"\t({numero})")
        p.paragraph_format.tab_stops.add_tab_stop(Inches(6.5), alignment=2)  # derecha
    return p


def marcador(texto):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"[{texto}]")
    r.bold = True
    r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    return p


def etiqueta(tipo, titulo_texto):
    contador[tipo] += 1
    nombre = {"tabla": "Tabla", "figura": "Figura", "codigo": "Código"}[tipo]
    p1 = doc.add_paragraph()
    p1.paragraph_format.keep_with_next = True
    p1.paragraph_format.space_before = Pt(6)
    p1.add_run(f"{nombre} {contador[tipo]}").bold = True
    p2 = doc.add_paragraph()
    p2.paragraph_format.keep_with_next = True
    p2.add_run(titulo_texto).italic = True
    return contador[tipo]


def nota(texto):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_after = Pt(12)
    p.add_run("Nota. ").italic = True
    texto_con_formato(p, texto)
    for r in p.runs:
        r.font.size = Pt(10)


def figura(archivo, titulo_texto, nota_texto=None, ancho=6.3):
    n = etiqueta("figura", titulo_texto)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.keep_with_next = bool(nota_texto)
    p.add_run().add_picture(str(FIG / archivo), width=Inches(ancho))
    if nota_texto:
        nota(nota_texto)
    else:
        doc.add_paragraph().paragraph_format.line_spacing = 1.0
    return n


def bordes_celda(celda, **lados):
    tcPr = celda._tc.get_or_add_tcPr()
    tcb = tcPr.find(qn("w:tcBorders"))
    if tcb is None:
        tcb = OxmlElement("w:tcBorders"); tcPr.append(tcb)
    for lado in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{lado}")
        if lados.get(lado):
            el.set(qn("w:val"), "single"); el.set(qn("w:sz"), "8"); el.set(qn("w:color"), "000000")
        else:
            el.set(qn("w:val"), "nil")
        tcb.append(el)


def tabla(titulo_texto, encabezados, filas, anchos, nota_texto=None, tam=10, mono_cols=()):
    """Tabla con estilo APA: solo líneas horizontales (arriba, bajo el encabezado y al final)."""
    n = etiqueta("tabla", titulo_texto)
    t = doc.add_table(rows=1 + len(filas), cols=len(encabezados))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for j, ancho in enumerate(anchos):
        for fila in t.rows:
            fila.cells[j].width = Inches(ancho)
    for i, fila in enumerate([encabezados] + filas):
        for j, valor in enumerate(fila):
            c = t.rows[i].cells[j]
            c.width = Inches(anchos[j])
            p = c.paragraphs[0]
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.first_line_indent = Inches(0)
            if i == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = p.add_run(str(valor)); r.bold = True
            else:
                texto_con_formato(p, str(valor))
                if j in mono_cols:
                    for r in p.runs:
                        r.font.name = MONO
            for r in p.runs:
                r.font.size = Pt(tam)
            ultima = i == len(filas)
            bordes_celda(c, top=(i == 0), bottom=(i == 0 or ultima))
    # repetir encabezado si la tabla cruza de página
    trPr = t.rows[0]._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true"); trPr.append(th)
    if nota_texto:
        nota(nota_texto)
    else:
        doc.add_paragraph().paragraph_format.line_spacing = 1.0
    return n


def bloque_codigo(codigo, titulo_texto=None, tam=8.5):
    if titulo_texto:
        etiqueta("codigo", titulo_texto)
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = t.rows[0].cells[0]
    c.width = Inches(6.5)
    tcPr = c._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "F3F4F6")
    tcPr.append(shd)
    bordes_celda(c, top=True, bottom=True, left=True, right=True)
    lineas = codigo.rstrip("\n").split("\n")
    p = c.paragraphs[0]
    for k, linea in enumerate(lineas):
        if k:
            p = c.add_paragraph()
        pf = p.paragraph_format
        pf.line_spacing = 1.0
        pf.space_after = Pt(0)
        pf.first_line_indent = Inches(0)
        r = p.add_run(linea if linea else " ")
        r.font.name = MONO
        r.font.size = Pt(tam)
        r._r.get_or_add_rPr().get_or_add_rFonts().set(qn("w:hAnsi"), MONO)
        if linea.lstrip().startswith("#"):
            r.font.color.rgb = RGBColor(0x2E, 0x6B, 0x30)
    doc.add_paragraph().paragraph_format.line_spacing = 1.0


def salto_pagina():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

# ======================================================================================
# 3. Extracción de código desde los scripts
# ======================================================================================
def codigo_script(nombre):
    return (SCRIPTS / nombre).read_text(encoding="utf-8")


def celdas_codigo(nombre):
    """Devuelve solo las celdas de código (sin markdown) de un script en formato percent."""
    bloques = re.split(r"^# %%(.*)$", codigo_script(nombre), flags=re.M)
    celdas = [cuerpo.strip("\n") for marca, cuerpo in zip(bloques[1::2], bloques[2::2]) if "[markdown]" not in marca]
    return celdas


def fragmento(nombre, inicio, fin=None):
    """Líneas desde la que contiene `inicio` hasta la anterior a la que contiene `fin`."""
    lineas = codigo_script(nombre).split("\n")
    i = next(k for k, l in enumerate(lineas) if inicio in l)
    j = len(lineas) if fin is None else next(k for k, l in enumerate(lineas[i + 1:], i + 1) if fin in l)
    return "\n".join(lineas[i:j]).rstrip()


# ======================================================================================
# 4. Contenido
# ======================================================================================
configurar_documento()
S1 = "01_reconocimiento_facial_red_siamesa.py"
NB1 = S1.replace(".py", ".ipynb")
COLAB = f"https://colab.research.google.com/github/DiegoYanzir/Taller-Redes-Neuronales-V/blob/main/notebooks/{NB1}"
C, D, E, MOD = M["configuracion"], M["dataset"], M["entrenamiento"], M["modelo"]
H = E["historial"]
V = M["verificacion"]
VA, VD = V["antes"]["prueba"], V["despues"]["prueba"]
VVAL = V["despues"]["validacion"]
TAU, TAU0 = V["despues"]["umbral"], V["antes"]["umbral"]
SH = {int(k): v for k, v in M["one_few_shot"].items()}
PROPIAS = D["personas_propias"]
PART = D["particiones"]
N_EPOCAS = len(H["perdida"])
CON_PROPIAS = bool(PROPIAS)


def captura(n, texto):
    marcador(f"INSERTAR CAPTURA {n}: {texto}")


def lista_personas(nombres):
    nombres = [n.replace("_", " ") for n in nombres]
    return ", ".join(nombres[:-1]) + " y " + nombres[-1] if len(nombres) > 1 else nombres[0]


# ---------------- Portada (APA 7, trabajo de estudiante) ----------------
for _ in range(3):
    doc.add_paragraph()
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run("Redes Neuronales V: Reconocimiento Facial mediante Arquitecturas Siamesas").bold = True
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run("Verificación de identidad con Triplet Loss y aprendizaje One-shot y Few-shot en Google Colab")
doc.add_paragraph()
for linea in (AUTOR, "[Nombre del integrante 2, si aplica]",
              "Especialización en Inteligencia Artificial, Universidad de Cundinamarca",
              "Deep Learning – Conceptos", "[Nombre del docente]", "30 de octubre de 2026"):
    p = doc.add_paragraph(linea); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if linea.startswith("["):
        p.runs[0].font.highlight_color = WD_COLOR_INDEX.YELLOW
salto_pagina()

# ---------------- Tabla de contenido ----------------
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run("Tabla de Contenido").bold = True
p = doc.add_paragraph()
campo(p, 'TOC \\o "1-2" \\h \\z \\u')
salto_pagina()

# ---------------- Introducción ----------------
titulo("Introducción", 1)
parrafo("Los talleres anteriores abordaron redes que **clasifican**: dada una imagen, la red elige una de $K$ "
        "clases fijas conocidas de antemano. El reconocimiento facial no encaja en ese esquema. Un sistema de "
        "control de acceso debe reconocer a una persona registrada con **una o pocas fotografías**, y debe admitir "
        "personas nuevas sin reentrenar la red; además, debe saber **rechazar** a quien no está registrado. Este "
        "informe estudia la solución estándar a ese problema: las **redes neuronales siamesas**, que en lugar de "
        "una clase aprenden una **función de distancia** entre rostros (Bromley et al., 1993; Chopra et al., 2005; "
        "Schroff et al., 2015).")
parrafo("La parte práctica implementa en Google Colab, con TensorFlow + Keras, un sistema completo de verificación "
        "facial: construcción de un dataset propio con consentimiento informado, detección y alineación del rostro "
        "con MTCNN, una red de *embeddings* basada en MobileNetV2 que proyecta cada rostro en un vector de 128 "
        "dimensiones con norma 1, un modelo siamés de tres ramas con pesos compartidos entrenado con **Triplet "
        "Loss**, y una evaluación en escenarios **one-shot** (una foto de referencia) y **few-shot** (tres y cinco "
        "fotos). Las lecturas de la semana —*What are Siamese Neural Networks in Deep Learning?* (s.f.) y Jaiswal "
        "(2022)— se usan como punto de partida y se contrastan con los artículos originales del área.")
parrafo("**Todas las cifras de este informe provienen de una ejecución real del notebook** (CPU, semilla 42) y se "
        "leen automáticamente del archivo `results/metricas.json`. Los marcadores resaltados en amarillo con la "
        "forma [INSERTAR CAPTURA n: …] indican qué captura de pantalla de Google Colab debe pegarse en cada lugar; "
        "la sección 4.7 los reúne en una sola tabla.")
if not CON_PROPIAS:
    marcador("NOTA PARA EL AUTOR (borrar antes de entregar): esta versión del informe se generó con la ejecución de "
             "verificación que usa solo identidades públicas de LFW. Agregue sus fotos en dataset_facial/fotos_propias/"
             "<persona>/, ejecute de nuevo el notebook, copie la carpeta results/ al repositorio y regenere el informe "
             "con python herramientas/construir_informe.py para que cifras y figuras incluyan su dataset propio.")

# ======================================================================================
# 1. Información del proyecto
# ======================================================================================
salto_pagina()
titulo("1. Información del Proyecto", 1)
titulo("1.1 Datos Generales", 2)
tabla("Información general de la actividad",
      ["Campo", "Descripción"],
      [["Título", "Redes Neuronales V – Reconocimiento Facial mediante Arquitecturas Siamesas"],
       ["Curso", "Deep Learning – Conceptos, Especialización en Inteligencia Artificial, Universidad de Cundinamarca"],
       ["Semana", "10: redes siamesas, preprocesamiento para *Face Recognition*, Triplet Loss, one/few/zero-shot learning"],
       ["Autor", AUTOR],
       ["Repositorio oficial", REPO],
       ["Notebook en Colab", f"`notebooks/{NB1}`"],
       ["Tecnologías", "Python 3, TensorFlow 2 / Keras 3, MTCNN, OpenCV, scikit-learn, NumPy, Matplotlib"],
       ["Entorno", "Google Colab (GPU T4 recomendada; también ejecuta en CPU)"]],
      [1.6, 4.9])
titulo("1.2 Repositorio de GitHub", 2)
parrafo("El código fuente, el notebook listo para Google Colab, las figuras y las métricas de la ejecución se "
        "publican en el repositorio oficial del proyecto:")
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run(REPO); r.font.color.rgb = RGBColor(0x05, 0x63, 0xC1); r.underline = True
parrafo("La estructura del repositorio es la siguiente (la carpeta `dataset_facial/` se excluye del control de "
        "versiones mediante `.gitignore` porque contiene fotografías de personas, que son datos personales):",
        sangria=False)
bloque_codigo("""Taller-Redes-Neuronales-V/
├── notebooks/01_reconocimiento_facial_red_siamesa.ipynb   # notebook de Google Colab
├── scripts/01_reconocimiento_facial_red_siamesa.py        # mismo código en formato .py (celdas # %%)
├── results/
│   ├── figuras/            # 12 figuras generadas por el notebook
│   ├── metricas.json       # configuración, historial, umbral y métricas
│   └── salida_01_*.txt     # salida de consola de la ejecución
├── requirements.txt
└── README.md""", tam=8.5)
captura(1, "Página principal del repositorio https://github.com/DiegoYanzir/Taller-Redes-Neuronales-V mostrando "
           "las carpetas notebooks/, scripts/, results/ y el README")

titulo("1.3 Declaración Ética y de Datos", 2)
parrafo("**Nota metodológica.** El dataset de prueba de esta actividad fue construido utilizando **fotografías "
        "propias del autor y de participantes que otorgaron su consentimiento explícito e informado para su uso "
        "exclusivamente académico**. A cada participante se le explicó el objetivo de la actividad, el tipo de "
        "procesamiento que se aplicaría a sus imágenes (detección, recorte y extracción de un vector numérico de "
        "características), el lugar de almacenamiento y el plazo de conservación, y se le informó de su derecho a "
        "retirar sus imágenes en cualquier momento.")
parrafo("Los rasgos faciales son **datos biométricos** y, por tanto, **datos sensibles** según el artículo 5 de "
        "la Ley Estatutaria 1581 de 2012 de Colombia, cuyo tratamiento exige autorización explícita del titular "
        "(Congreso de la República de Colombia, 2012). En consecuencia se adoptaron las siguientes medidas:")
vinetas([
    "Las fotografías se almacenan únicamente en el Google Drive personal del autor y **no se publican** en el "
    "repositorio de GitHub (la carpeta `dataset_facial/` está excluida en `.gitignore`).",
    "Los *embeddings* y el modelo entrenado se usan solo para las pruebas descritas en este informe; no se integran "
    "a ningún sistema de vigilancia, identificación o control real.",
    "Las imágenes se eliminarán al finalizar el periodo académico o antes, si un participante lo solicita.",
    "El complemento de entrenamiento proviene de **Labeled Faces in the Wild** (LFW; Huang et al., 2007), un "
    "conjunto público creado específicamente para la investigación en verificación facial, distribuido por "
    "scikit-learn (Pedregosa et al., 2011).",
])
marcador("Complete: número de participantes, forma del consentimiento (formato firmado / aceptación escrita por "
         "mensaje) y fecha de recolección de las fotografías")

# ======================================================================================
# 2. Marco conceptual
# ======================================================================================
salto_pagina()
titulo("2. Marco Conceptual y Fundamentos de Deep Learning Facial", 1)

# ---------------- 2.1 Redes siamesas ----------------
titulo("2.1 Redes Neuronales Siamesas: Verificación frente a Clasificación", 2)
parrafo("Una **red neuronal siamesa** está formada por dos o más subredes **idénticas** —misma arquitectura y "
        "**exactamente los mismos pesos**— que procesan entradas distintas en paralelo. Cada subred transforma su "
        "entrada en un vector de características (*embedding*), y una función de distancia compara esos vectores "
        "para decidir si las entradas son similares. El nombre alude a los gemelos siameses: dos cuerpos que "
        "comparten un mismo organismo (*What are Siamese Neural Networks in Deep Learning?*, s.f.).")
parrafo("La arquitectura fue propuesta por Bromley et al. (1993) para verificar firmas manuscritas y fue llevada "
        "a la verificación de rostros por Chopra et al. (2005), quienes la entrenaron para que pares de la misma "
        "persona quedaran cerca y pares de personas distintas, lejos. Koch et al. (2015) la usaron para "
        "**reconocimiento one-shot** de caracteres, y Schroff et al. (2015) la extendieron a tres ramas con la "
        "**Triplet Loss** en FaceNet, que alcanzó un 99,63 % de exactitud en LFW.")
parrafo("Compartir los pesos es la propiedad esencial: garantiza que dos imágenes idénticas produzcan el mismo "
        "*embedding* sin importar por qué rama entren, de modo que la distancia entre *embeddings* es **simétrica** "
        "y mide solo diferencias de contenido. Además, el número de parámetros no crece con el número de ramas. En "
        "la práctica las ramas no son copias de la red sino **la misma instancia** aplicada varias veces; durante "
        "la retropropagación, los gradientes de las tres ramas se suman sobre un único conjunto de pesos.")
parrafo("La diferencia frente a la clasificación tradicional es de planteamiento (Tabla 2). Un clasificador con "
        "salida *softmax* aprende $P(\\text{clase}\\mid x)$ para un conjunto **cerrado** de $K$ identidades: añadir "
        "una persona exige cambiar la última capa y reentrenar con muchas fotos de ella, y ante un desconocido la red "
        "igualmente asigna una de las $K$ clases. Una red siamesa aprende una **similitud** $s(x_1, x_2)$ que "
        "generaliza a identidades nunca vistas; registrar a una persona consiste simplemente en guardar el "
        "*embedding* de su foto, y un desconocido se rechaza porque su distancia a todos los registrados supera el "
        "umbral.")
tabla("Clasificación tradicional frente a verificación con redes siamesas",
      ["Aspecto", "Clasificación (softmax)", "Verificación (red siamesa)"],
      [["Pregunta que responde", "¿A cuál de las $K$ clases pertenece $x$?", "¿Son $x_1$ y $x_2$ la misma persona?"],
       ["Salida de la red", "Vector de $K$ probabilidades", "*Embedding* de dimensión fija (aquí 128)"],
       ["Función de pérdida", "Entropía cruzada", "Contrastiva o Triplet Loss"],
       ["Datos por identidad", "Decenas o cientos de ejemplos", "Pocos; basta 1 foto de referencia para registrar"],
       ["Nueva identidad", "Cambiar la capa de salida y reentrenar", "Guardar su *embedding*, sin reentrenar"],
       ["Persona desconocida", "Se asigna a la clase más probable (error)", "Se rechaza: distancia > umbral"],
       ["Ejemplos de uso", "Reconocer dígitos, prendas, especies", "Desbloqueo facial, firmas, *re-identificación*"]],
      [1.55, 2.4, 2.55], tam=9.5,
      nota_texto="Elaboración propia a partir de Chopra et al. (2005), Koch et al. (2015) y Schroff et al. (2015).")
parrafo("En visión por computadora las redes siamesas se usan para verificación facial y de firmas, "
        "re-identificación de personas en cámaras (Hermans et al., 2017), búsqueda de imágenes por similitud, "
        "seguimiento de objetos y detección de cambios. Su desventaja es que el entrenamiento es más costoso (cada "
        "ejemplo de entrenamiento son pares o tripletas, cuyo número crece combinatoriamente) y que su salida es una "
        "distancia, no una probabilidad, por lo que requiere fijar un umbral.")

# ---------------- 2.2 Preprocesamiento ----------------
titulo("2.2 El Papel Crítico del Preprocesamiento Facial y la Calidad del Dataset", 2)
parrafo("Una red de *embeddings* aprende a medir diferencias entre imágenes. Si las imágenes de entrada difieren "
        "por factores ajenos a la identidad —fondo, escala, inclinación de la cabeza, iluminación—, la distancia "
        "terminará midiendo esos factores. Por eso, todos los sistemas de reconocimiento facial modernos siguen el "
        "mismo pipeline de cuatro etapas: **detección → alineación → representación → comparación** (Wang & Deng, "
        "2021). Jaiswal (2022) lo ilustra con la biblioteca `face_recognition`: primero `face_locations` localiza "
        "el rostro, luego `face_encodings` lo convierte en un vector y por último `compare_faces` decide si dos "
        "vectores corresponden a la misma persona por su distancia.")
vinetas([
    "**Detección.** Localiza la cara y descarta el fondo. Los detectores clásicos de Viola y Jones (2001) —las "
    "*Haar Cascades* de OpenCV— son rápidos pero fallan con giros e iluminación difícil. **MTCNN** (Zhang et al., "
    "2016) usa tres redes convolucionales en cascada (P-Net, R-Net, O-Net) que devuelven la caja del rostro y cinco "
    "puntos faciales (ojos, nariz y comisuras de la boca).",
    "**Alineación.** Usando los puntos de los ojos se rota la imagen para que la línea que los une quede horizontal. "
    "Así todas las caras quedan en una posición canónica y la red no gasta capacidad en aprender invariancia a la "
    "rotación. Taigman et al. (2014) mostraron que la alineación es decisiva para la exactitud de DeepFace.",
    "**Recorte y redimensionamiento.** Se recorta un cuadrado alrededor de la cara con un margen pequeño (que "
    "conserva mentón, frente y contorno) y se lleva a un tamaño fijo, porque la CNN exige entradas de igual forma.",
    "**Normalización fotométrica.** La conversión al espacio de color correcto (OpenCV lee en BGR), la ecualización "
    "del contraste y el escalado de los píxeles al rango que espera la red reducen la variación por iluminación y "
    "por cámara.",
])
parrafo("La **calidad y organización del dataset** influyen tanto como la arquitectura. Para un dataset facial "
        "pequeño los factores críticos son: (a) **variabilidad intra-clase controlada**: cada persona debe aparecer "
        "con distintas expresiones, ángulos leves, fondos e iluminación, para que la red aprenda que esas "
        "variaciones no cambian la identidad; (b) **iluminación**: sombras laterales fuertes o contraluz alteran la "
        "textura que la red usa para distinguir rostros y producen falsos rechazos; (c) **resolución y nitidez**: "
        "fotos borrosas o con la cara muy pequeña pierden los detalles finos; (d) **una sola cara por foto y "
        "etiquetas correctas**: una foto mal etiquetada genera tripletas contradictorias; y (e) **número de "
        "identidades**: la Triplet Loss necesita muchas personas distintas para aprender un espacio que separe "
        "identidades en general y no solo las del entrenamiento (Schroff et al., 2015). La técnica de *Data "
        "Augmentation* (volteo horizontal, rotación leve, brillo y contraste) amplía artificialmente la "
        "variabilidad de un dataset pequeño sin alterar la identidad.")

# ---------------- 2.3 Embeddings ----------------
titulo("2.3 Extracción y Comparación de Embeddings Faciales en Espacios Métricos", 2)
parrafo("Un **embedding facial** es la salida $f(x) \\in \\mathbb{R}^d$ de una red neuronal que proyecta una "
        "imagen de rostro $x$ en un vector de dimensión fija ($d = 128$ en FaceNet y en este trabajo). El objetivo "
        "del entrenamiento es que el espacio resultante sea **métrico con significado**: la distancia entre dos "
        "*embeddings* debe ser pequeña si y solo si los rostros pertenecen a la misma persona. Una vez entrenada la "
        "red, todas las tareas faciales se reducen a operaciones geométricas sobre vectores (Schroff et al., 2015): "
        "**verificación** (comparar una distancia con un umbral), **identificación** (buscar el vecino más cercano "
        "en una galería) y **agrupamiento** (*clustering* de fotos por persona).")
parrafo("El *embedding* se normaliza con la norma L2 para que viva en la hiperesfera unitaria, "
        "$\\lVert f(x) \\rVert_2 = 1$. Esto acota las distancias, impide que la red reduzca la pérdida simplemente "
        "escalando los vectores y vuelve equivalentes las dos métricas habituales. La distancia euclidiana al "
        "cuadrado y la similitud coseno son:")
ecuacion("d^2(x_1,x_2)=\\lVert f(x_1)-f(x_2)\\rVert_2^2=\\sum_{i=1}^{d}\\left(f_i(x_1)-f_i(x_2)\\right)^2", 1)
ecuacion("\\cos(x_1,x_2)=\\frac{f(x_1)\\cdot f(x_2)}{\\lVert f(x_1)\\rVert\\,\\lVert f(x_2)\\rVert}", 2)
parrafo("Con vectores de norma 1 ambas se relacionan por $d^2 = 2 - 2\\cos$, de modo que $d^2 \\in [0, 4]$: vale 0 "
        "para *embeddings* idénticos, 2 para vectores ortogonales y 4 para vectores opuestos. La decisión de "
        "verificación es entonces una regla de umbral:")
ecuacion("\\text{misma identidad} \\iff d^2(x_1,x_2) \\le \\tau", 3)
parrafo("El umbral $\\tau$ fija el compromiso entre seguridad y comodidad: un $\\tau$ pequeño reduce los **falsos "
        "positivos** (impostores aceptados) a costa de más **falsos negativos** (usuarios legítimos rechazados), y "
        "viceversa. Por eso se elige con datos de validación y se reporta junto con la curva ROC.")

# ---------------- 2.4 Triplet Loss ----------------
titulo("2.4 La Función de Costo Triplet Loss", 2)
parrafo("La **Triplet Loss** (Schroff et al., 2015) entrena la red con tripletas de imágenes:")
vinetas([
    "**Ancla** (*Anchor*, $A$): una imagen de referencia de una persona.",
    "**Positiva** (*Positive*, $P$): **otra** imagen de la **misma** persona que el ancla.",
    "**Negativa** (*Negative*, $N$): una imagen de una persona **distinta**.",
])
parrafo("El objetivo es que el ancla quede más cerca de la positiva que de la negativa, y no por poco, sino al menos "
        "por un **margen** $\\alpha$:")
ecuacion("\\lVert f(A)-f(P)\\rVert_2^2+\\alpha<\\lVert f(A)-f(N)\\rVert_2^2", 4)
parrafo("La pérdida penaliza cuánto se incumple esa desigualdad y vale cero cuando se cumple:")
ecuacion("L(A,P,N)=\\max\\left(\\lVert f(A)-f(P)\\rVert_2^2-\\lVert f(A)-f(N)\\rVert_2^2+\\alpha,\\;0\\right)", 5)
parrafo("y sobre un lote de $m$ tripletas se minimiza el promedio $J = \\frac{1}{m}\\sum_{i=1}^{m} "
        "L(A^{(i)}, P^{(i)}, N^{(i)})$. El gradiente actúa en dos direcciones a la vez: **acerca** $f(P)$ a $f(A)$ y "
        "**aleja** $f(N)$ de $f(A)$. El **margen** $\\alpha$ cumple dos funciones. Primero, evita la solución "
        "trivial: sin él, una red que produjera el mismo vector para todas las imágenes tendría $d(A,P) = d(A,N) = 0$ "
        "y pérdida nula; con el margen esa solución cuesta exactamente $\\alpha$. Segundo, impone una separación "
        "mínima entre las distribuciones de distancias de pares genuinos e impostores, que es lo que luego permite "
        "fijar un umbral. FaceNet usa $\\alpha = 0{,}2$, el mismo valor de este trabajo.")
parrafo("Según la posición de la negativa las tripletas se clasifican en **fáciles** ($d(A,N) > d(A,P) + \\alpha$, "
        "pérdida cero y gradiente nulo), **semi-difíciles** o *semi-hard* ($d(A,P) < d(A,N) < d(A,P) + \\alpha$) y "
        "**difíciles** o *hard* ($d(A,N) < d(A,P)$). Con tripletas aleatorias, la mayoría se vuelven fáciles en "
        "pocas épocas y el entrenamiento se estanca; por eso FaceNet selecciona negativas *semi-hard*, que aportan "
        "gradiente sin la inestabilidad de las más difíciles (Schroff et al., 2015; Hermans et al., 2017). La "
        "Tabla 3 muestra el cálculo con vectores bidimensionales de norma 1 y $A = (1, 0)$, verificado en el "
        "notebook.")
tabla("Cálculo de la Triplet Loss en los tres tipos de tripleta (α = 0,2)",
      ["Tipo", "f(P)", "f(N)", "d(A, P)", "d(A, N)", "L"],
      [["Fácil", "(0,8; 0,6)", "(−1; 0)", "0,40", "4,00", "max(0,40 − 4,00 + 0,2; 0) = 0"],
       ["*Semi-hard*", "(0,8; 0,6)", "(0,75; 0,66)", "0,40", "0,50", "max(0,40 − 0,50 + 0,2; 0) = 0,10"],
       ["Difícil", "(0,6; 0,8)", "(0,8; 0,6)", "0,80", "0,40", "max(0,80 − 0,40 + 0,2; 0) = 0,60"]],
      [0.9, 0.9, 0.9, 0.75, 0.75, 2.3], tam=9.5,
      nota_texto="$d$ denota la distancia euclidiana al cuadrado. La negativa *semi-hard* está más lejos que la "
                 "positiva ($0{,}50 > 0{,}40$) pero dentro del margen ($0{,}50 < 0{,}40 + \\alpha = 0{,}60$), por lo "
                 "que aún produce pérdida y gradiente. Valores impresos por la celda 5.1 del notebook.")

# ---------------- 2.5 One/Few/Zero-shot ----------------
titulo("2.5 One-shot, Few-shot y Zero-shot Learning en Biometría y Visión Artificial", 2)
parrafo("El aprendizaje profundo convencional supone abundantes ejemplos etiquetados por clase. Los paradigmas de "
        "aprendizaje con pocos datos invierten esa suposición: el modelo debe reconocer **clases nuevas** a partir "
        "de muy pocos ejemplos —o de ninguno— en el momento de uso, sin reentrenamiento. Todos comparten la misma "
        "idea: aprender en una fase previa un **espacio de representación** general (aquí, el espacio de "
        "*embeddings* faciales) en el que las clases nuevas se puedan reconocer por similitud.")
parrafo("En **One-shot Learning** se dispone de **un único ejemplo** de cada clase nueva (Fei-Fei et al., 2006). Es "
        "el caso típico del reconocimiento facial: una empresa registra a un empleado con una sola foto y desde "
        "entonces el sistema debe reconocerlo. La red siamesa resuelve el problema comparando la consulta con ese "
        "único *embedding* de referencia (Koch et al., 2015).")
parrafo("En **Few-shot Learning** se dispone de **pocos ejemplos** ($k$) por clase, con $k$ pequeño (típicamente 2–10). El "
        "problema se describe como *N-way k-shot*: $N$ clases con $k$ ejemplos de soporte cada una. Con varios "
        "ejemplos la referencia es más robusta: las *Prototypical Networks* (Snell et al., 2017) representan cada "
        "clase por el **prototipo**, la media de sus *embeddings* de soporte, y clasifican por el prototipo más "
        "cercano; Vinyals et al. (2016) formalizaron el entrenamiento por episodios con este esquema. En biometría "
        "equivale a registrar a una persona con varias fotos (de frente, con gafas, con distinta luz).")
parrafo("En **Zero-shot Learning** no hay **ningún ejemplo** visual de la clase nueva: el modelo la reconoce a "
        "partir de **información auxiliar** que la describe, como atributos semánticos (Lampert et al., 2009) o "
        "texto en lenguaje natural, que se proyectan al mismo espacio que las imágenes (Radford et al., 2021). En "
        "biometría su uso es limitado, porque una identidad no se puede describir con palabras con la precisión "
        "necesaria; se aplica más bien a reconocer **atributos** faciales no vistos (por ejemplo, \"persona con "
        "gafas y barba\") o a buscar personas a partir de descripciones testimoniales. La Tabla 4 resume las "
        "diferencias.")
tabla("Comparación entre One-shot, Few-shot y Zero-shot Learning",
      ["Criterio", "One-shot", "Few-shot", "Zero-shot"],
      [["Ejemplos de la clase nueva", "1", "$k$ pequeño (2–10)", "0"],
       ["Información de referencia", "Un *embedding*", "Prototipo: media de $k$ *embeddings*",
        "Descripción semántica (atributos, texto)"],
       ["Mecanismo", "Distancia a la única referencia", "Distancia al prototipo o vecinos más cercanos",
        "Proyección imagen ↔ semántica en un espacio común"],
       ["Modelos representativos", "Red siamesa (Koch et al., 2015)", "Prototypical / Matching Networks",
        "Atributos (Lampert et al., 2009), CLIP"],
       ["Robustez a pose y luz", "Baja: depende de una foto", "Mayor: promedia variaciones", "Depende de la descripción"],
       ["Aplicación biométrica", "Registro con una foto (documento, carné)", "Registro con varias fotos (desbloqueo facial)",
        "Búsqueda por atributos o descripción"],
       ["En este trabajo", "$k = 1$, galería de identidades no vistas", "$k = 3$ y $k = 5$ con prototipo",
        "Marco conceptual (no implementado)"]],
      [1.45, 1.55, 1.75, 1.75], tam=9,
      nota_texto="Elaboración propia a partir de Fei-Fei et al. (2006), Koch et al. (2015), Snell et al. (2017), "
                 "Lampert et al. (2009) y Radford et al. (2021).")

# ======================================================================================
# 3. Implementación
# ======================================================================================
salto_pagina()
titulo("3. Explicación Técnica de la Implementación", 1)
parrafo(f"Todo el trabajo práctico está en un único notebook de Google Colab, `{NB1}`, dividido en las secciones "
        "que se describen a continuación. El Anexo A reproduce el código completo.")

# ---------------- 3.1 Entorno y dataset ----------------
titulo("3.1 Configuración del Entorno y Estructura del Dataset", 2)
parrafo("La primera celda importa las bibliotecas, instala MTCNN (el único paquete que Colab no trae) y fija la "
        "semilla 42 en Python, NumPy y TensorFlow para que la ejecución sea reproducible. En Colab se monta Google "
        "Drive y se crea la carpeta `Redes_Neuronales_V/dataset_facial/` con una subcarpeta por persona dentro de "
        "`fotos_propias/`; el nombre de la subcarpeta es la etiqueta de identidad. Como alternativa a subir fotos, "
        "la celda 1.2 permite capturarlas con la cámara web del navegador.")
bloque_codigo(fragmento(S1, "# 1.1 Rutas de trabajo", "for carpeta in (DIR_PROPIAS"),
              "Montaje de Google Drive y rutas del dataset")
captura(2, "Celda 1.1 ejecutada: autorización de Google Drive y panel 'Archivos' de Colab con la estructura "
           "dataset_facial/fotos_propias/<una carpeta por persona>")
if CON_PROPIAS:
    parrafo(f"El dataset propio contiene {len(PROPIAS)} personas ({lista_personas(PROPIAS)}). ")
parrafo(f"Para entrenar un espacio métrico general la red necesita ver muchas identidades, por lo que el dataset "
        f"propio se complementa con **{D['identidades_lfw']} identidades de LFW** (Huang et al., 2007), con "
        f"{C.get('lfw_fotos', 20)} fotos cada una, que el notebook descarga y exporta a disco como archivos JPG. Así "
        f"las fotos públicas y las propias atraviesan **el mismo pipeline** de preprocesamiento. En total se "
        f"inventariaron **{D['imagenes_totales']} imágenes**.")
captura(3, "Salida de la celda 1.3: 'Imágenes encontradas', lista de personas propias con el número de fotos de "
           "cada una, y la figura de fotografías originales")
figura("01_dataset_original.png", "Muestra de fotografías originales del dataset antes del preprocesamiento",
       "Cada fila es una identidad. Las fotografías tienen fondo, encuadre e iluminación variables.", ancho=5.2)

# ---------------- 3.2 Preprocesamiento ----------------
titulo("3.2 Pipeline de Preprocesamiento Facial", 2)
parrafo("La función `preprocesar_imagen` aplica a cada foto la secuencia de la Tabla 5. Su justificación se "
        "desarrolló en la sección 2.2; aquí se detalla la implementación.")
tabla("Pasos del preprocesamiento implementado",
      ["Paso", "Implementación", "Resultado"],
      [["1. Lectura y color", "`cv2.imread` + `cv2.cvtColor(BGR→RGB)`; fotos > 1024 px se reducen", "Imagen RGB uint8"],
       ["2. Detección", "MTCNN, confianza ≥ 0,90, cara de mayor área; Haar Cascades como respaldo",
        "Caja $[x, y, w, h]$ y 5 puntos faciales"],
       ["3. Alineación", "Ángulo $\\theta = \\operatorname{atan2}(y_{od}-y_{oi},\\, x_{od}-x_{oi})$; "
        "`cv2.getRotationMatrix2D` + `warpAffine` alrededor del centro de los ojos", "Ojos horizontales"],
       ["4. Recorte", "Cuadrado de lado $1{,}2\\cdot\\max(w,h)$ centrado en la cara; relleno por reflexión en bordes",
        "Rostro con 20 % de margen"],
       ["5. Redimensionamiento", "`cv2.resize(…, (160, 160), INTER_AREA)`", "Tensor (160, 160, 3)"],
       ["6. Iluminación", "CLAHE (*clipLimit* 2, rejilla 8×8) sobre el canal L de LAB", "Contraste local uniforme"],
       ["7. Normalización", "Capa `Rescaling(1/127,5, offset=−1)` dentro del modelo", "Píxeles en [−1, 1]"],
       ["8. *Data Augmentation*", "`RandomFlip`, `RandomRotation(±9°)`, `RandomBrightness(±15 %)`, "
        "`RandomContrast(±15 %)`, solo durante el entrenamiento", "Nueva variante en cada época"]],
      [1.3, 3.5, 1.7], tam=9,
      nota_texto="$(x_{oi}, y_{oi})$ y $(x_{od}, y_{od})$ son las coordenadas del ojo izquierdo y derecho en la "
                 "imagen. CLAHE: *Contrast Limited Adaptive Histogram Equalization* (Zuiderveld, 1994).")
parrafo("La **alineación** es el paso más delicado. Con las coordenadas de los ojos devueltas por MTCNN se calcula "
        "el ángulo de inclinación de la cabeza y se rota toda la imagen alrededor del punto medio entre los ojos. "
        "Luego el centro de la caja se transforma con la misma matriz de rotación $M$ para recortar la cara ya "
        "nivelada. La **ecualización CLAHE** se aplica solo al canal de luminancia L del espacio LAB, de modo que se "
        "corrige el contraste sin alterar el tono de piel. La **normalización** al rango $[-1, 1]$ que espera "
        "MobileNetV2 se hace dentro del modelo, con lo cual la función de verificación recibe directamente píxeles "
        "en $[0, 255]$ y no hay riesgo de olvidar el escalado en inferencia.")
bloque_codigo(fragmento(S1, "def alinear_y_recortar(", "def preprocesar_imagen("),
              "Alineación por la línea de los ojos y recorte cuadrado")
parrafo(f"De las {D['imagenes_totales']} imágenes, **{D['rostros_procesados']}** produjeron un rostro válido y "
        f"{D['descartadas']} se descartaron por no detectarse una cara. Los rostros procesados se guardan en "
        f"`dataset_facial/procesado/` (caché que evita repetir la detección) y forman un tensor de forma "
        f"({D['rostros_procesados']}, 160, 160, 3) de tipo uint8. La Figura 2 muestra el antes y el después.")
captura(4, "Salida de la celda 2.3: 'Rostros procesados: … | descartados: …' y forma del tensor del dataset")
figura("02_preprocesamiento_antes_despues.png",
       "Antes y después del preprocesamiento facial y ejemplos de Data Augmentation",
       "Columna 1: foto original con la caja del rostro (azul) y los ojos detectados por MTCNN (naranja). "
       "Columna 2: rostro alineado y recortado a 160×160. Columna 3: tras CLAHE (entrada final de la red). "
       "Columnas 4–5: dos variantes aleatorias generadas por Data Augmentation.")
captura(5, "Figura 'Pipeline de preprocesamiento facial' generada por la celda 2.4, de preferencia mostrando "
           "fotos propias")

# ---------------- 3.3 Particiones y tripletas ----------------
titulo("3.3 Particiones del Dataset y Organización de las Tripletas", 2)
texto_propias = ("Las fotos de cada persona propia se dividen al 50 %: la mitad participa en el entrenamiento y la "
                 "otra mitad se reserva para la prueba. ") if CON_PROPIAS else \
                ("Las fotos de cada persona propia se dividen al 50 %: la mitad participa en el entrenamiento y la "
                 "otra mitad se reserva para la prueba (en la ejecución reportada no había fotos propias). ")
parrafo("La partición se hace **por identidad** para medir el aprendizaje *one-shot* de forma honesta. El 25 % de "
        f"las identidades LFW ({len(D['ids_prueba_lfw'])} personas) se reserva por completo para la prueba: el "
        "modelo **nunca** ve sus fotos durante el entrenamiento. " + texto_propias +
        "Del resto, un 20 % de las imágenes forma la validación, que se usa para vigilar la pérdida, guardar los "
        "mejores pesos y elegir el umbral. El archivo `manifiesto_particiones.csv` registra la partición de cada "
        "imagen.")
tabla("Distribución de las particiones",
      ["Partición", "Imágenes", "Identidades", "Uso"],
      [["Entrenamiento", PART["entrenamiento"]["imagenes"], PART["entrenamiento"]["identidades"],
        "Generación de tripletas y ajuste de los pesos"],
       ["Validación", PART["validacion"]["imagenes"], PART["validacion"]["identidades"],
        f"Pérdida de validación, mejores pesos y umbral $\\tau$ ({D['pares_validacion']} pares)"],
       ["Prueba", PART["prueba"]["imagenes"], PART["prueba"]["identidades"],
        f"Verificación ({D['pares_prueba']} pares) y episodios one/few-shot"]],
      [1.3, 0.9, 1.0, 3.3], tam=10)
parrafo("Las tripletas se generan en cada época: se elige al azar una identidad, dos de sus imágenes como ancla y "
        "positiva, y una imagen de otra identidad como negativa. Durante las dos primeras épocas las negativas son "
        f"aleatorias; desde la época {C['mineria_desde_epoca']} se calculan los *embeddings* de todo el "
        "entrenamiento con el modelo actual y se elige una negativa **semi-hard** para cada par $(A, P)$. Cada "
        f"época usa {C['tripletas_por_epoca']} tripletas en lotes de {C['tam_lote']}. Además, se guardan diez "
        "tripletas de ejemplo en `dataset_facial/tripletas_muestra/tripleta_XX/{ancla, positiva, negativa}.png`.")
bloque_codigo(fragmento(S1, "            ea, ep = embeddings[posicion[a]]", "        tripletas.append("),
              "Selección de la negativa semi-hard")
figura("03_tripletas_ejemplo.png", "Ejemplos de tripletas Ancla–Positiva–Negativa usadas en el entrenamiento",
       "Cada columna es una tripleta: ancla y positiva son fotos distintas de la misma persona; la negativa "
       "(título en naranja) es otra persona.", ancho=5.0)
captura(6, "Salida de la celda 3.1 con el número de imágenes e identidades por partición, y figura de tripletas "
           "de la celda 3.2")

# ---------------- 3.4 Arquitectura ----------------
titulo("3.4 Arquitectura de la Red Neuronal Siamesa", 2)
parrafo("La **red generadora de embeddings** $f$ aplica *transfer learning* sobre **MobileNetV2** (Sandler et "
        "al., 2018) preentrenada en ImageNet. MobileNetV2 se eligió porque ofrece buenas características visuales "
        "con solo 2,26 millones de parámetros, gracias a las convoluciones separables en profundidad y a los "
        "bloques residuales invertidos con cuellos de botella lineales, lo que permite entrenar en minutos en la "
        "GPU gratuita de Colab. Sobre ella se añade una cabeza de proyección que termina en el *embedding* de 128 "
        "dimensiones normalizado con L2 (Tabla 7).")
tabla("Arquitectura de la red de embeddings y dimensiones de los tensores",
      ["#", "Capa", "Forma de salida", "Activación / detalle", "Parámetros"],
      [["0", "Entrada `rostro`", "(None, 160, 160, 3)", "Píxeles RGB en [0, 255]", "0"],
       ["1", "`Rescaling` (normalización)", "(None, 160, 160, 3)", "$x/127{,}5 - 1 \\in [-1, 1]$", "0"],
       ["2", "MobileNetV2 (ImageNet, sin *top*)", "(None, 5, 5, 1280)", "ReLU6; 17 bloques residuales invertidos",
        "2 257 984"],
       ["3", "`GlobalAveragePooling2D`", "(None, 1280)", "Promedio espacial de cada mapa", "0"],
       ["4", "`Dropout`", "(None, 1280)", "$p = 0{,}3$ (solo en entrenamiento)", "0"],
       ["5", "`Dense`", "(None, 256)", "ReLU", "327 936"],
       ["6", "`BatchNormalization`", "(None, 256)", "Normaliza activaciones", "1 024"],
       ["7", "`Dense` (embedding)", "(None, 128)", "Lineal", "32 896"],
       ["8", "`UnitNormalization` (L2)", "(None, 128)", "$f(x)/\\lVert f(x)\\rVert_2$", "0"]],
      [0.3, 2.0, 1.45, 1.95, 0.8], tam=9,
      nota_texto=f"Total: {miles(MOD['parametros_totales'])} parámetros. En la fase 1 solo se entrenan las capas 5–7 "
                 f"(361 344 parámetros; los 512 restantes de BatchNormalization son medias y varianzas móviles, no entrenables); "
                 f"en la fase 2 se suman las últimas {C['capas_ajuste_fino']} capas de MobileNetV2, para "
                 f"{miles(MOD['parametros_entrenables_fase2'])} parámetros entrenables. Una capa `Dense` de $n$ a "
                 "$m$ neuronas tiene $n \\cdot m + m$ parámetros: $1280 \\cdot 256 + 256 = 327\\,936$.")
parrafo("El **modelo siamés** recibe tres tensores de forma (None, 160, 160, 3) —ancla, positiva y negativa— y "
        "aplica a cada uno **la misma instancia** `red_embedding`, devolviendo tres tensores de forma (None, 128). "
        "El notebook verifica con dos `assert` que el modelo siamés tiene exactamente los mismos pesos y el mismo "
        f"número de parámetros ({miles(MOD['parametros_totales'])}) que la red de *embeddings*: las tres ramas no "
        "agregan ni un parámetro.")
bloque_codigo(fragmento(S1, "def construir_red_embedding(", "red_embedding.summary("),
              "Red de embeddings con MobileNetV2 y modelo siamés de tres ramas con pesos compartidos")
captura(7, "Salida de red_embedding.summary() y modelo_siames.summary() de la celda 4.1, incluida la línea "
           "'Pesos compartidos por las 3 ramas'")

# ---------------- 3.5 Triplet Loss y entrenamiento ----------------
titulo("3.5 Triplet Loss y Configuración del Entrenamiento", 2)
parrafo("La Triplet Loss se implementa de forma explícita, traduciendo línea a línea la Ecuación 5. La función "
        "devuelve también las distancias $d(A,P)$ y $d(A,N)$ para medir la fracción de tripletas activas.")
bloque_codigo(fragmento(S1, "def triplet_loss(", "# Verificación numérica"), "Implementación de la Triplet Loss")
captura(8, "Salida de la celda 5.1: verificación numérica de la Triplet Loss con los casos fácil, semi-hard y "
           "difícil")
parrafo("El entrenamiento usa un bucle propio con `tf.GradientTape` compilado con `@tf.function`. En cada paso: se "
        "aplica *Data Augmentation* a las tres imágenes, se calculan los tres *embeddings* con los pesos "
        "compartidos, se evalúa la Triplet Loss, se obtienen los gradientes respecto a los pesos entrenables y se "
        "actualizan con Adam (Kingma & Ba, 2015). Al final de cada época se calculan la pérdida y la exactitud de "
        "tripletas sobre un conjunto fijo de tripletas de validación, y se guardan los pesos de la mejor época.")
bloque_codigo(fragmento(S1, "def crear_paso_entrenamiento(", "def evaluar_tripletas("),
              "Paso de entrenamiento con GradientTape")
tabla("Hiperparámetros del entrenamiento",
      ["Hiperparámetro", "Valor", "Justificación"],
      [["Margen $\\alpha$", dec(C["alpha"], 1), "Valor de FaceNet para *embeddings* de norma 1"],
       ["Dimensión del *embedding*", C["dim_embedding"], "Compromiso entre capacidad y compacidad (FaceNet)"],
       ["Optimizador", "Adam ($\\beta_1 = 0{,}9$; $\\beta_2 = 0{,}999$)", "Tasa adaptativa por parámetro"],
       ["Fase 1: épocas / $\\eta$", f"{C['epocas_fase1']} / $10^{{-3}}$", "Backbone congelado: entrena la cabeza"],
       ["Fase 2: épocas / $\\eta$", f"{C['epocas_fase2']} / $5\\cdot10^{{-5}}$",
        f"Ajuste fino de las últimas {C['capas_ajuste_fino']} capas; $\\eta$ bajo para no destruir lo aprendido"],
       ["BatchNorm del backbone", "Congeladas", "Lotes pequeños darían estadísticos ruidosos"],
       ["Tamaño de lote", C["tam_lote"], "32 tripletas = 96 imágenes por paso"],
       ["Tripletas por época", C["tripletas_por_epoca"], "Se regeneran en cada época"],
       ["Minería", f"*Semi-hard* desde la época {C['mineria_desde_epoca']}", "Evita tripletas fáciles sin gradiente"],
       ["Selección del modelo", "Mínima pérdida de validación", "Equivale a Early Stopping con restauración"]],
      [1.75, 1.75, 3.0], tam=9.5)

# ======================================================================================
# 4. Resultados
# ======================================================================================
salto_pagina()
titulo("4. Análisis de Resultados y Evidencias de Ejecución", 1)
entorno = "GPU" if M["entorno"]["gpu"] else "CPU"
parrafo(f"Los resultados corresponden a la ejecución del notebook con semilla 42 en {entorno} "
        f"(TensorFlow {M['entorno']['tensorflow']}, detector {M['entorno']['detector']}); el entrenamiento tomó "
        f"{dec(E['minutos'], 1)} minutos. "
        + (f"El conjunto de prueba incluye las fotos reservadas de {lista_personas(PROPIAS)} y "
           f"{len(D['ids_prueba_lfw'])} identidades LFW no vistas." if CON_PROPIAS else
           f"El conjunto de prueba está formado por {len(D['ids_prueba_lfw'])} identidades LFW que el modelo nunca "
           "vio durante el entrenamiento."))

# ---------------- 4.1 Convergencia ----------------
titulo("4.1 Convergencia del Entrenamiento", 2)
h_p, h_v, h_a, h_e = H["perdida"], H["perdida_val"], H["activas"], H["exactitud_val"]
f1 = C["epocas_fase1"]
parrafo(f"Antes de entrenar, la cabeza de proyección tiene pesos aleatorios y la pérdida de validación es "
        f"{dec(E['perdida_val_inicial'])}, con una exactitud de tripletas (proporción de tripletas con "
        f"$d(A,P) < d(A,N)$) de {pct(E['exactitud_val_inicial'], 1)}: las características de ImageNet ya separan "
        f"algo los rostros, pero de forma pobre. Durante la fase 1 la pérdida de validación baja de "
        f"{dec(h_v[0])} a {dec(h_v[f1 - 1])}. En la fase 2 "
        f"el ajuste fino del backbone lleva la pérdida de validación a su mínimo, {dec(E['mejor_perdida_val'])}, "
        f"en la época {E['mejor_epoca']}, con una exactitud de tripletas de validación de "
        f"{pct(h_e[E['mejor_epoca'] - 1], 1)}.")
parrafo(f"La curva de la derecha de la Figura 4 explica la dinámica: la fracción de tripletas **activas** "
        f"($L > 0$) baja de {pct(h_a[0], 0)} en la primera época a {pct(h_a[C['mineria_desde_epoca'] - 2], 0)} en la "
        f"época {C['mineria_desde_epoca'] - 1}, porque las tripletas aleatorias se vuelven fáciles. Al activar la "
        f"minería *semi-hard* en la época {C['mineria_desde_epoca']} la fracción vuelve a subir "
        f"({pct(h_a[C['mineria_desde_epoca'] - 1], 0)}) y la pérdida de entrenamiento salta: el modelo recibe "
        "tripletas más informativas, más difíciles que las aleatorias. Por eso, a partir de esa época, la pérdida "
        "de entrenamiento no es comparable con la de validación (que siempre usa las mismas tripletas fijas) y la "
        "convergencia se juzga con esta última.")
figura("04_curva_convergencia.png", "Curva de convergencia de la Triplet Loss",
       "Izquierda: pérdida media por época (la línea punteada marca $\\alpha$, la pérdida de una red que produce el "
       "mismo vector para toda imagen). Derecha: fracción de tripletas activas y exactitud de tripletas en "
       "validación. Zona sombreada: fase 2 (ajuste fino). Línea discontinua: época con los mejores pesos.")
captura(9, "Salida del bucle de entrenamiento de la celda 5.2 (líneas 'Época …' de ambas fases y la línea final "
           "con el tiempo y la mejor época)")
captura(10, "Figura de la curva de convergencia generada por la celda 5.3")

# ---------------- 4.2 Distancias ----------------
titulo("4.2 Distancias Intra-clase frente a Inter-clase y Umbral de Decisión", 2)
parrafo(f"Para evaluar la verificación se forman todos los pares genuinos de la prueba (misma persona, "
        f"{D['pares_prueba_genuinos']} pares) y el mismo número de pares impostores (personas distintas). El "
        f"umbral óptimo se eligió en validación, maximizando la exactitud: $\\tau = {dec(TAU, 3)}$ (en distancia "
        f"euclidiana al cuadrado, equivalente a una similitud coseno de {dec(1 - TAU / 2, 3)}). La Tabla 9 compara "
        "el modelo antes y después del entrenamiento sobre los pares de prueba.")
tabla("Métricas de verificación facial sobre el conjunto de prueba",
      ["Métrica", "Antes de entrenar", "Después de Triplet Loss"],
      [["Umbral $\\tau$ (elegido en validación)", dec(TAU0, 3), dec(TAU, 3)],
       ["Distancia intra-clase media ± DE", f"{dec(VA['distancia_intra_media'], 3)} ± {dec(VA['distancia_intra_std'], 3)}",
        f"{dec(VD['distancia_intra_media'], 3)} ± {dec(VD['distancia_intra_std'], 3)}"],
       ["Distancia inter-clase media ± DE", f"{dec(VA['distancia_inter_media'], 3)} ± {dec(VA['distancia_inter_std'], 3)}",
        f"{dec(VD['distancia_inter_media'], 3)} ± {dec(VD['distancia_inter_std'], 3)}"],
       ["Exactitud de verificación", pct(VA["exactitud"]), f"**{pct(VD['exactitud'])}**"],
       ["Tasa de verdaderos positivos (TVP)", pct(VA["tasa_verdaderos_positivos"]),
        pct(VD["tasa_verdaderos_positivos"])],
       ["Tasa de falsos positivos (TFP)", pct(VA["tasa_falsos_positivos"]), pct(VD["tasa_falsos_positivos"])],
       ["Precisión", pct(VA["precision"]), pct(VD["precision"])],
       ["Área bajo la curva ROC (AUC)", dec(VA["auc"], 3), f"**{dec(VD['auc'], 3)}**"]],
      [2.6, 1.8, 2.1], tam=10,
      nota_texto=f"{D['pares_prueba']} pares de prueba. Distancias: euclidiana al cuadrado entre *embeddings* de norma "
                 f"1, rango [0, 4]. DE: desviación estándar. En validación el modelo entrenado obtuvo una exactitud "
                 f"de {pct(VVAL['exactitud'])} y un AUC de {dec(VVAL['auc'], 3)}.")
sep_antes = VA["distancia_inter_media"] - VA["distancia_intra_media"]
sep_desp = VD["distancia_inter_media"] - VD["distancia_intra_media"]


def dprima(m):
    return (m["distancia_inter_media"] - m["distancia_intra_media"]) / np.sqrt(
        (m["distancia_intra_std"] ** 2 + m["distancia_inter_std"] ** 2) / 2)


parrafo(f"El efecto de la Triplet Loss se ve en la separación entre las dos distribuciones. Antes de entrenar, la "
        f"distancia media de los pares genuinos ({dec(VA['distancia_intra_media'], 3)}) y la de los impostores "
        f"({dec(VA['distancia_inter_media'], 3)}) difieren en solo {dec(sep_antes, 3)}, y los histogramas se "
        f"superponen casi por completo (Figura 5, izquierda): con la cabeza de proyección aleatoria todos los "
        f"*embeddings* ocupan una región pequeña de la hiperesfera. El entrenamiento los dispersa, por lo que "
        f"**ambas** distancias medias crecen, pero de forma muy desigual: la intra-clase pasa a "
        f"{dec(VD['distancia_intra_media'], 3)} y la inter-clase a {dec(VD['distancia_inter_media'], 3)}, de modo "
        f"que la diferencia sube a {dec(sep_desp, 3)}. Como las escalas cambian, la comparación justa es el índice "
        f"de separabilidad $d' = (\\mu_{{inter}} - \\mu_{{intra}}) / \\sqrt{{(\\sigma_{{intra}}^2 + "
        f"\\sigma_{{inter}}^2)/2}}$, que mide la distancia entre ambas distribuciones en desviaciones estándar: "
        f"pasa de {dec(dprima(VA), 2)} a **{dec(dprima(VD), 2)}**. En consecuencia, el AUC pasa de {dec(VA['auc'], 3)} a "
        f"{dec(VD['auc'], 3)} y la exactitud de verificación de {pct(VA['exactitud'], 1)} a {pct(VD['exactitud'], 1)}, "
        "**sobre identidades que el modelo no vio nunca**, lo que confirma que aprendió una métrica general de "
        "similitud facial y no a memorizar las personas del entrenamiento.")
parrafo(f"La superposición residual entre ambas distribuciones explica los errores: los pares genuinos con "
        f"distancia mayor que $\\tau$ son **falsos rechazos** ({VD['fn']} pares) y los pares impostores con "
        f"distancia menor son **falsas aceptaciones** ({VD['fp']} pares). Los primeros suelen corresponder a fotos "
        "con giro pronunciado, gafas de sol o iluminación muy distinta, y los segundos a personas de rasgos "
        "parecidos fotografiadas con pose e iluminación similares. "
        "Mover el umbral desplaza errores de un tipo al otro, como describe la curva ROC; en una aplicación de "
        "seguridad convendría un $\\tau$ más bajo para reducir la TFP.")
figura("05_distancias_intra_inter_roc.png",
       "Distribución de distancias intra-clase e inter-clase antes y después del entrenamiento, y curva ROC",
       "Azul: pares de la misma persona. Naranja: pares de personas distintas. Línea discontinua: umbral $\\tau$ "
       "elegido en validación. Derecha: curvas ROC antes (punteada) y después (continua) del entrenamiento.")
captura(11, "Salida de la celda 6.1 con el umbral τ, la exactitud y el AUC antes y después, y la figura de "
            "histogramas y curva ROC")

# ---------------- 4.3 Verificación ----------------
titulo("4.3 Verificación de Identidad entre Pares y Tripletas", 2)
parrafo("La función `verificar_identidad` encapsula el sistema completo: recibe dos fotos crudas, aplica el mismo "
        "preprocesamiento del entrenamiento, calcula ambos *embeddings*, mide la distancia euclidiana al cuadrado y "
        "la similitud coseno, y devuelve el veredicto **\"Misma Identidad - Verificado\"** si $d^2 \\le \\tau$ o "
        "**\"Identidad Diferente - Rechazado\"** en caso contrario.")
bloque_codigo(fragmento(S1, "def verificar_identidad(", "def mostrar_pares("), "Función de verificación de identidad")
figura("06_verificacion_pares.png", "Verificación facial sobre pares de prueba",
       "Las cuatro primeras columnas son pares de la misma persona y las cuatro últimas, de personas distintas. "
       "Bajo cada par se muestran la distancia, la similitud coseno y el veredicto (verde: verificado; rojo: "
       "rechazado), y si la decisión fue un acierto o un error.")
figura("07_tripletas_prueba_distancias.png", "Tripletas de prueba con las distancias calculadas por el modelo",
       "La distancia ancla–positiva debe ser menor que $\\tau$ (verde) y la ancla–negativa, mayor (rojo).", ancho=4.2)
captura(12, "Figura de verificación de pares de la celda 6.2 y la línea impresa 'verificar_identidad(…) → {…}' "
            "con dos fotos propias")

# ---------------- 4.4 One/Few-shot ----------------
titulo("4.4 Reconocimiento One-shot y Few-shot", 2)
parrafo(f"Para cada valor de $k$ se ejecutaron {C['episodios']} episodios. En cada episodio se eligen al azar $k$ "
        "fotos de soporte por persona de prueba —que forman la galería— y el resto de sus fotos actúan como "
        "consultas. Con $k = 1$ la referencia es un único *embedding* (**one-shot**); con $k = 3$ y $k = 5$ es el "
        "**prototipo**, la media renormalizada de los $k$ *embeddings* (**few-shot**). Se mide la **identificación** "
        "(¿el prototipo más cercano es el de la persona correcta, entre todas las de la galería?) y la "
        "**verificación** (¿la consulta se acepta frente a su propio prototipo y se rechaza frente al de otra "
        "persona usando $\\tau$?).")
filas_shot = []
for k in sorted(SH):
    r = SH[k]
    filas_shot.append([f"{'One-shot' if k == 1 else 'Few-shot'} ($k = {k}$)", r["identidades"],
                       r["consultas_por_episodio"],
                       f"{pct(r['identificacion_media'], 1)} ± {pct(r['identificacion_std'], 1)}",
                       f"{pct(r['verificacion_media'], 1)} ± {pct(r['verificacion_std'], 1)}"])
tabla("Exactitud en los escenarios One-shot y Few-shot",
      ["Escenario", "Identidades en la galería", "Consultas por episodio", "Identificación (rank-1)",
       "Verificación con umbral τ"],
      filas_shot, [1.5, 1.1, 1.1, 1.4, 1.4], tam=9.5,
      nota_texto=f"Media ± desviación estándar sobre {C['episodios']} episodios. La exactitud de identificación por "
                 f"azar sería {pct(SH[1]['azar'], 1)} (1 entre {SH[1]['identidades']} identidades).")
k_max = max(SH)
parrafo(f"Con **una sola foto de referencia** el modelo identifica correctamente a la persona en el "
        f"{pct(SH[1]['identificacion_media'], 1)} de las consultas, frente a un {pct(SH[1]['azar'], 1)} esperable por "
        f"azar, entre identidades que nunca vio durante el entrenamiento. Al pasar a $k = 3$ la identificación sube "
        f"a {pct(SH[3]['identificacion_media'], 1)} y con $k = {k_max}$ llega a "
        f"{pct(SH[k_max]['identificacion_media'], 1)}. La mejora se explica porque una sola foto puede estar "
        "afectada por una pose, una expresión o una iluminación particulares, mientras que el prototipo promedia "
        "esas variaciones y se acerca al «centro» de la identidad en el espacio de *embeddings*. La verificación "
        f"con umbral sigue la misma tendencia ({pct(SH[1]['verificacion_media'], 1)} con $k = 1$ frente a "
        f"{pct(SH[k_max]['verificacion_media'], 1)} con $k = {k_max}$) y la desviación estándar entre episodios es "
        "menor con varias fotos: el resultado depende menos de qué foto se eligió como referencia. Es el mismo motivo por el que los sistemas de desbloqueo facial piden registrar el rostro "
        "desde varios ángulos.")
figura("08_one_shot_few_shot.png", "Exactitud de identificación y verificación según el número de imágenes de soporte",
       "Barras: media de 30 episodios; bigotes: desviación estándar. Línea punteada: identificación por azar.",
       ancho=4.8)
figura("09_demo_one_shot.png", "Demostración de reconocimiento One-shot",
       "Para cada consulta (primera columna) se muestra la única imagen de referencia de esa persona en la galería "
       "y, a la derecha, la identidad más cercana, su distancia y el veredicto.", ancho=4.0)
figura("11_demo_few_shot_k5.png", "Demostración de reconocimiento Few-shot con k = 5 imágenes de soporte",
       "La referencia de cada persona es el prototipo de sus cinco imágenes de soporte.", ancho=6.0)
captura(13, "Salida impresa de la celda 6.3 (líneas 'One-shot (k=1)…' y 'Few-shot (k=3/5)…') y la gráfica de "
            "barras")
captura(14, "Figuras de demostración One-shot y Few-shot de la celda 6.4, de preferencia con una consulta de una "
            "persona propia")

# ---------------- 4.5 t-SNE ----------------
titulo("4.5 Visualización del Espacio de Embeddings", 2)
parrafo("La Figura 11 proyecta a dos dimensiones, con t-SNE (van der Maaten & Hinton, 2008), los *embeddings* de "
        "128 D de las imágenes de prueba. Antes del entrenamiento los puntos de las distintas personas aparecen "
        "mezclados; después, las fotos de cada persona forman grupos compactos y separados entre sí. Es la "
        "representación visual de lo que mide la Tabla 9: la Triplet Loss hizo que la dispersión intra-clase fuera pequeña en relación con "
        "la separación inter-clase, que es la condición para que una regla de umbral funcione.")
figura("12_tsne_embeddings.png", "Proyección t-SNE de los embeddings de prueba antes y después del entrenamiento",
       "Cada color es una persona del conjunto de prueba. t-SNE conserva vecindades locales, no distancias "
       "absolutas.")
captura(15, "Figura t-SNE de la celda 6.5")

# ---------------- 4.6 Limitaciones ----------------
titulo("4.6 Limitaciones y Consideraciones", 2)
vinetas([
    "**Tamaño del dataset.** FaceNet se entrenó con 100–200 millones de imágenes de 8 millones de personas; aquí "
    f"se usaron {PART['entrenamiento']['imagenes']} imágenes de {PART['entrenamiento']['identidades']} identidades. "
    "Por eso la exactitud es menor que la de los sistemas comerciales; aumentar el número de identidades es la "
    "mejora con mayor impacto.",
    "**Backbone general.** MobileNetV2 se preentrenó en ImageNet (objetos), no en rostros. Un backbone preentrenado "
    "en caras (FaceNet, ArcFace) daría *embeddings* mucho más discriminativos desde el inicio.",
    "**Umbral dependiente del dominio.** $\\tau$ se eligió con fotos tipo LFW; con fotos de webcam de otra calidad "
    "conviene recalibrarlo con algunos pares del nuevo dominio.",
    "**Sesgo demográfico.** LFW está dominado por figuras públicas adultas y de piel clara; los sistemas "
    "entrenados con datos desbalanceados tienen más errores en grupos subrepresentados (Buolamwini & Gebru, 2018). "
    "Un sistema real debe evaluarse por subgrupos antes de usarse.",
    "**Ataques de presentación.** El sistema no distingue un rostro real de una foto impresa o en pantalla; un uso "
    "en control de acceso requeriría detección de vida (*liveness detection*).",
])

# ---------------- 4.7 Guía de evidencias ----------------
titulo("4.7 Guía de Evidencias de Ejecución", 2)
parrafo("La Tabla 11 reúne las capturas de pantalla de Google Colab que evidencian la ejecución, en el orden en "
        "que aparecen los marcadores [INSERTAR CAPTURA n] en este documento. Se recomienda capturar la celda "
        "completa (código y salida) e incluir el nombre del notebook visible en la parte superior.")
tabla("Capturas de pantalla requeridas como evidencia",
      ["N.º", "Sección del notebook", "Qué debe verse en la captura", "Sección del informe"],
      [["1", "GitHub", "Repositorio con notebooks/, scripts/, results/ y README", "1.2"],
       ["2", "1.1 Rutas", "Drive montado y carpetas `fotos_propias/<persona>/`", "3.1"],
       ["3", "1.3 Inventario", "Conteo de fotos por persona propia y figura de fotos originales", "3.1"],
       ["4", "2.3 Procesamiento", "Rostros procesados/descartados y forma (N, 160, 160, 3)", "3.2"],
       ["5", "2.4 Antes/después", "Figura del pipeline con fotos propias", "3.2"],
       ["6", "3.1–3.2 Particiones", "Imágenes por partición y figura de tripletas", "3.3"],
       ["7", "4.1 Arquitectura", "`summary()` de ambos modelos y 'Pesos compartidos'", "3.4"],
       ["8", "5.1 Triplet Loss", "Verificación numérica de los tres casos", "3.5"],
       ["9", "5.2 Entrenamiento", "Registro de épocas de las fases 1 y 2", "4.1"],
       ["10", "5.3 Convergencia", "Curva de la Triplet Loss", "4.1"],
       ["11", "6.1 Umbral y ROC", "τ, exactitud, AUC e histogramas intra/inter", "4.2"],
       ["12", "6.2 Verificación", "Pares con distancia y veredicto; salida de `verificar_identidad`", "4.3"],
       ["13", "6.3 One/Few-shot", "Exactitud para k = 1, 3, 5 y gráfica de barras", "4.4"],
       ["14", "6.4 Demostraciones", "Consultas one-shot y few-shot con veredicto", "4.4"],
       ["15", "6.5 t-SNE", "Agrupamiento de los embeddings antes y después", "4.5"]],
      [0.45, 1.45, 3.4, 1.2], tam=9)

# ---------------- Conclusiones ----------------
salto_pagina()
titulo("Conclusiones", 1)
parrafo("Las redes neuronales siamesas cambian la pregunta que responde la red: en lugar de «¿qué clase es?» "
        "aprenden «¿qué tan parecidos son?». Al proyectar cada rostro en un *embedding* de 128 dimensiones con norma "
        "1, la identidad se convierte en una cuestión de distancia, y registrar a una persona nueva se reduce a "
        "guardar un vector, sin reentrenar la red.")
parrafo(f"La Triplet Loss, implementada explícitamente, produjo ese espacio métrico: sobre identidades que el "
        f"modelo nunca vio, el índice de separabilidad d′ entre distancias genuinas e impostoras pasó de {dec(dprima(VA), 2)} a "
        f"{dec(dprima(VD), 2)}, el AUC subió de {dec(VA['auc'], 3)} a {dec(VD['auc'], 3)} y la exactitud de "
        f"verificación con el umbral elegido en validación fue de {pct(VD['exactitud'], 1)}. El margen $\\alpha$ y la "
        "minería de negativas *semi-hard* fueron necesarios para que el entrenamiento no se estancara con "
        "tripletas fáciles.")
parrafo(f"El experimento one-shot / few-shot confirma la utilidad práctica del enfoque: con una sola foto de "
        f"referencia se identificó correctamente el {pct(SH[1]['identificacion_media'], 1)} de las consultas, y con "
        f"cinco fotos el {pct(SH[k_max]['identificacion_media'], 1)}, porque el prototipo promedio es más robusto a "
        "la pose y la iluminación que una imagen aislada.")
parrafo("Finalmente, el trabajo muestra que en reconocimiento facial el preprocesamiento y los datos pesan tanto "
        "como la arquitectura: sin detección, alineación y normalización de la iluminación la red compararía "
        "fondos y sombras; y al tratarse de datos biométricos, el consentimiento informado, la no publicación de las "
        "fotografías y la conciencia de los sesgos del dataset forman parte del diseño del sistema, no un añadido.")

# ---------------- Referencias (APA 7) ----------------
salto_pagina()
titulo("Referencias", 1)
referencias = [
    "Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., Devin, M., Ghemawat, S., Irving, G., Isard, M., "
    "Kudlur, M., Levenberg, J., Monga, R., Moore, S., Murray, D. G., Steiner, B., Tucker, P., Vasudevan, V., Warden, P., "
    "… Zheng, X. (2016). TensorFlow: A system for large-scale machine learning. En *12th USENIX Symposium on Operating "
    "Systems Design and Implementation (OSDI 16)* (pp. 265–283). USENIX Association.",
    "Bromley, J., Guyon, I., LeCun, Y., Säckinger, E., & Shah, R. (1993). Signature verification using a \"Siamese\" "
    "time delay neural network. En *Advances in Neural Information Processing Systems 6* (pp. 737–744). Morgan "
    "Kaufmann.",
    "Buolamwini, J., & Gebru, T. (2018). Gender shades: Intersectional accuracy disparities in commercial gender "
    "classification. En *Proceedings of the 1st Conference on Fairness, Accountability and Transparency* (Vol. 81, "
    "pp. 77–91). PMLR. https://proceedings.mlr.press/v81/buolamwini18a.html",
    "Chopra, S., Hadsell, R., & LeCun, Y. (2005). Learning a similarity metric discriminatively, with application to "
    "face verification. En *2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition* (Vol. 1, "
    "pp. 539–546). IEEE. https://doi.org/10.1109/CVPR.2005.202",
    "Congreso de la República de Colombia. (2012, 17 de octubre). *Ley Estatutaria 1581 de 2012, por la cual se "
    "dictan disposiciones generales para la protección de datos personales*. Diario Oficial No. 48.587.",
    "Fei-Fei, L., Fergus, R., & Perona, P. (2006). One-shot learning of object categories. *IEEE Transactions on "
    "Pattern Analysis and Machine Intelligence, 28*(4), 594–611. https://doi.org/10.1109/TPAMI.2006.79",
    "Hermans, A., Beyer, L., & Leibe, B. (2017). *In defense of the triplet loss for person re-identification* "
    "(arXiv:1703.07737). arXiv. https://doi.org/10.48550/arXiv.1703.07737",
    "Huang, G. B., Ramesh, M., Berg, T., & Learned-Miller, E. (2007). *Labeled Faces in the Wild: A database for "
    "studying face recognition in unconstrained environments* (Informe técnico 07-49). University of Massachusetts, "
    "Amherst.",
    "Jaiswal, A. (2022, 4 de abril). *Face recognition system using Python*. Analytics Vidhya. "
    "https://www.analyticsvidhya.com/blog/2022/04/face-recognition-system-using-python/",
    "Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. En *3rd International Conference on "
    "Learning Representations (ICLR 2015)*. https://doi.org/10.48550/arXiv.1412.6980",
    "Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese neural networks for one-shot image recognition. En "
    "*ICML 2015 Deep Learning Workshop*.",
    "Lampert, C. H., Nickisch, H., & Harmeling, S. (2009). Learning to detect unseen object classes by between-class "
    "attribute transfer. En *2009 IEEE Conference on Computer Vision and Pattern Recognition* (pp. 951–958). IEEE. "
    "https://doi.org/10.1109/CVPR.2009.5206594",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., "
    "Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. "
    "(2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.",
    "Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., "
    "Clark, J., Krueger, G., & Sutskever, I. (2021). Learning transferable visual models from natural language "
    "supervision. En *Proceedings of the 38th International Conference on Machine Learning* (Vol. 139, pp. "
    "8748–8763). PMLR. https://proceedings.mlr.press/v139/radford21a.html",
    "Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and "
    "linear bottlenecks. En *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 4510–4520). "
    "IEEE. https://doi.org/10.1109/CVPR.2018.00474",
    "Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and "
    "clustering. En *2015 IEEE Conference on Computer Vision and Pattern Recognition* (pp. 815–823). IEEE. "
    "https://doi.org/10.1109/CVPR.2015.7298682",
    "Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical networks for few-shot learning. En *Advances in Neural "
    "Information Processing Systems 30* (pp. 4077–4087). Curran Associates.",
    "Taigman, Y., Yang, M., Ranzato, M., & Wolf, L. (2014). DeepFace: Closing the gap to human-level performance in "
    "face verification. En *2014 IEEE Conference on Computer Vision and Pattern Recognition* (pp. 1701–1708). IEEE. "
    "https://doi.org/10.1109/CVPR.2014.220",
    "van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research, "
    "9*(86), 2579–2605.",
    "Vinyals, O., Blundell, C., Lillicrap, T., Kavukcuoglu, K., & Wierstra, D. (2016). Matching networks for one shot "
    "learning. En *Advances in Neural Information Processing Systems 29* (pp. 3630–3638). Curran Associates.",
    "Viola, P., & Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. En *2001 IEEE "
    "Computer Society Conference on Computer Vision and Pattern Recognition* (Vol. 1, pp. I-511–I-518). IEEE. "
    "https://doi.org/10.1109/CVPR.2001.990517",
    "Wang, M., & Deng, W. (2021). Deep face recognition: A survey. *Neurocomputing, 429*, 215–244. "
    "https://doi.org/10.1016/j.neucom.2020.10.081",
    "*What are Siamese neural networks in deep learning?* (s.f.). Towards Data Science. Recuperado el 30 de octubre "
    "de 2026, de https://towardsdatascience.com/what-are-siamese-neural-networks-in-deep-learning-bb092f749dcb/",
    "Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded "
    "convolutional networks. *IEEE Signal Processing Letters, 23*(10), 1499–1503. "
    "https://doi.org/10.1109/LSP.2016.2603342",
    "Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. En P. S. Heckbert (Ed.), *Graphics "
    "gems IV* (pp. 474–485). Academic Press.",
]
for ref in referencias:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.5)
    p.paragraph_format.first_line_indent = Inches(-0.5)       # sangría francesa
    texto_con_formato(p, ref)

# ---------------- Anexo: código completo ----------------
salto_pagina()
titulo("Anexo A. Código Completo del Notebook de Reconocimiento Facial", 1)
parrafo(f"Celdas de código del notebook `notebooks/{NB1}` (idénticas a `scripts/{S1}`), en el orden de "
        f"ejecución. Las celdas de texto explicativo se omiten porque su contenido se desarrolla en el cuerpo del "
        f"informe. Enlace para abrirlo en Google Colab: {COLAB}", sangria=False)
for k, c in enumerate(celdas_codigo(S1), 1):
    bloque_codigo(f"# ---- Celda {k} ----\n{c}", tam=7.5)

SALIDA.parent.mkdir(exist_ok=True)
doc.save(SALIDA)
print(f"Informe generado: {SALIDA}")
print("Tablas:", contador["tabla"], "| Figuras:", contador["figura"], "| Códigos:", contador["codigo"])
