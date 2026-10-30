"""Convierte los scripts de /scripts (formato "percent": celdas separadas por '# %%') en notebooks de Colab.

Uso (desde la raíz del repositorio):
    python herramientas/construir_notebooks.py
"""
import json
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
REPO_GITHUB = "DiegoYanzir/Taller-Redes-Neuronales-V"   # ajustar si el repositorio tiene otro nombre
RAMA = "main"


def badge_colab(nombre_nb):
    url = f"https://colab.research.google.com/github/{REPO_GITHUB}/blob/{RAMA}/notebooks/{nombre_nb}"
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


def celda(tipo, lineas):
    # Se eliminan líneas vacías al inicio y al final de la celda
    while lineas and not lineas[0].strip():
        lineas.pop(0)
    while lineas and not lineas[-1].strip():
        lineas.pop()
    fuente = [l + "\n" for l in lineas]
    if fuente:
        fuente[-1] = fuente[-1].rstrip("\n")
    c = {"cell_type": tipo, "metadata": {}, "source": fuente}
    if tipo == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def script_a_notebook(ruta_py, ruta_ipynb):
    texto = ruta_py.read_text(encoding="utf-8")
    bloques = re.split(r"^# %%(.*)$", texto, flags=re.M)
    celdas = []
    # bloques = [preambulo, marcador1, cuerpo1, marcador2, cuerpo2, ...]
    for marcador, cuerpo in zip(bloques[1::2], bloques[2::2]):
        lineas = cuerpo.split("\n")
        if "[markdown]" in marcador:
            lineas = [re.sub(r"^# ?", "", l) for l in lineas]
            celdas.append(celda("markdown", lineas))
        else:
            celdas.append(celda("code", lineas))
    # Badge "Open in Colab" justo después del título
    celdas.insert(1, celda("markdown", [badge_colab(ruta_ipynb.name)]))
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": celdas,
    }
    nb["metadata"]["accelerator"] = "GPU"     # la red siamesa usa un backbone convolucional: se sugiere GPU
    ruta_ipynb.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(celdas)


if __name__ == "__main__":
    destino = RAIZ / "notebooks"
    destino.mkdir(exist_ok=True)
    for py in sorted((RAIZ / "scripts").glob("[0-9][0-9]_*.py")):
        ipynb = destino / (py.stem + ".ipynb")
        n = script_a_notebook(py, ipynb)
        print(f"{py.name} -> notebooks/{ipynb.name} ({n} celdas)")
