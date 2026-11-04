# 🌿 Clasificador de Enfermedades en Hojas de Cultivos

Proyecto de Deep Learning (CNN) para el diagnóstico y clasificación automatizada de 38 clases de cultivos y enfermedades a partir de imágenes de hojas. Desarrollado bajo la metodología **CRISP-DM** en la **Institución Universitaria Colegio Mayor del Cauca**.

---

## 👥 Integrantes
* Juan Diego Chantre Cerón
* Edwin Jair Guzman Diaz
* Bryan Santiago Castro Rojas
* Maicol Stiven Sanchez
* Felipe Alejandro Navarro
* Jesus Manuel Caicedo

**Profesor:** Carlos Fernando Realpe Garcés | **Asignatura:** Inteligencia Artificial

---

## 🚀 Aspectos Clave del Proyecto

* **Dataset & Fuga de Datos:** Se corrigió un problema de *data leakage* en el dataset original (*New Plant Diseases Dataset*) aplicando `GroupShuffleSplit` por foto original (70% Train, 15% Val, 15% Test).
* **Modelo CNN (desde cero):** Arquitectura propia de 4 bloques convolucionales con 427,654 parámetros (~5.2 MB), diseñada para ejecutar eficientemente en CPU.
* **Métricas en Test:** 
  * **Accuracy:** 88.44%
  * **F1-Score Macro:** 0.885
* **Despliegue:** Interfaz web interactiva construida con **Streamlit** que permite la predicción en tiempo real mediante subida de imágenes o uso de la cámara.

---

## 🛠️ Ejecución Local

1. **Clonar el repositorio:**
   git clone https://github.com/TU-USUARIO/clasificador-hojas-plantas.git
   cd clasificador-hojas-plantas

2. **Instalar dependencias:**
   pip install -r requirements.txt

3. **Ejecutar la app:**
   streamlit run app.py
