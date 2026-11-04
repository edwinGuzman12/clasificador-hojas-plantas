import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

TAMANO = (128, 128)   # mismo tamaño usado en el entrenamiento

st.set_page_config(page_title="Diagnóstico de hojas", page_icon="🌿")


@st.cache_resource
def cargar_modelo():
    modelo = tf.keras.models.load_model("modelo/modelo.keras")
    with open("modelo/clases.json", encoding="utf-8") as f:
        clases = json.load(f)
    return modelo, clases


def bonito(c):
    cultivo, estado = c.split("___")
    return f"{cultivo.replace('_', ' ')} — {estado.replace('_', ' ')}"


def preprocesar(imagen):
    # Igual que en el entrenamiento: RGB -> resize 128x128 -> uint8.
    # NO se divide entre 255: el modelo ya trae la capa Rescaling.
    arr = np.array(imagen.convert("RGB"))
    arr = tf.image.resize(arr, TAMANO)
    arr = tf.cast(tf.cast(arr, tf.uint8), tf.float32)
    return tf.expand_dims(arr, 0)


st.title("Diagnóstico de enfermedades en hojas de cultivos")
st.write("Sube o toma una foto de una hoja. El modelo es una CNN propia, "
         "entrenada desde cero, que reconoce 38 clases de cultivo y estado.")

modelo, clases = cargar_modelo()

fuente = st.radio("¿Cómo quieres cargar la foto?",
                  ["Subir archivo", "Usar la cámara"], horizontal=True)
if fuente == "Subir archivo":
    archivo = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])
else:
    archivo = st.camera_input("Toma una foto de la hoja")

if archivo is not None:
    imagen = ImageOps.exif_transpose(Image.open(archivo))   # corrige fotos de celular
    st.image(imagen, caption="Imagen cargada", width=300)

    if st.button("Predecir"):
        salida = modelo(preprocesar(imagen), training=False).numpy()[0]
        top = np.argsort(salida)[::-1][:3]

        st.success(f"Predicción: {bonito(clases[top[0]])}")
        st.metric("Confianza", f"{salida[top[0]] * 100:.1f} %")
        if salida[top[0]] < 0.6:
            st.warning("Confianza baja: toma la foto de una sola hoja, "
                       "con buena luz y fondo liso.")

        st.subheader("Top 3 de clases más probables")
        for i in top:
            st.write(f"{bonito(clases[i])}: {salida[i] * 100:.1f} %")
            st.progress(float(salida[i]))
