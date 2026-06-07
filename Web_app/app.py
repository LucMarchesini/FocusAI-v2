"""
Focus Detector — Web App
Mesma lógica do testar_modelo.py, porém servido como aplicação web.

O navegador captura a webcam, envia frames para este backend Flask,
que roda o MESMO modelo focus_model.keras e devolve a probabilidade.
O HUD e o relatório da sessão são desenhados no frontend.

Rodar:
    python app.py
Depois abra http://localhost:5000 no navegador.
"""

import os
# O modelo é .keras (agnóstico de backend). Usamos TensorFlow, que está instalado.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import base64
import numpy as np
import cv2
import keras
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Reutiliza o modelo já treinado, sem duplicar arquivo.
MODELO = os.path.join(BASE_DIR, "..", "Modelo_funcional", "focus_model.keras")
TAM = 224

app = Flask(__name__)

print("Carregando modelo...")
model = keras.models.load_model(MODELO)
# "Aquecimento" para a primeira predição não travar a primeira request.
model.predict(np.zeros((1, TAM, TAM, 3), dtype="float32"), verbose=0)
print("Modelo carregado!")


def _decode_frame(data_url: str) -> np.ndarray:
    """Converte um dataURL (JPEG base64) vindo do navegador em entrada do modelo."""
    header_sep = data_url.find(",")
    raw = base64.b64decode(data_url[header_sep + 1:] if header_sep != -1 else data_url)
    buf = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Frame inválido.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (TAM, TAM))
    return np.expand_dims(resized.astype("float32"), axis=0)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    image = data.get("image")
    if not image:
        return jsonify(error="campo 'image' ausente"), 400
    try:
        entrada = _decode_frame(image)
        prob = float(model.predict(entrada, verbose=0)[0][0])
    except Exception as exc:  # não derruba a sessão por um frame ruim
        return jsonify(error=str(exc)), 400
    return jsonify(prob=prob)


if __name__ == "__main__":
    # threaded=False evita concorrência no grafo do modelo.
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)
