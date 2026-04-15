from dash import dcc, html, Output, Input, callback
import dash_bootstrap_components as dbc
import base64
import io
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "best_by_f1.keras"
TRAIN_CSV = PROJECT_ROOT / "Human_Action_Recognition" / "Training_set.csv"


@lru_cache(maxsize=1)
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@lru_cache(maxsize=1)
def get_class_names():
    if TRAIN_CSV.exists():
        train_df = pd.read_csv(TRAIN_CSV)
        return sorted(train_df["label"].unique().tolist())

    output_dim = load_model().output_shape[-1]
    return [f"class_{idx}" for idx in range(output_dim)]


def preprocess_image(contents):
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    img = Image.open(io.BytesIO(decoded)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)


UploadForm = dbc.Col([
    dcc.Upload(
        id='upload-image',
        children=html.Div(['Clique ou arraste uma imagem para classificar a ação humana']),
        className='upload-area',
        accept='image/*',
        multiple=False
    ),
    html.Div(
        id='output-result',
        className='upload-result',
        style={"marginTop": "15px"}
    )
])


@callback(
    Output('output-result', 'children'),
    Input('upload-image', 'contents'),
    prevent_initial_call=True
)
def classify_image(contents):
    if contents is None:
        return "Nenhuma imagem enviada."

    image = preprocess_image(contents)
    model = load_model()
    class_names = get_class_names()

    probs = model.predict(image, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:3]

    details = [
        html.Li(f"{class_names[idx]}: {probs[idx]:.2%}")
        for idx in top_indices
    ]

    return html.Div([
        html.H5(f"Classe mais provável: {class_names[top_indices[0]]}"),
        html.P("Probabilidades das top 3 ações:"),
        html.Ul(details),
        html.Img(src=contents, style={'width': '300px', 'marginTop': '15px'})
    ])
