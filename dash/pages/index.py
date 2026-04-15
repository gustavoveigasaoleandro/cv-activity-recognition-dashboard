from dash import html

import dash_bootstrap_components as dbc
from components.uploadForm import UploadForm

index_layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Classificação de Ações Humanas"),
                html.P(
                    "Envie uma imagem para identificar a ação mais provável e conferir as três principais previsões do modelo."
                ),
            ], width=12, className="mb-4"),
        ]),
        dbc.Row([
            UploadForm,
        ])
    ], className="responsive-container"),
])
