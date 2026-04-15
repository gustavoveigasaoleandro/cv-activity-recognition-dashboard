from dash import Dash
import dash_bootstrap_components as dbc
import pandas as pd
from pages.index import index_layout
app = Dash(__name__, external_stylesheets=[
           dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)


# Callback que renderiza o layout de acordo com o pathname da URL

app.layout = index_layout  # ← sem callback aqui

if __name__ == "__main__":
    app.run(debug=True)
