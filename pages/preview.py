"""
This module creates the preview of the modelizations page of the application.
"""

### Imports ###

from dash import html
import dash_bootstrap_components as dbc

### HTML Components ###

preview_page = html.Div(
    id="div-preview",
    style={"display": "flex", "alignItems": "center", "margin": "5vh"},
    children=[

        # Back button
        html.Div(
            style={
                "display": "inline-block",
                "verticalAlign": "middle",
                "width": "5%",
                "position": "fixed",
                "top": "50%",
                "left": "2%"
            },
            children=[
                dbc.Button(
                    "←", 
                    id="prev-button-preview", 
                    color="dark", 
                    className="mt-4", 
                    style={"fontSize": "20px"}
                ),
            ], 
        ),

        # Graphical part
        html.Div([
            dbc.Container([

                    # Title
                    html.H1(
                        "Prévisualisation des modélisations", 
                        className="text-center my-4"
                    ),
                    html.Hr(),

                    # Modeling graphs
                    html.Div(
                        id="preview-graphs",
                        style={"padding": "10px", "gap": "10px"}
                    )
                ]
            )
        ], 
        style={"display": "inline-block", "width": "90%", "textAlign": "center", "margin-left": "5%", "margin-right": "5%"}),

        # Next button
        html.Div(
            style={
                "display": "inline-block",
                "verticalAlign": "middle",
                "width": "5%",
                "position": "fixed",
                "top": "50%",
                "right": "2%"
            },
            children=[
                dbc.Button(
                    "→", 
                    id="next-button-preview", 
                    color="dark", 
                    className="mt-4", 
                    style={"fontSize": "20px"}
                ),
            ], 
        ),
    ], 
)