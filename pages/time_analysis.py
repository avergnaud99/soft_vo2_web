"""
This module creates the time analysis page of the application.
"""

### Imports ###

from dash import html, dcc
import dash_bootstrap_components as dbc

### HTML Components ###

time_analysis_page = html.Div(
    style={"display": "flex", "alignItems": "center", "margin": "5vh"},
    id="time-analysis-div",
    children=[

        # Left navigation arrow (left column)
        html.Div([
            dbc.Button("←", id="prev-button-time", color="dark", className="mt-4", style={"fontSize": "20px"}),  # Flèche gauche
        ], style={"display": "inline-block", "verticalAlign": "middle", "width": "5%"}),  # Colonne gauche

        # Main content (time analysis page)
        html.Div([
            dbc.Container(
                [
                    html.H1(id="title_analysis_time", className="text-center my-4"),
                    html.Hr(),

                    # Filters to interact with the graph
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Choix du sélecteur"),
                                    dcc.Dropdown(
                                        id="dropdown-vertical-bars",
                                        options=[
                                            {"label": "Début de test", "value": "start-cut"},
                                            {"label": "Fin de test", "value": "end-cut"},
                                            {"label": "Plateau", "value": "plateau"},
                                            {"label": "VO2 Max", "value": "max"}
                                        ],
                                        value="start-cut",
                                        clearable=False,
                                        searchable=False,
                                        className="mb-3"
                                    ),
                                ],
                                width=True,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Position de la barre"),
                                    html.Div(id="slider-div-time")
                                ],
                                width=True,
                            ),
                            dbc.Col(
                                dbc.Row(
                                    [dbc.Col(
                                        [
                                            html.Label("Temps"),
                                            html.P(html.B(id="time-value"))
                                        ],
                                        width=True,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label("VO2 (L/min)"),
                                            html.P(html.B(id="vo2-value"))
                                        ],
                                        width=True,
                                    )],
                                    className="align-items-stretch"
                                ),
                                width=True
                            )
                        ],
                        className="align-items-stretch"
                    ),

                    html.Hr(),

                    # Dynamic graph
                    dcc.Graph(id="graph-time", style={"height": "70vh"}),
                ]
            )
        ], 
        style={"display": "inline-block", "width": "90%", "textAlign": "center"}),

        # Right navigation arrow (right column)
        html.Div([
            dbc.Button("→", id="next-button-time", color="dark", className="mt-4", style={"fontSize": "20px"}),
        ], 
        style={"display": "inline-block", "verticalAlign": "middle", "width": "5%"}
        )
    ]
)
