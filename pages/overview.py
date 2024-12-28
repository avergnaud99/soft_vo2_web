"""
This module creates the overview of the data page of the application.
"""

### Imports ###

from dash import html
import dash_bootstrap_components as dbc
from dash import dash_table

### HTML Components ###

overview_page = html.Div(
    style={"margin": "5vh", "textAlign": "center"},
    id="tables-div",
    children=[
        # Title of the page
        html.H1("Visualisation des données", className="text-center my-4"),

        # Card for the first DataFrame (VO2)
        dbc.Card([
            dbc.CardHeader(html.H4("Données VO2")),
            dbc.CardBody([
                dash_table.DataTable(
                    id='table-vo2',
                    page_size=10,
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    },
                    style_cell={
                        'textAlign': 'center',
                        'padding': '10px'
                    },
                    style_data={'border': '1px solid grey'},
                    style_table={'overflowX': 'auto'},
                )
            ])
        ], className="mb-4"),

        # Card for the second DataFrame (Lactate)
        dbc.Card([
            dbc.CardHeader(
                dbc.Row([
                        dbc.Col(html.H4("Données Lactates", className="my-auto"), width="auto"),
                        dbc.Col(
                            dbc.Switch(
                                id='switch-lactate',
                                label="Activé",
                                value=True,
                                input_style={
                                    "transform": "scale(1.5)", 
                                    "border-radius": "10px",
                                    "position": "absolute",
                                },
                                label_style={
                                    "margin-left": "10px"
                                }
                            ),
                            width="auto"
                        )
                    ], align="center", justify="between")
            ),
            dbc.CardBody([
                dash_table.DataTable(
                    id='table-lactate',
                    page_size=10,
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    },
                    style_cell={
                        'textAlign': 'center',
                        'padding': '10px'
                    },
                    style_data={'border': '1px solid grey'},
                    style_table={'overflowX': 'auto'},
                )
            ]),
        ]),

        # Validation button
        dbc.Button(
            "Continuer l'analyse", 
            id="validate-button-overview",
            color="dark",
            className="mt-4",
            style={'width': '100%', 'maxWidth': '300px'}
        )
    ]
)