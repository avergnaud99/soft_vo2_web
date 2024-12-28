"""
This module creates the modelization analysis pages of the application.
"""

### Imports ###

from dash import html, dcc
import dash_bootstrap_components as dbc

### Methods ###

def create_analysis_page(type, modelizations):
    """
    Creates the layout for the analysis page.

    Parameters
    ----------
    type : str
        The type of analysis (used to uniquely identify elements).
    modelizations : list of str
        List of modelizations for the dropdown options.

    Returns
    -------
    html.Div
        A Dash Div element containing the analysis page layout.
    """

    return html.Div(
        id=f"div-{type}",
        style={"display": "flex", "alignItems": "center", "margin": "5vh"},
        children=[

            # Back Button
            html.Div(
                style={"display": "inline-block", "verticalAlign": "middle", "width": "5%"},
                children=[
                    dbc.Button(
                        "←", 
                        id=f"prev-button-{type}", 
                        color="dark", 
                        className="mt-4", 
                        style={"fontSize": "20px"}
                    ),
                ], 
            ),

            # Graph Section
            html.Div([
                dbc.Container([

                        # Title
                        html.H1(
                            id=f"title_analysis_{type}", 
                            className="text-center my-4"
                        ),
                        html.Hr(),

                        # Interactions with the graph
                        dbc.Row([

                                # Left-side protection slider
                                dbc.Col(
                                    [
                                        html.Label("Protection des valeurs de début"),
                                        html.Div(id=f"slider-div-protect-left-{type}")
                                    ],
                                    width=True,
                                ),

                                # Right-side protection slider
                                dbc.Col(
                                    [
                                        html.Label("Protection des valeurs de fin"),
                                        html.Div(id=f"slider-div-protect-right-{type}")
                                    ],
                                    width=True,
                                ),

                                # Outlier selection slider
                                dbc.Col(
                                    [
                                        html.Label("Facteur sur l'écart-type"),
                                        html.Div(id=f"slider-div-outlier-{type}")
                                    ],
                                    width=True,
                                ),

                                # Dropdown for selecting modeling type
                                dbc.Col(
                                    [
                                        html.Label("Type de modélisation"),
                                        dcc.Dropdown(
                                            id=f"dropdown-model-{type}",
                                            options=[
                                                {"label": modelizations[0], "value": f"first-model-{type}"},
                                                {"label": modelizations[1], "value": f"second-model-{type}"},
                                            ],
                                            value=f"first-model-{type}",
                                            searchable=False,
                                            clearable=False,
                                            className="mb-3"
                                        ),
                                    ],
                                    width=True,
                                )
                            ],
                            className="align-items-stretch"
                        ),

                        html.Hr(),

                        # Dynamic Graph
                        dcc.Graph(id=f"graph-{type}", style={"height": "70vh"}),
                    ]
                )
            ], 
            style={"display": "inline-block", "width": "90%", "textAlign": "center"}),

            # Forward Button
            html.Div(
                style={"display": "inline-block", "verticalAlign": "middle", "width": "5%"},
                children=[
                    dbc.Button(
                        "→", 
                        id=f"next-button-{type}", 
                        color="dark", 
                        className="mt-4", 
                        style={"fontSize": "20px"}
                    ),
                ], 
            ),
        ], 
    )

### HTML Components ###

cubic_analysis_page = create_analysis_page("cubic", ["1 segment cubique", "2 segments cubiques"])
first_linear_analysis_page = create_analysis_page("first_linear", ["2 segments linéaires", "3 segments linéaires"])
second_linear_analysis_page = create_analysis_page("second_linear", ["2 segments linéaires", "3 segments linéaires"])
