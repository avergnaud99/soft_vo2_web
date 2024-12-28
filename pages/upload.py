"""
This module creates the upoad page of the application.
"""

### Imports ###

from dash import html, dcc
import dash_bootstrap_components as dbc

### HTML Components ###

upload_page = html.Div([
    html.Div(
        style={
            'display': 'flex',
            'alignItems': 'center',
            'justifyContent': 'center',
            'height': '90vh',
            'backgroundColor': '#f8f9fa',
            'padding': '40px'
        },
        children=[
            html.Div(
                style={'textAlign': 'center', 'width': '100%'},
                children=[

                    # Page Titles
                    html.H1(
                        "Téléchargement des Données",
                        className="display-5 mb-4",
                        style={"color": "#495057"}
                    ),
                    html.P(
                        "Veuillez télécharger les fichiers nécessaires pour l'analyse.",
                        className="lead",
                        style={"color": "#6c757d"}
                    ),
                    html.Hr(style={"width": "50%", "margin": "auto", "marginBottom": "20px"}),

                    # File Upload Section
                    html.Div(
                        style={'display': 'flex', 'gap': '30px', 'justifyContent': 'center'},
                        children=[

                            # VO2 File Upload Button
                            dcc.Upload(
                                id="upload-vo2",
                                children=[
                                    dbc.Button(
                                        children=[
                                            html.I(className="bi bi-upload me-2"),
                                            "Fichier VO2"
                                        ],
                                        id="button-1",
                                        className="btn btn-outline-secondary",
                                        color="secondary",
                                        size="lg"
                                    )
                                ],
                                multiple=False,
                                style={'textAlign': 'center'}
                            ),

                            # Lactate File Upload Button
                            dcc.Upload(
                                id="upload-lactate",
                                children=[
                                    dbc.Button(
                                        children=[
                                            html.I(className="bi bi-upload me-2"),
                                            "Fichier Lactate"
                                        ],
                                        id="button-2",
                                        className="btn btn-outline-secondary",
                                        color="secondary",
                                        size="lg"
                                    )
                                ],
                                multiple=False,
                                style={'textAlign': 'center'}
                            )
                        ]
                    ),
                    html.Hr(style={"width": "50%", "margin": "auto", "marginTop": "20px", "marginBottom": "20px"}),

                    # Submit Button below the upload buttons
                    dbc.Button(
                        "Valider",
                        id="validate-button-upload",
                        color="dark",
                        className="mt-4",
                        style={'width': '100%', 'maxWidth': '300px'}
                    ),

                    # Modal for Athlete Information
                    dbc.Modal(
                        [
                            dbc.ModalHeader("Création d'un nouvel athlète"),
                            dbc.ModalBody(id="modal-body"),
                            dbc.ModalFooter([
                                dbc.Button("Oui", id="confirm-btn-athlete", color="success"),
                                dbc.Button("Non", id="cancel-btn-athlete", color="danger"),
                            ])
                        ],
                        id="confirmation-modal-athlete",
                        centered=True,
                        is_open=False
                    ),
                ]
            )
        ]
    )
])
