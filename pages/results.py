"""
This module creates the results page of the application.
"""

### Imports ###
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash import dash_table

### HTML Components ###
results_page = html.Div(
    id="div-results",
    style={"display": "flex"},
    children=[

        # Back button
        html.Div(
            style={
                "display": "inline-block",
                "verticalAlign": "middle",
                "width": "5%",
                "position": "fixed",
                "top": "50%",
                "left": "1.5%"
            },
            children=[
                dbc.Button(
                    "←", 
                    id="prev-button-results", 
                    color="dark", 
                    className="mt-4", 
                    style={"fontSize": "20px"}
                ),
            ], 
        ),

        # Main container for results
        dbc.Container(
            style={"textAlign": "center", "margin-top": "5vh", "display": "inline-block"},
            children=[

                # Title Section
                html.Div(
                    [html.H1(
                        "Résultats de l'analyse", 
                        className="text-center my-4"
                    ),
                    dbc.Tabs(
                        [
                            dbc.Tab(label="Analyse", tab_id="tab-analyse"),
                            dbc.Tab(label="Rapport", tab_id="tab-rapport"),
                        ],
                        id="tabs-analysis",
                        active_tab="tab-analyse",
                    )]
                ),

                # Progress bar for PDF generation
                html.Div(
                    id="div-progress-pdf",
                    hidden=True,
                    children=[
                        html.Progress(id="progress-pdf")
                    ]
                ),

                # PDF preview section
                html.Div(
                    id="div-pdf-preview",
                    hidden=True,
                    children=[

                        # PDF viewer with remarks
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([

                                    # Left column: PDF display
                                    dbc.Col([
                                        html.Div(
                                            html.Iframe(
                                                id="pdf-viewer",
                                                style={
                                                    "width": "100%",
                                                    "height": "600px",
                                                    "border": "none",
                                                    "overflow": "hidden"
                                                }
                                            )
                                        )
                                    ], width=9),  # La majorité de l'espace est allouée au PDF
                                    
                                    # Right column: Remarks
                                    dbc.Col([
                                        html.Div(
                                            className="d-flex flex-column align-items-center",
                                            children=[

                                                # Remarks modification section
                                                dbc.Card([
                                                    dbc.CardHeader(html.H4("Modifier les remarques"), className="bg-dark text-white"),
                                                    dbc.CardBody([
                                                        dcc.Textarea(
                                                            id="remark-input",
                                                            style={"width": "100%", "height": "80%", "resize": "none"},
                                                            placeholder="Saisissez vos remarques ici..."
                                                        ),
                                                        dbc.Button(
                                                            "Valider",
                                                            id="remark-submit",
                                                            color="dark",
                                                            className="mt-4",
                                                            style={"width": "60%"}
                                                        )
                                                    ])
                                                ], style={"width": "100%", "height": "100%"})
                                            ]
                                        )      
                                    ], width=3, className="d-flex align-items-stretch")
                                ])
                            ])
                        ], style={"margin-top": "5vh"}),

                        # Export buttons (PDF and Excel)
                        html.Div(
                            style={'margin-bottom': '5vh', 'display': 'flex', 'justify-content': 'center'},
                            children=[

                                # PDF export button
                                dbc.Button(
                                    "Exporter PDF",
                                    id="export-pdf-button",
                                    color="dark",
                                    className="mt-4",
                                    style={'maxWidth': '250px', 'display': 'inline-block', 'margin-right': '5vh'}
                                ),

                                # Excel export button
                                dcc.Upload(
                                    id="upload-excel-file",
                                    children=[
                                        dbc.Button(
                                            "Exporter Excel",
                                            id="export-excel-button",
                                            color="dark",
                                            className="mt-4",
                                            style={'maxWidth': '250px', 'display': 'inline-block'}
                                        )
                                    ],
                                    multiple=False,
                                    style={'textAlign': 'center', 'display': 'inline-block', 'margin-right': '5vh'}
                                ),

                                # Save and exit button
                                dbc.Button(
                                    "Quitter et sauvegarder",
                                    id="quit-save-button",
                                    color="dark",
                                    className="mt-4",
                                    style={'maxWidth': '250px', 'display': 'inline-block'}
                                )
                            ]
                        ),

                        # Download components for Excel and PDF
                        dcc.Download(id="download_excel"),
                        dcc.Download(id="download_pdf")
                    ]
                ),

                # Results analysis section
                html.Div(
                    id="div-results-analysis",
                    hidden=False,

                    # Header Section
                    children=[
                        dbc.Card([
                            dbc.CardBody([
                                html.H3("Informations de l'Athlète", className="card-title"),
                                dbc.Row([
                                    dbc.Col([
                                        html.P(html.B("Nom : "), className="card-text"),
                                        html.P(html.B("Date du Test : "), className="card-text"),
                                        html.P(html.B("Taille : "), className="card-text"),
                                        html.P(html.B("VO2 Max : "), className="card-text")
                                    ], md=3),
                                    dbc.Col([
                                        html.P([html.Span("", id="athlete-name")], className="card-text"),
                                        html.P([html.Span("", id="test-date")], className="card-text"),
                                        html.P([html.Span("", id="athlete-height"), " cm"], className="card-text"),
                                        html.P([html.Span("", id="results-vo2-max"), " L/min"], className="card-text")
                                    ], md=3),
                                    dbc.Col([
                                        html.P(html.B("Prénom : "), className="card-text"),
                                        html.P(html.B("Date de Naissance : "), className="card-text"),
                                        html.P(html.B("Poids : "), className="card-text"),
                                        html.P(html.B("FC Max : "), className="card-text")
                                    ], md=3),
                                    dbc.Col([
                                        html.P([html.Span("", id="athlete-firstname")], className="card-text"),
                                        html.P([html.Span("", id="athlete-birth-date")], className="card-text"),
                                        html.P([html.Span("", id="athlete-weight"), " kg"], className="card-text"),
                                        html.P([html.Span("", id="results-fc-max"), " bpm"], className="card-text")
                                    ], md=3),
                                ])
                            ])
                        ], className="card border-light mb-3"),
                        html.Hr(),

                        # Remarks Section
                        dbc.Card([
                            dbc.CardBody([
                                html.H3("Remarques et Paliers", className="card-title"),
                                dbc.Row([
                                    dbc.Col([
                                        dash_table.DataTable(
                                            id='remarks-table',
                                            columns=[{"id": name, "name": name} for name in ["Paliers", "Remarques"]],
                                            style_header={
                                                'backgroundColor': 'rgb(230, 230, 230)',
                                                'fontWeight': 'bold'
                                            },
                                            style_cell={
                                                'textAlign': 'center',
                                                'padding': '10px'
                                            },
                                            style_data={'border': '1px solid lightgray'},
                                            style_table={'overflowX': 'auto'},
                                        )
                                    ], md=6),
                                    dbc.Col([
                                        html.H6("Temps maintenu au dernier palier (secondes)"),
                                        dbc.Input(id="tps-last", value=60, type="number", placeholder="Temps (s)", max=60, min=0, required=True, inputmode="numeric", style={"textAlign": "center"})
                                    ], md=6, align="center")
                                ])
                                
                            ])
                        ], className="card border-light mb-3"),
                        html.Hr(),

                        # Threshold Results Section
                        dbc.Card([
                            dbc.CardBody([
                                html.H3("Résultats des Seuils", className="card-title"),
                                dbc.Row([
                                    dbc.Col([
                                        html.H4("Seuil 1"),
                                        html.P(html.I(id="used-curves-s1")),
                                        dbc.Progress(
                                            id="seuil1-vo2-percent",
                                            min=0, 
                                            max=100, 
                                            striped=True,
                                            animated=True,
                                            color="gray",
                                            style={
                                                "border-radius": "20px",
                                                "height": "30%"
                                            }
                                        ),
                                        html.P(html.B(["VO2 : ", html.Span("", id="seuil1-vo2"), " L/min"])),
                                        dbc.Progress(
                                            id="seuil1-fc-percent",
                                            min=0, 
                                            max=100, 
                                            striped=True,
                                            animated=True,
                                            color="gray",
                                            style={
                                                "border-radius": "20px",
                                                "height": "30%"
                                            }
                                        ),
                                        html.P(html.B(["FC : ", html.Span("", id="seuil1-fc"), " bpm"])),
                                    ], md=6),
                                    dbc.Col([
                                        html.H4("Seuil 2"),
                                        html.P(html.I(id="used-curves-s2")),
                                        dbc.Progress(
                                            id="seuil2-vo2-percent",
                                            min=0, 
                                            max=100, 
                                            striped=True,
                                            animated=True,
                                            color="gray",
                                            style={
                                                "border-radius": "20px",
                                                "height": "30%"
                                            }
                                        ),
                                        html.P(html.B(["VO2 : ", html.Span("", id="seuil2-vo2"), " L/min"])),
                                        dbc.Progress(
                                            id="seuil2-fc-percent",
                                            min=0, 
                                            max=100, 
                                            striped=True,
                                            animated=True,
                                            color="gray",
                                            style={
                                                "border-radius": "20px",
                                                "height": "30%"
                                            }
                                        ),
                                        html.P(html.B(["FC : ", html.Span("", id="seuil2-fc"), " bpm"])),
                                    ], md=6),
                                ], style={"margin-bottom": '10vh'})
                            ])
                        ], className="card border-light mb-3"),
                        html.Hr(),

                        # Graph Section
                        dbc.Card([
                            dbc.CardBody([
                                html.Div(
                                    style={
                                        'position': 'sticky',
                                        'top': '0px',
                                        'background-color': "white",
                                        "zIndex": 1000,
                                        "padding-top": "2vh"
                                    },
                                    children=[
                                        html.H3("Analyse des Résultats", className="card-title"),
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Label("Position du SV1"),
                                                html.Div(id="div-slider-sv1")
                                            ], md=3),
                                            dbc.Col([
                                                dbc.Label("Position du SV2"),
                                                html.Div(id="div-slider-sv2")
                                            ], md=3),
                                            dbc.Col([
                                                dbc.Label("Position du SL1"),
                                                html.Div(id="div-slider-sl1")
                                            ], md=3),
                                            dbc.Col([
                                                dbc.Label("Position du SL2"),
                                                html.Div(id="div-slider-sl2")
                                            ], md=3),
                                        ]),
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Checkbox(
                                                    id="check-sv1",
                                                    value=True,
                                                    input_style={
                                                        "position": "absolute",
                                                    },
                                                    label_style={
                                                        "margin-left": "10px"
                                                    }
                                                )
                                            ], md=3),
                                            dbc.Col([
                                                dbc.Checkbox(
                                                    id="check-sv2",
                                                    value=True,
                                                    input_style={
                                                        "position": "absolute",
                                                    },
                                                    label_style={
                                                        "margin-left": "10px"
                                                    }
                                                )
                                            ], md=3),
                                            dbc.Col([
                                                dbc.Checkbox(
                                                    id="check-sl1",
                                                    value=True,
                                                    input_style={
                                                        "position": "absolute",
                                                    },
                                                    label_style={
                                                        "margin-left": "10px"
                                                    }
                                                )
                                            ], md=3),
                                            dbc.Col([
                                                dbc.Checkbox(
                                                    id="check-sl2",
                                                    value=True,
                                                    input_style={
                                                        "position": "absolute",
                                                    },
                                                    label_style={
                                                        "margin-left": "10px"
                                                    }
                                                )
                                            ], md=3),
                                        ]),
                                        dbc.Row(
                                            dbc.Col(
                                                dbc.Button(
                                                    "Calculer", 
                                                    id="checkbox-button",
                                                    color="dark",
                                                    className="mt-4",
                                                    style={'width': '100%', 'maxWidth': '200px', 'border-radius': "20px"}
                                                ), 
                                                md=12
                                            )
                                        ),
                                        html.Hr()
                                    ]
                                ),
                                dcc.Graph(id="graph-results")
                            ])
                        ], className="card border-light mb-3")
                    ]
                )
            ]
        )
    ]
)

