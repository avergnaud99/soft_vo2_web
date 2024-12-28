"""
This module creates the database page of the application.
"""

### Imports ###

from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

### HTML Components ###

bd_page = html.Div(
    [
        dbc.Container(
            [
                # Page Header
                html.H1("Base de données", className="text-center my-4"),
                html.Hr(),
                
                # Tabs for sections (Athletes, Teams, Tests)
                dbc.Tabs(
                    [
                        dbc.Tab(label="Athlètes", tab_id="tab-athletes"),
                        dbc.Tab(label="Équipes", tab_id="tab-teams"),
                        dbc.Tab(label="Tests", tab_id="tab-tests"),
                    ],
                    id="tabs",
                    active_tab="tab-athletes"
                ),
                
                # Card for adding a team (Hidden by default)
                html.Div(
                    id="add-team-div",
                    hidden=True,
                    children=[
                        dbc.Card(
                            dbc.CardBody([
                                html.H3("Ajouter une équipe"),
                                dcc.Input(
                                    id="team-name-input",
                                    type="text",
                                    placeholder="Nom de l'équipe",
                                    style={'width': '20%', "margin-right": '2vh'}
                                ),
                                dbc.Button(
                                    "Ajouter",
                                    id="add-team-button",
                                    color="dark",
                                    style={"maxWidth": "250px"}
                                ),
                            ]),
                            className="mt-4", style={'textAlign': 'center'}
                        )
                    ]
                ),
                
                # Card for exporting follow-up report (Hidden by default)
                html.Div(
                    id="export-followup-bd",
                    hidden=True,
                    children=[
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H3("Exporter le bilan de suivi", className="text-center"),
                                    
                                    # Dropdown and Button for exporting
                                    html.Div(
                                        [
                                            # Dropdown for selecting tests
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id="dropdown-tests-followup",
                                                    multi=True,
                                                    placeholder="Sélectionner les tests...",
                                                ),
                                                width=3,
                                            ),
                                            
                                            # Button for exporting the report as PDF
                                            dbc.Col(
                                                dbc.Button(
                                                    "Télécharger PDF",
                                                    id="export-followup-report",
                                                    color="dark",
                                                    style={"maxWidth": "250px", "width": "auto"}
                                                ),
                                                width="auto",
                                            ),
                                        ],
                                        className="d-flex justify-content-center align-items-center",
                                        style={"gap": "2vh", "align-items": "center"},
                                    ),
                                ]
                            ),
                            className="mt-4"
                        )
                    ]
                ),
                
                # Data table for displaying database records
                dbc.Card(
                    dbc.CardBody([
                        dash_table.DataTable(
                            id="bd-table",
                            page_size=10,
                            page_current=0,
                            filter_action="native",
                            filter_options=dict(placeholder_text="Rechercher..."),
                            sort_action="native",
                            style_table={"overflowX": "auto"},
                            style_cell={"textAlign": "center"},
                            style_data_conditional=[
                                {"if": {"column_id": "Télécharger"}, "cursor": "pointer"},
                                {"if": {"column_id": "Supprimer"}, "cursor": "pointer"}
                            ],
                        ),
                        dcc.Download(id="download-report-bd")
                    ]),
                    className="mt-4", style={"margin-bottom": "5vh"}
                ),
                
                # Modal for team report generation
                dbc.Modal(
                    [
                        dbc.ModalHeader('Hétérogénéité des protocoles'),
                        dbc.ModalBody(id="modal-body-team"),
                        dbc.ModalFooter([
                            dbc.Button("Tests à plat", id="flat-report-team", color="success"),
                            dbc.Button("Tests avec pentes positives", id="slope-report-team", color="success", style={'margin-left': '3vh'})
                        ], className="d-flex justify-content-center align-items-center")
                    ],
                    id="confirmation-modal-team",
                    centered=True,
                    is_open=False
                ),
                
                # Modal for follow-up report generation
                dbc.Modal(
                    [
                        dbc.ModalHeader('Hétérogénéité des protocoles'),
                        dbc.ModalBody(id="modal-body-followup"),
                        dbc.ModalFooter([
                            dbc.Button("Tests à plat", id="flat-report-followup", color="success"),
                            dbc.Button("Tests avec pentes positives", id="slope-report-followup", color="success", style={'margin-left': '3vh'})
                        ], className="d-flex justify-content-center align-items-center")
                    ],
                    id="confirmation-modal-followup",
                    centered=True,
                    is_open=False
                ),
                
                # Modal for confirming deletion of a test
                dbc.Modal(
                    [
                        dbc.ModalHeader("Suppression"),
                        dbc.ModalBody(id="modal-body-delete"),
                        dbc.ModalFooter([
                            dbc.Button("Oui", id="confirm-delete", color="success"),
                            dbc.Button("Non", id="cancel-delete", color="danger", style={'margin-left': '3vh'})
                        ], className="d-flex justify-content-center align-items-center")
                    ],
                    id="confirmation-modal-delete",
                    centered=True,
                    is_open=False
                ),
            ],
            style={"marginTop": "5vh"},
        )
    ]
)