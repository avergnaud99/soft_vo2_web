"""
This module creates the login page of the application.
"""

### Imports ###

from dash import html, dcc
import dash_bootstrap_components as dbc

### HTML Components ###

login_page = html.Div(
    [
        dbc.Container(
            [
                # Centered Row for Login Form
                dbc.Row(
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    # Heading
                                    html.H2(
                                        "Connexion",
                                        className="text-center mb-4"
                                    ),
                                    
                                    # Username Input Field
                                    dcc.Input(
                                        id="username",
                                        value="",
                                        type="text",
                                        placeholder="Nom d'utilisateur",
                                        className="form-control mb-3"
                                    ),
                                    
                                    # Password Input Field
                                    dcc.Input(
                                        id="password",
                                        value="",
                                        type="password",
                                        placeholder="Mot de passe",
                                        className="form-control mb-3"
                                    ),
                                    
                                    # Login Button
                                    dbc.Button(
                                        "Se connecter",
                                        id="login-button",
                                        color="primary",
                                        className="w-100 mb-3"
                                    ),
                                    
                                    # Login Message Display (e.g., error messages)
                                    html.Div(
                                        children="",
                                        id="login-message",
                                        className="text-danger text-center mt-3"
                                    ),
                                ]
                            ),
                            className="shadow-lg p-4",
                            style={"maxWidth": "400px", "margin": "auto"}
                        ),
                        width=12
                    )
                )
            ],
            className="vh-100 d-flex align-items-center justify-content-center",
            fluid=True
        )
    ]
)