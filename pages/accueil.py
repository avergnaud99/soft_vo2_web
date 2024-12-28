"""
This module creates the welcome page of the application.
"""

### Imports ###

from dash import html
import dash_bootstrap_components as dbc
from utils.database import *

### HTML Components ###

# Main container for the welcome page
accueil_page = html.Div(
    [   # Main content of the welcome page
        dbc.Container(
            [
                html.H1("Accueil", className="text-center mt-4"),
                html.Hr(),
                html.Div(
                    id="welcome-message",
                    className="text-center my-4",
                    style={"fontSize": "1.5rem"},
                ),
            ],
            style={"marginTop": "5vh"},
        ),
    ]
)