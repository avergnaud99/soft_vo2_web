'''
This module creates the Dash application, layout, and navigation bar.
'''

### Imports ###
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from flask import Flask
from flask import send_from_directory
import logging
import os

### Methods ###

# Function to create the navigation bar
def create_navbar(items):
    return dbc.NavbarSimple(
        id="navbar",
        children=items,
        brand="VO2 Software",
        className="navbar-expand-lg bg-dark",
        dark=True,
    )

### Main ###

# Initialize Flask server
server = Flask(__name__)
server.secret_key = "votre_cle_secrete"  # Clé secrète pour les sessions

# Create the Dash application
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.LUX, "https://cdn.jsdelivr.net/npm/bootstrap-icons/font/bootstrap-icons.css"])

# Define the folder where images are stored
TEMP_FOLDER = os.path.join(os.getcwd(), 'temp')

# Flask route to serve image files from the 'temp/' folder
@app.server.route('/temp/<path:filename>')
def serve_image(filename):
    return send_from_directory(TEMP_FOLDER, filename)

# Define the main layout of the application
app.layout = html.Div([

    # Manages the URLs for routing
    dcc.Location(id="url", refresh=True),   

    # Session data storage components
    dcc.Store(id="login-status", data=False, storage_type="session"),   # Stockage de l'état de connexion
    dcc.Store(id="login-data", storage_type="session"),   # Stockage des informations de l'utilisateur
    dcc.Store(id="data_athlete", storage_type="session"), # Stockage des données de l'athlète
    dcc.Store(id="test", storage_type="session"), # Sauvegarde du test en cours d'analyse
    dcc.Store(id="use_lactate", storage_type="session"), # Utilisation des lactates
    dcc.Store(id="back", storage_type="session"), # Sauvegarde du retour en arrière
    dcc.Store(id="initial_df_first_modelization", storage_type="session"), # Sauvegarde du dataframe avant la première modélisation 
    dcc.Store(id="initial_df_second_modelization", storage_type="session"), # Sauvegarde du dataframe avant la seconde modélisation
    dcc.Store(id="initial_df_third_modelization", storage_type="session"), # Sauvegarde du dataframe avant la troisième modélisation
    dcc.Store(id="initial_df_preview", storage_type="session"), # Sauvegarde du dataframe avant les résultats
    dcc.Store(id="curves", storage_type="session"), # Sauvegarde des courbes de modélisations
    dcc.Store(id="results", storage_type="session"), # Sauvegarde des résultats des seuils
    dcc.Store(id="final_models", storage_type="session"), # Stockage des modèles finaux
    dcc.Store(id="report-results", storage_type="session"), # Stockage des données du rapport
    dcc.Store(id="threshold-levels", storage_type="session"), # Sauvegarde des paliers associés aux seuils

    # Temporary data storage
    dcc.Store(id="data_vo2", storage_type="memory"),   # Stockage temporaire des données VO2
    dcc.Store(id="data_lactate", storage_type="memory"),   # Stockage temporaire des données lactates
    dcc.Store(id="data_outliers", storage_type="memory"), # Stockage temporaire des outliers
    dcc.Store(id="pos_start_cut", storage_type="memory"), # Stockage temporaire de la position de la barre de suppression du début de test
    dcc.Store(id="pos_end_cut", storage_type="memory"), # Stockage temporaire de la position de la barre de suppression de fin de test
    dcc.Store(id="pos_start_plateau", storage_type="memory"), # Stockage temporaire de la position de la barre de début du plateau
    dcc.Store(id="pos_vo2max", storage_type="memory"), # Stockage temporaire de la position de la barre de VO2max
    dcc.Store(id="n_outliers", storage_type="memory"), # Stockage temporaire du nombre d'outliers
    dcc.Store(id="error-rer", storage_type="memory"), # Stockage temporaire de l'erreur RER
    dcc.Store(id="pdf_report", storage_type="memory"), # Stockage temporaire du string LATEX du rapport PDF de l'analyse
    dcc.Store(id="remarks", storage_type="memory"), # Stockage temporaire des remarques
    dcc.Store(id="slope-flat-tests-team", storage_type="memory"), # Stockage temporaire des tests avec pentes et à plat pour les bilans de team
    dcc.Store(id="slope-flat-tests-followup", storage_type="memory"), # Stockage temporaire des tests avec pentes et à plat pour les bilans de suivi

    # Display the navigation bar
    html.Div(id="navbar-container", style={'height': '10vh'}),

    # Display the page content
    html.Div(id="page-content"), 

    # Toast notification for error messages
    dbc.Toast(
        id="error-toast-app",
        header="Erreur",
        icon="danger",
        is_open=False,
        dismissable=True,
        style={
            "position": "fixed",
            "top": "11%",
            "right": "1%"
        }
    ), 

    # Modal for alerts
    dbc.Modal(
        [
            dbc.ModalHeader("Attention"),
            dbc.ModalBody(id="modal-body-alert-app"),
            dbc.ModalFooter([
                dbc.Button("Oui", id="confirm-btn-alert-app", color="success"),
                dbc.Button("Non", id="cancel-btn-alert-app", color="danger"),
            ])
        ],
        id="alert-app",
        centered=True,
        is_open=False
    )
])

# Normal navigation bar components (for general pages)
navbar_components_normal = [
    dbc.NavItem(dcc.Link(children="Accueil", href="/", className="nav-link")),
    dbc.NavItem(dcc.Link(children="Analyse Test", href="/upload", className="nav-link")),
    dbc.NavItem(dcc.Link(children="BD", href="/bd", className="nav-link")),
    dbc.Button("Se déconnecter", id="logout-button", color="secondary")
]

# Navigation bar components for the analysis page
navbar_components_analysis = [
    dbc.Button("Quitter l'analyse", id="quit-analysis", color="danger")
]

# Configure logging for the application
logging.basicConfig(
    filename="app.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)