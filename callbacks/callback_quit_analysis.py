"""
This module handles the callbacks in order to quit the analysis.

Features :
- Quit the analysis.

"""

### Imports ###

from dash import Input, Output, State, callback, callback_context, no_update
from app import navbar_components_normal
from utils.crud import *
from utils.database import get_db
import logging
import traceback
from utils.pdf_processing import *

### Callbacks ###

# --- SECTION 1 : Callback related to the exit of the analysis ---
@callback(
    Output("url", "pathname", allow_duplicate=True),
    Output('navbar', 'children', allow_duplicate=True),
    Output('alert-app', 'is_open'),
    Output('modal-body-alert-app', "children"),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input('quit-analysis', "n_clicks"),
    Input("confirm-btn-alert-app", "n_clicks"),
    Input("cancel-btn-alert-app", "n_clicks"),
    State("test", "data"),
    prevent_initial_call=True
)
def navigate(quit_btn, confirm_btn, cancel_btn, test_id):
    """
    Callback to handle the navigation logic when the user attempts to quit an analysis, 
    confirm quitting, or cancel the action. This callback controls the alert modal 
    behavior and updates the current analysis state.

    Parameters
    ----------
    quit_btn : int
        The number of times the 'quit-analysis' button has been clicked.
    confirm_btn : int
        The number of times the 'confirm-btn-alert-app' button has been clicked.
    cancel_btn : int
        The number of times the 'cancel-btn-alert-app' button has been clicked.
    test_id : str
        The ID of the test currently in progress.

    Returns
    -------
    tuple
        - `pathname` : str
            The next page path to navigate to ("/upload" or no_update).
        - `navbar` : str
            The updated navbar components.
        - `is_open` : bool
            Whether the alert modal should be open (True or False).
        - `modal_body` : str
            The message to be displayed in the alert modal.
        - Visibility and message of the error toast.
    """
    try:

        # Detect the trigger of the callback
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        # Handle the scenario when the user clicks to quit the analysis
        if trigger == "quit-analysis" and quit_btn:
            message = ("Vous vous apprêtez à quitter l'analyse en cours. Êtes-vous sûr de vouloir continuer ?")
            return no_update, no_update, True, message, False, ""

        # Handle the scenario when the user confirms quitting
        if trigger == "confirm-btn-alert-app" and confirm_btn:

            # Create the database session
            session = next(get_db())

            # Delete the ongoing test from the database
            delete(session, Test, test_id)

            # Clean the temporary folder
            clear_folder("./temp")

            return "/upload", navbar_components_normal, False, "", False, ""

        # Handle the scenario when the user cancels the alert
        if trigger == "cancel-btn-alert-app" and cancel_btn:
            return no_update, no_update, False, "", False, ""

        # No changes if no button is pressed
        return no_update
    
    except Exception as e:
        logging.error(f"Error in navigate : {e}\n{traceback.format_exc()}")
        return (
            no_update, no_update, no_update, no_update,
            True, "Une erreur s'est produite. Consultez les journaux pour plus de détails."
        )

        

            
        
