"""
This module handles the callbacks related to the upload part of the analysis.

Features :
- Upload the files.
- Treat the data.

"""

### Imports ###
from dash import Input, Output, State, callback, callback_context, no_update
import json
import pandas as pd
from io import StringIO
from utils.crud import *
from utils.database import get_db
from utils.data_processing import *
from utils.database_updating import *

### Callbacks ###

# --- SECTION 1 : Callback related to the upload of the files ---
@callback(
    Output("data_vo2", "data", allow_duplicate=True),
    Output("data_lactate", "data", allow_duplicate=True),
    Output("data_athlete", "data"),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Output("confirmation-modal-athlete", "is_open", allow_duplicate=True),
    Output("modal-body", "children"),
    Input("validate-button-upload", "n_clicks"),
    State("upload-vo2", "contents"),
    State("upload-lactate", 'contents'),
    prevent_initial_call=True
)
def upload_files(click_validate, contents_vo2, contents_lactate) :
    """
    Callback to handle the file upload process for VO2 and lactate data. 
    This callback validates the uploaded files, processes them, and updates the 
    application state accordingly. If errors occur during file processing, 
    they are reported via a modal alert.

    Parameters
    ----------
    click_validate : int
        The number of times the 'validate-button-upload' button has been clicked.
    contents_vo2 : str
        Base64-encoded content of the VO2 data file uploaded by the user.
    contents_lactate : str
        Base64-encoded content of the lactate data file uploaded by the user.

    Returns
    -------
    tuple
        - `data_vo2` : str
            The JSON-encoded records for the VO2 data.
        - `data_lactate` : str
            The JSON-encoded records for the lactate data.
        - `data_athlete` : str
            The JSON-encoded data related to the athlete.
        - `error_is_open` : bool
            Whether the error modal should be open (True if an error occurs).
        - `error_message` : str
            The error message to be displayed if there is an issue with the upload.
        - `confirmation_is_open` : bool
            Whether the athlete confirmation modal should be shown.
        - `modal_body` : str
            The message to be displayed in the athlete confirmation modal.
    """
    try:

        # Detect the trigger for the callback
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        # Load the files when the user clicks the button
        if trigger == "validate-button-upload" and click_validate:
            
            # Missing file(s)
            if not contents_vo2 or not contents_lactate:
                return "", "", "", True, "Un fichier est manquant. Veuillez télécharger les deux fichiers (VO2 et lactate).", False, ""

            # Load data
            df_vo2, error_message_vo2 = upload_vo2(contents_vo2)
            df_lactate, error_message_lactate, data_athlete = upload_lactate(contents_lactate)

            # Error during VO2 data loading
            if error_message_vo2:
                return "", "", "", True, error_message_vo2, False, ""
            
            # Error during lactate data loading
            if error_message_lactate:
                return "", "", "", True, error_message_lactate, False, ""

            return (
                df_vo2.to_json(orient="records"), 
                df_lactate.to_json(orient="records"), 
                data_athlete, 
                False, 
                "", 
                True, 
                data_athlete["text"]
            )
        
        return "", "", "", False, "", False, ""
    
    except Exception as e:
        logging.error(f"Error in upload_files: {e}\n{traceback.format_exc()}")
        return "", "", "", True, "Une erreur s'est produite. Consultez les journaux pour plus de détails.", False, ""

# --- SECTION 2 : Callback for data treatment ---
@callback(
    Output("url", "pathname", allow_duplicate=True),
    Output("test", "data", allow_duplicate=True),
    Output("data_lactate", "data", allow_duplicate=True),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Output("confirmation-modal-athlete", "is_open", allow_duplicate=True),
    Input("confirm-btn-athlete", "n_clicks"),
    Input("cancel-btn-athlete", "n_clicks"),
    State("data_vo2", 'data'),
    State("data_lactate", 'data'),
    State("data_athlete", "data"),
    State("url", "pathname"),
    prevent_initial_call=True
)
def treat_athlete(click_yes_athlete, click_no_athlete, temp_df_vo2, temp_df_lactate, athlete_data, current_href):
    """
    Callback function to handle the athlete profile confirmation and processing of VO2 and lactate data.
    It processes data based on user input (confirm or cancel) and either creates or updates the athlete's profile
    in the database. If an error occurs during the processing, an error message is shown.

    Parameters
    ----------
    click_yes_athlete : int
        The number of times the 'confirm-btn-athlete' button has been clicked (Yes button).
    click_no_athlete : int
        The number of times the 'cancel-btn-athlete' button has been clicked (No button).
    temp_df_vo2 : str
        The JSON-encoded VO2 data passed from the front end.
    temp_df_lactate : str
        The JSON-encoded lactate data passed from the front end.
    athlete_data : dict
        The data of the athlete including mode (new or update), athlete info, and test details.
    current_href : str
        The current URL path of the application.

    Returns
    -------
    tuple
        - `pathname` : str
            The URL to redirect the user after the action is completed.
        - `test_data` : str or None
            The test data in JSON format or None if there is no test data.
        - `data_lactate` : str
            The lactate data in JSON format to be used in the next step.
        - `error_is_open` : bool
            Whether the error modal should be displayed.
        - `error_message` : str
            The error message to display in case of issues during processing.
        - `confirmation_is_open` : bool
            Whether the confirmation modal should be opened.
    """
    try:

        # Detect the trigger for the callback
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        # No action if the user is already on the 'overview' page
        if current_href == '/overview':
            return no_update

        # If the user clicks 'Yes' to confirm the athlete data
        if trigger == "confirm-btn-athlete" and click_yes_athlete:

            # Load VO2 data
            df_vo2 = pd.read_json(StringIO(temp_df_vo2))

            # Process lactate data
            df_lactate = treat_lactate(temp_df_lactate)
            
            # Create new athlete profile if needed
            if athlete_data["mode"] == "new":
                id_athlete = save_athlete_profile(athlete_data)
                
            # Update existing athlete profile if needed
            if athlete_data["mode"] == "update":
                id_athlete = update_athlete_profile(athlete_data)
                
            # Create a new test for the athlete
            session = next(get_db())
            test = create(session, Test, {
                "athlete_id": id_athlete,
                "date": pd.to_datetime(athlete_data["date"]).date(),
                "weight": get_by_id(session, Athlete, id_athlete).weight,
                "source_excel": df_lactate.to_json(orient="records"),
                "source_vo2": df_vo2.to_json(orient="records")
            })

            return '/overview', test.id, df_lactate.to_json(orient="records"), False, "", False
        
        # If the user clicks 'No' to cancel, return to the current page with no action
        if trigger == "cancel-btn-athlete" and click_no_athlete:
            return current_href, None, no_update, False, "", False
        
        return no_update
    
    except Exception as e:
        logging.error(f"Error in treat_athlete: {e}\n{traceback.format_exc()}")
        return no_update, no_update, no_update, True, "Une erreur s'est produite. Consultez les journaux pour plus de détails.", False



