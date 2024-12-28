"""
This module handles the callbacks for the previsualisation of the original data after upload of the analysis.

Features :
- Data display.
- Select lactate mode.
- Analysis step management.

"""

### Imports ###
from dash import Input, Output, State, callback, callback_context, no_update
import pandas as pd
from io import StringIO
import json
from utils.data_processing import *
from utils.graph_updating import *
from utils.crud import *

### Callbacks ###

# --- SECTION 1 : Callback related to the visualization of the data ---
@callback(
    Output("table-vo2", "columns"),
    Output("table-vo2", "data"),
    Output("table-lactate", "columns"),
    Output("table-lactate", "data"),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input("tables-div", "children"),
    State("test", "data"),
    prevent_initial_call=True
)
def display_data(tables, test_id) :
    """
    Callback to display data in two Dash tables: `table-vo2` and `table-lactate`.

    This function retrieves the VO2 and lactate data from a test record in the database,
    formats them appropriately, and returns the formatted data to be displayed in Dash tables.

    Parameters
    ----------
    tables : str
        The children of the 'tables-div' component, which could be used to trigger the callback.
    test_id : str
        The unique identifier of the test record to retrieve data from.

    Returns
    -------
    tuple
        - Columns and data formatted for the `table-vo2`.
        - Columns and data formatted for the `table-lactate`.
        - Visibility and message of the error toast
    """
    try:

        # Validate input: Return empty tables if test_id is not provided
        if not test_id:
            return (
                [], [], [], [], True,
                "Le test n'a pas pu être créé ou est introuvable."
            )
        
        # Retrieve the data
        session = next(get_db())
        test = get_by_id(session, Test, test_id)

        # Load data from the test record
        df_vo2 = pd.read_json(StringIO(test.source_vo2), orient="records")
        df_lactate = pd.read_json(StringIO(test.source_excel), orient="records")
    
        # Format columns and data for the VO2 and lactate tables
        columns_vo2 = format_columns(df_vo2, numeric_exclude=["t", "FC"])
        columns_lactate = format_columns(df_lactate, numeric_exclude=["temps", "paliers", "pente", 
                                                                    "vitesse", "fc", "rpe", "remarques"])

        # Convert dataframes to dictionaries
        df_vo2 = df_vo2.to_dict("records")
        df_lactate = df_lactate.to_dict("records")

        return columns_vo2, df_vo2, columns_lactate, df_lactate, False, ""
    
    except Exception as e:
        logging.error(f"Error in display_data : {e}\n{traceback.format_exc()}")
        return (
            [], [], [], [], 
            True, "Une erreur s'est produite. Consultez les journaux pour plus de détails."
        )

# --- SECTION 2 : Callback related to the selection of lactate data ---
@callback(
    Output('switch-lactate', "label"),
    Output("use_lactate", "data"),
    Input('switch-lactate', "value")
)
def choose_lactate_mode(switch_value):
    """
    Callback to handle the lactate switch mode. This function updates the label 
    and the internal state of the `use_lactate` variable based on the switch's 
    value.

    Parameters
    ----------
    switch_value : bool
        The current value of the lactate switch. If `True`, the mode is activated, 
        otherwise, it is deactivated.

    Returns
    -------
    tuple
        - A string indicating whether the lactate mode is "Activé" (enabled) 
          or "Désactivé" (disabled).
        - A boolean indicating whether the lactate mode is enabled (`True`) or 
          disabled (`False`).
    """
    try:
        # If the switch is off, return "Désactivé" and False for use_lactate
        if not switch_value:
            return "Désactivé", False
        
        # If the switch is on, return "Activé" and True for use_lactate
        return "Activé", True

    except Exception as e:
        logging.error(f"Error in choose_lactate_mode: {str(e)}\n{traceback.format_exc()}")
        return "Erreur", False

# --- SECTION 3 : Callback that handle the next step of the analysis ---
@callback(
    Output("url", "pathname", allow_duplicate=True),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input('validate-button-overview', 'n_clicks'),
    State('use_lactate', 'data'),
    State("test", "data"),
    prevent_initial_call=True
)
def continue_analysis_overview(click_validate, bool_lactate, test_id):
    """
    Callback to handle the validation of the overview analysis and proceed to 
    the next step in the analysis process, updating the test data if lactate 
    mode is enabled.

    Parameters
    ----------
    click_validate : int
        The number of clicks on the "validate-button-overview". 
        Each click triggers the callback to validate and proceed.

    bool_lactate : bool
        Indicates whether the lactate mode is enabled. If `True`, it updates 
        the lactate information in the database; otherwise, it simply proceeds.

    test_id : str
        The ID of the test from the session, used to retrieve and update the 
        test information.

    Returns
    -------
    str
        The pathname of the next step in the analysis process. In this case, 
        it returns "/time_analysis" to proceed to the time analysis step.
    bool
        Visibility of the error toast
    str
        Message of the error toast
    """
    try:
        # Detect the triggering element of the callback
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        if trigger == "validate-button-overview" and click_validate:
            
            # If lactate mode is enabled, update the test data
            if bool_lactate:

                # Load the lactate data from the test source
                session = next(get_db())
                test = get_by_id(session, Test, test_id)
                df_lactate = pd.read_json(StringIO(test.source_excel), orient="records")

                # Update the test data with the maximum lactate value
                test = update(
                    db=session, 
                    model=Test, 
                    object_id=test_id, 
                    obj_in={
                        'lactate_max': df_lactate['lactate'].max()
                    }
                )

            return "/time_analysis", False, ""
        
        return no_update
    
    except Exception as e:
        logging.error(f"Error in continue_analysis_overview: {str(e)}\n{traceback.format_exc()}")
        return no_update, True, "Une erreur s'est produite. Consultez les journaux pour plus de détails."