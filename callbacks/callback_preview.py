"""
This module handles the callbacks related to the previsualization part of the modelizations of the analysis.

Features :
- Display the modelizations.
- Manage the RER error.
- Change the analysis step.

"""

### Imports ###

from dash import Input, Output, State, callback, callback_context, no_update
from utils.crud import *
from utils.data_processing import *
from utils.graph_updating import *
import logging
import traceback

### Callbacks ###

# --- SECTION 1 : Callback related to the figures ---
@callback(
    Output('preview-graphs', 'children'),
    Output("final_models", "data"),
    Output("error-rer", "data"),
    Output("data_vo2", "data"),
    Output('data_lactate', "data"),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input('div-preview', 'children'),
    State("test", "data"),
    State("back", "data"),
    State("use_lactate", "data"),
    State("initial_df_preview", "data"),
    prevent_initial_call='initial_duplicate'
)
def display_modelizations(graphs, test_id, bool_back, bool_lactate, initial_df):
    """
    Callback function to generate and display modelizations and graphs based on VO2 and lactate data.

    This function loads VO2 and lactate data, processes the data, and generates the appropriate models 
    and graphs. If an error occurs during the data loading or modeling process, the error is logged 
    and returned to the user.

    Parameters
    ----------
    graphs : str
        The input value representing the content of the preview graphs div (trigger).
    test_id : str
        The test ID used to fetch the relevant data from the database.
    bool_back : bool
        A flag indicating whether the user has clicked the "back" button and should load previous data.
    bool_lactate : bool
        A flag indicating whether lactate data should be included in the modeling.
    initial_df : str
        The initial data used for previewing the graphs before any modifications.

    Returns
    -------
    tuple
        preview_graphs : str
            The HTML or plot components representing the preview graphs.
        models : dict
            The model data that will be returned in JSON format for further use.
        error : str or None
            The error message if an issue occurred, or None if no error occurred.
        df_vo2 : str
            The VO2 data in JSON format for downstream usage.
        df_lactate : str or None
            The lactate data in JSON format if lactate is used, or None if not.
        tuple
            Visibility and message of the error toast.
    """
    try:
        
        # Retrieve VO2 data based on whether the user clicked back or not
        df_vo2 = load_dataframe_vo2(initial_df) if bool_back else load_dataframe_from_id(test_id)

        # Retrieve lactate data and create modelizations and graphs if lactate data is enabled
        if bool_lactate:
            df_lactate = load_dataframe_from_id(test_id, lactate=True)
            preview_graphs, models, error = plot_modelizations(df_vo2, df_lactate)
            df_lactate = df_lactate.to_json(orient="records", date_format="iso")
        else:
            df_lactate = None
            preview_graphs, models, error = plot_modelizations(df_vo2)

        return preview_graphs, models, error, df_vo2.to_json(orient="records", date_format="iso"), df_lactate, False, ""

    except Exception as e:
        logging.error(f"Error in display_modelizations : {e}\n{traceback.format_exc()}")
        return (
            no_update, no_update, no_update, no_update, no_update,
            True, "Une erreur s'est produite. Consultez les journaux pour plus de détails."
        )
    
# --- SECTION 2 : Callback related to the selection of the RER error ---
@callback(
    Output('preview-graphs', 'children', allow_duplicate=True),
    Output("final_models", "data", allow_duplicate=True),
    Output("error-rer", "data", allow_duplicate=True),
    Output('switch-rer', "label"),
    Output('div-switch-rer', "hidden"),
    Output("data_vo2", "data", allow_duplicate=True),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input("div-switch-rer", "children"),
    Input('switch-rer', "value"),
    State("error-rer", "data"),
    State("test", "data"),
    State('data_lactate', "data"),
    State("use_lactate", "data"),
    State('div-switch-rer', "hidden"),
    prevent_initial_call=True
)
def choose_rer_error(div_switch, switch_value, error, test_id, data_lactate, bool_lactate, hidden):
    """
    Callback function to handle the RER (Respiratory Exchange Ratio) error handling and visualization.

    This function processes the data based on whether the RER switch is activated or not and generates
    the corresponding graphs and models. It also handles any errors related to the data processing.

    Parameters
    ----------
    div_switch : str
        The content of the div containing the switch for activating/deactivating the RER error feature.
    switch_value : bool
        The current value of the switch that indicates whether the RER error feature is activated.
    error : str
        The error data that might affect the start of the test.
    test_id : str
        The ID of the test used to fetch the relevant data.
    data_lactate : str
        The lactate data, if available.
    bool_lactate : bool
        A flag indicating whether lactate data should be included in the modeling.
    hidden : bool
        A flag indicating whether the switch should be hidden based on certain conditions.

    Returns
    -------
    tuple
        preview_graphs : str
            The HTML or plot components representing the preview graphs.
        models : dict
            The model data that will be returned in JSON format for further use.
        error : str
            The error message if an issue occurred, or None if no error occurred.
        switch_label : str
            The label of the switch ("Activé" or "Désactivé").
        hidden : bool
            Whether the switch is hidden based on error or initial state.
        df_vo2 : str
            The VO2 data in JSON format for downstream usage.
        tuple
            Visibility and message of the error toast
    """
    try:
        # Handle error at the beginning of the test and the visibility of the switch
        if error or hidden == False:
            hidden = False
        else:
            hidden = True

        # Load the VO2 data from the database
        df_vo2 = load_dataframe_from_id(test_id)

        # Change switch value label based on the RER switch state
        switch_label = "Désactivé"
        if switch_value:
            switch_label = "Activé"
            start = df_vo2[error >= df_vo2['VO2']].index[-1]
            df_vo2 = df_vo2.iloc[start:].reset_index(drop=True)

        # Load lactate data and create modelizations and graphs if lactate data is available
        if bool_lactate:
            df_lactate = load_dataframe_lactate(data_lactate)
            preview_graphs, models, error = plot_modelizations(df_vo2, df_lactate)
        else:
            preview_graphs, models, error = plot_modelizations(df_vo2)

        return (
            preview_graphs, models, error, 
            switch_label, hidden, 
            df_vo2.to_json(orient="records", date_format="iso"), 
            False, ""
        )
    
    except Exception as e:
        logging.error(f"Error in choose_rer_error : {e}\n{traceback.format_exc()}")
        return (
            no_update, no_update, no_update, no_update, no_update, no_update,
            True, "Une erreur s'est produite. Consultez les journaux pour plus de détails."
        )

# --- SECTION 3 : Callback related to the change of step of the analysis ---
@callback(
    Output("url", "pathname", allow_duplicate=True),
    Output("back", "data", allow_duplicate=True),
    Output("curves", 'data'),
    Output("results", "data"),
    Output("initial_df_preview", "data"),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input('prev-button-preview', 'n_clicks'),
    Input('next-button-preview', 'n_clicks'),
    State('data_vo2', 'data'),
    State('data_lactate', 'data'),
    State("final_models", "data"),
    State("test", "data"),
    State("use_lactate", "data"),
    prevent_initial_call=True
)
def change_analysis_step_preview(prev_btn, next_btn, data_vo2, data_lactate, final_models, test_id, use_lactate):
    """
    Callback function to navigate between the preview steps in the analysis, either moving to the previous
    or the next step. It processes data for the next step, computes results, and updates the test in the database.

    Parameters
    ----------
    prev_btn : int
        The number of times the "Previous" button has been clicked.
    next_btn : int
        The number of times the "Next" button has been clicked.
    data_vo2 : str
        The VO2 data in JSON format.
    data_lactate : str
        The lactate data in JSON format.
    final_models : dict
        The models containing curve data for the analysis.
    test_id : str
        The ID of the test being processed.
    use_lactate : bool
        A flag indicating whether lactate data should be used in the analysis.

    Returns
    -------
    tuple
        pathname : str
            The URL path for navigation (either to the previous or next step).
        back : bool
            Whether to go back to the previous step.
        curves : dict
            A dictionary containing the curve data for the analysis.
        results : dict
            The computed results for the analysis.
        initial_df_preview : str
            The preview of the initial data in JSON format.
    """
    try:

        # Detect the element that triggered the callback
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        if trigger == "prev-button-preview" and prev_btn:

            return "/second_linear_analysis", True, None, None, None, False, ""
        
        if trigger == "next-button-preview" and next_btn:

            # Load lactate data for the current test
            df_lactate = load_dataframe_lactate(data_lactate)

            # Compute the results based on the lactate data and final models
            results = computation(
                df_lactate=df_lactate,
                models=final_models,
                test_id=test_id,
                use_lactate=use_lactate
            )

            # Load VO2 data
            session = next(get_db())
            df_vo2 = load_dataframe_vo2(data_vo2)

            # Retrieve the maximum heart rate value (either from lactate or VO2 data)
            fc_max = int(load_dataframe_lactate(data_lactate)['fc'].max()) if use_lactate else int(df_vo2['FC'].max())

            # Update the test data in the database with the computed values
            test = update(
                db=session, 
                model=Test, 
                object_id=test_id, 
                obj_in={
                    've_max': df_vo2['VE'].max(),
                    'hr_max': fc_max,
                    'computed_dataframe': df_vo2.to_json(orient="records", date_format="iso")
                }
            )

            # Define standard curve names to be generated
            curve_names = ["FC", "VO2", "RER", "VCO2", "VE", "VE/VO2", "VE/VCO2"]

            # Conditionally add additional curves if present in final models
            if "PetCO2" in final_models and "PetO2" in final_models:
                curve_names.extend(["PetCO2", "PetO2"])

            # Save the curves data
            curves = {}
            for curve in curve_names:
                if curve in final_models:
                    curves[curve] = [
                        final_models[curve][0],
                        final_models[curve][3],
                        final_models[curve][2],
                        final_models[curve][1],
                    ]

            return '/results', False, curves, results, df_vo2.to_json(orient="records", date_format="iso"), False, ""
        
        return no_update
    
    except Exception as e:
        logging.error(f"Error in change_analysis_step_preview : {e}\n{traceback.format_exc()}")
        return (
            no_update, no_update, no_update, no_update, no_update,
            True, "Une erreur s'est produite. Consultez les journaux pour plus de détails."
        )