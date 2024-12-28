"""
This module contains callback functions that handle the first linear modelization of the analysis.

Features :
- Graph visualization.
- Graph interactions (modelization type, points selection, points protection)
- Outliers management.
- Analysis step change.

"""

### Imports ###

from dash import Input, Output, State, callback, callback_context, no_update
from utils.crud import *
from dash.exceptions import PreventUpdate
from utils.data_processing import *
from utils.graph_updating import *
import logging

### Callbacks ###

# --- SECTION 1 : Callback for the graph visualization ---
@callback(
    Output("graph-first_linear", "figure", allow_duplicate=True),
    Output("title_analysis_first_linear", "children"),
    Output('slider-div-outlier-first_linear', "children"),
    Output("data_vo2", "data", allow_duplicate=True),
    Output('data_outliers', "data", allow_duplicate=True),
    Output('back', 'data', allow_duplicate=True),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input("div-first_linear", "children"),
    State("test", "data"),
    State("back", "data"),
    State("initial_df_second_modelization", "data"),
    prevent_initial_call='initial_duplicate'
)
def display_figure_first_linear(graph, test_id, bool_back, initial_df):
    """
    Displays the first linear analysis figure along with the associated outlier slider and data.
    It either loads the initial dataframe or a test-specific dataframe depending on the back flag.

    Parameters
    ----------
    graph : dict
        The current graph figure dictionary, containing layout and data.
    test_id : str
        The identifier of the test data in the database.
    bool_back : bool
        A flag indicating whether to go back to the previous analysis step or continue with the current step.
    initial_df : list
        The initial dataframe used for second modelization if returning to a previous step.

    Returns
    -------
    tuple
        - The updated graph figure with the linear analysis and protective bars.
        - The title of the analysis.
        - The slider element for controlling outlier thresholds.
        - The data for VO2 values in JSON format.
        - The data for outliers in JSON format.
        - A flag indicating whether the "back" button is activated.
        - Visibility and message of the error toast.
    """

    # Load data based on the back flag
    df = load_dataframe_vo2(initial_df) if bool_back else load_dataframe_from_id(test_id)

    try:
        # Generate the figure and outlier data for linear analysis
        fig, df_outliers = plot_analysis(df, "linear")

        # Add protective vertical bars to the figure
        fig['layout']['shapes'] = create_protect_bars(df)

        # Create the outlier slider
        slider_outlier = dcc.Slider(
            id="slider-outlier-first_linear",
            min=0, 
            max=200, 
            step=1, 
            marks={i: str(i) for i in range(0, 201, 25)},
            value=100
        )

        return (
            fig, 
            "Modélisation linéaire des données", 
            slider_outlier, 
            df.to_json(orient="records", date_format="iso"), 
            df_outliers.to_json(orient="records", date_format="iso"), 
            False, 
            False, 
            ""
        )
    
    except Exception as e:
        logging.error(f"Error in display_figure_first_linear : {e}/n{traceback.format_exc()}")
        return (
            no_update, no_update, no_update, no_update, no_update, no_update,
            True, "Une erreur s'est produite. Consultez les journaux pour plus de détails."
        )
    
# --- SECTION 2 : Callback for the type of modelization selection and points deletion ---
@callback(
    Output('graph-first_linear', 'figure', allow_duplicate=True),
    Output('data_outliers', "data", allow_duplicate=True),
    Output('data_vo2', 'data', allow_duplicate=True),
    Output("slider-div-protect-left-first_linear", "children"),
    Output("slider-div-protect-right-first_linear", "children"),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input('graph-first_linear', 'clickData'),
    Input("dropdown-model-first_linear", "value"),
    State('graph-first_linear', 'figure'),
    State('test', 'data'),
    State('data_vo2', 'data'),
    State('n_outliers', 'data'),
    State("initial_df_second_modelization", "data"),
    State("back", "data"),
    prevent_initial_call=True
)
def update_figure_first_linear(click_data, modelization_type, figure, test_id, data_vo2, n_outliers, initial_df, bool_back):
    """
    Updates the first linear analysis figure based on user interactions such as clicking on the graph 
    or changing the modelization type. It updates the figure, data, and sliders for outlier protection.

    Parameters
    ----------
    click_data : dict, optional
        Data corresponding to the clicked point on the graph.
    modelization_type : str
        The selected modelization type, either "first-model-first_linear" or "second-model-first_linear".
    figure : dict
        The current figure data for the graph.
    test_id : str
        The test identifier used to fetch data.
    data_vo2 : list
        The current VO2 data.
    n_outliers : int
        The number of outliers to consider in the modelization.
    initial_df : list
        The initial dataframe for second modelization if returning to a previous analysis step.
    bool_back : bool
        Flag indicating whether the user has pressed the back button.

    Returns
    -------
    tuple
        - The updated figure data for the graph.
        - The updated outliers data in JSON format.
        - The updated VO2 data in JSON format.
        - The slider elements for protecting the left and right bars.
        - Error toast details if any error occurs.
    """
    try:

        # Load data depending on the 'back' flag or available VO2 data
        if not data_vo2:
            df = load_dataframe_vo2(initial_df) if bool_back else load_dataframe_from_id(test_id)
        else:
            df = load_dataframe_vo2(data_vo2)

        # Handle click on a data point in the graph
        if click_data:
            point = click_data['points'][0]
            clicked_time = point['x']

            # Ignore clicks on modelization points
            if point.get('curveNumber') == 1:
                raise PreventUpdate

            # Remove the clicked point from the dataframe
            df = remove_clicked_point(df, clicked_time)
            if df.empty:
                logging.warning("Toutes les données ont été supprimées.")
                raise PreventUpdate
            
            # Update figure with modified data
            figure['data'][0]['x'] = df['t']
            figure['data'][0]['y'] = df['VO2']

        # Handle modelization type change
        if figure:
            
            # Retrieve positions of protective bars
            pos_left = figure['layout']['shapes'][0]['x0'] 
            pos_right = figure['layout']['shapes'][1]['x0'] 

            # Ensure positions are within data range
            if not(df['t'].min() <= pd.to_datetime(pos_left) <= df['t'].max()):
                pos_left = df['t'].min()
            if not(df['t'].min() <= pd.to_datetime(pos_right) <= df['t'].max()):
                pos_right = df['t'].max()

            # Retrieve the number of outliers
            n_outliers = n_outliers or 100

            # Create sliders
            first_part_df, second_part_df = divise_df(df)
            slider_left = create_slider(
                first_part_df, 
                first_part_df.searchsorted(pos_left), 
                "slider-bar-protect-left-first_linear", 
                3, 
                time=True
            )
            slider_right = create_slider(
                second_part_df, 
                second_part_df.searchsorted(pos_right), 
                "slider-bar-protect-right-first_linear", 
                3, 
                time=True
            )

            # Two-segment model
            if modelization_type == "first-model-first_linear":
                figure, df_outliers = update_figure_modelization(df, figure, n_outliers, pos_left, pos_right, 2)
                return (
                    figure, 
                    df_outliers.to_json(orient="records", date_format="iso"), 
                    df.to_json(date_format='iso', orient="records"), 
                    slider_left, 
                    slider_right,
                    False,
                    ""
                )

            # Three-segment model
            elif modelization_type == "second-model-first_linear":
                figure, df_outliers = update_figure_modelization(df, figure, n_outliers, pos_left, pos_right, 3)
                return (
                    figure, 
                    df_outliers.to_json(orient="records", date_format="iso"), 
                    df.to_json(date_format='iso', orient="records"), 
                    slider_left, 
                    slider_right,
                    False,
                    ""
                )

        # If no figure update, create default sliders
        first_part_df, second_part_df = divise_df(df)
        slider_left = create_slider(first_part_df, 0, "slider-bar-protect-left-first_linear", 3, time=True)
        slider_right = create_slider(second_part_df, len(second_part_df)-1, "slider-bar-protect-right-first_linear", 3, time=True)

        return no_update, no_update, no_update, slider_left, slider_right, False, ""
        
    except Exception as e:
        logging.error(f"Error in update_figure_first_linear : {e}/n{traceback.format_exc()}")
        return (
            no_update, no_update, no_update, no_update, no_update, 
            True, "Une erreur s'est produite. Consultez les journaux pour plus de détails."
        )
    
# --- SECTION 3 : Callback handling the vertical bars of data protection ---
@callback(
    Output('graph-first_linear', 'figure', allow_duplicate=True),
    Output('data_outliers', "data", allow_duplicate=True),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input("slider-bar-protect-left-first_linear", 'value'),
    Input("slider-bar-protect-right-first_linear", 'value'),
    State('graph-first_linear', 'figure'),
    State('data_vo2', 'data'),
    State('data_outliers', 'data'),
    prevent_initial_call=True
)
def display_bar_first_linear(slider_value_left, slider_value_right, graph, data_vo2, data_outliers):
    """
    Updates the protection bars' positions in the first linear analysis graph based on the slider values for left and right bars.
    The function also modifies the outliers' data points displayed in the graph.

    Parameters
    ----------
    slider_value_left : int
        The value selected for the left protection bar (slider).
    slider_value_right : int
        The value selected for the right protection bar (slider).
    graph : dict
        The current figure data for the graph.
    data_vo2 : list
        The current VO2 data.
    data_outliers : list
        The current outliers data.

    Returns
    -------
    tuple
        - Updated figure for the graph.
        - Updated outliers data in JSON format.
        - Error toast message and state if an error occurs.
    """
    try:
        if graph:
            
            # Get the data
            df = load_dataframe_vo2(data_vo2)
            first_part_df, second_part_df = divise_df(df)

            # Get the outliers position
            if data_outliers:
                df_outliers = load_dataframe_vo2(data_outliers)
                pos_left = df_outliers['t'].searchsorted(first_part_df.iloc[slider_value_left])
                pos_right = df_outliers['t'].searchsorted(second_part_df.iloc[slider_value_right])

                # Modify outlier data within the selected region
                graph['data'][2]['x'] = df_outliers['t'][pos_left:pos_right]
                graph['data'][2]['y'] = df_outliers['VO2'][pos_left:pos_right]

            # Detect the triggering element of the callback
            trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

            # Modify the position of the left protection bar
            if trigger == "slider-bar-protect-left-first_linear":
                graph['layout']['shapes'][0]['x0'] = first_part_df.iloc[slider_value_left]
                graph['layout']['shapes'][0]['x1'] = first_part_df.iloc[slider_value_left]

            # Modification de la position de la barre de droite
            if trigger == "slider-bar-protect-right-first_linear":
                graph['layout']['shapes'][1]['x0'] = second_part_df.iloc[slider_value_right]
                graph['layout']['shapes'][1]['x1'] = second_part_df.iloc[slider_value_right]

            return graph, df_outliers.to_json(orient="records", date_format="iso"), False, ""
        
        return no_update

    except Exception as e:
        logging.error(f"Error in display_bar_first_linear : {e}\n{traceback.format_exc()}")
        return (
            no_update, no_update,
            True, "Une erreur s'est produite. Consultez les journaux pour plus de détails."
        )
    
# --- SECTION 4 : Callback that handle the outliers ---
@callback(
    Output('graph-first_linear', 'figure', allow_duplicate=True),
    Output('data_outliers', "data", allow_duplicate=True),
    Output("n_outliers", "data", allow_duplicate=True),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input('slider-outlier-first_linear', 'value'),
    State('graph-first_linear', 'figure'),
    State('data_vo2', 'data'),
    State("dropdown-model-first_linear", "value"),
    prevent_initial_call=True
)
def outliers_first_linear(slider_value_outlier, graph, data_vo2, model_type):
    """
    Updates the outliers' data and the graph based on the selected slider value for outliers in the first linear analysis.

    Parameters
    ----------
    slider_value_outlier : int
        The value selected for the outlier detection (slider).
    graph : dict
        The current figure data for the graph.
    data_vo2 : list
        The current VO2 data.
    model_type : str
        The selected model type to determine the number of segments for outlier detection.

    Returns
    -------
    tuple
        - Updated graph figure.
        - Updated outliers data in JSON format.
        - The selected outlier slider value.
        - Error toast message and state if an error occurs.
    """
    try:
        if graph:
            
            # Get the data
            df = load_dataframe_vo2(data_vo2)

            # Retrieve the positions of the protection bars
            limit_left = graph['layout']['shapes'][0]['x0']
            limit_right = graph['layout']['shapes'][1]['x0']
                
            # Modelization based on the number of outliers
            n = 2 if 'first' in model_type else 3
            _, df_outliers = modelization_cubic_linear(df, slider_value_outlier, n)

            # Visualize the outliers
            if df_outliers.shape[0] != 0:
                pos_left = df_outliers['t'].searchsorted(limit_left)
                pos_right = df_outliers['t'].searchsorted(limit_right)

                # Modify outlier data within the selected range
                graph['data'][2]['x'] = df_outliers['t'].iloc[pos_left:pos_right]
                graph['data'][2]['y'] = df_outliers['VO2'].iloc[pos_left:pos_right]

            else:
                # No outliers, clear the outlier data from the graph
                graph['data'][2]['x'] = []
                graph['data'][2]['y'] = []

            return graph, df_outliers.to_json(orient="records", date_format="iso"), slider_value_outlier, False, ""
        
        return no_update

    except Exception as e:
        logging.error(f"Error in outliers_first_linear : {e}\n{traceback.format_exc()}")
        return (
            no_update, no_update, no_update,
            True, "Une erreur s'est produite. Consultez les journaux pour plus de détails."
        )

# --- SECTION 5 : Callback that handle the change of analysis step ---
@callback(
    Output("url", "pathname", allow_duplicate=True),
    Output("initial_df_third_modelization", "data"),
    Output("back", "data", allow_duplicate=True),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input('prev-button-first_linear', 'n_clicks'),
    Input('next-button-first_linear', 'n_clicks'),
    State('test', 'data'),
    State("data_vo2", "data"),
    State("data_outliers", "data"),
    State('graph-first_linear', 'figure'),
    prevent_initial_call=True
)
def change_analysis_step_first_linear(prev_btn, next_btn, test_id, data_vo2, data_outliers, graph):
    """
    Handles the transition between different steps of the analysis (first linear analysis) and updates the test data.

    Parameters
    ----------
    prev_btn : int
        The number of clicks for the "previous" button.
    next_btn : int
        The number of clicks for the "next" button.
    test_id : str
        The ID of the test.
    data_vo2 : list
        The current VO2 data.
    data_outliers : list
        The data with outliers to be removed.
    graph : dict
        The figure data for the graph.

    Returns
    -------
    tuple
        - Updated URL path.
        - The filtered data in JSON format.
        - The "back" data (True/False).
        - Error toast state and message in case of an error.
    """
    try:

        # Detect the triggering element of the callback
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        if trigger == "prev-button-first_linear" and prev_btn:

            return "/cubic_analysis", None, True, False, ""
        
        if trigger == "next-button-first_linear" and next_btn:

            # Load the data
            session = next(get_db())
            df = load_dataframe_vo2(data_vo2)

            # Modify the dataframe to exclude outliers if they exist
            if data_outliers:

                # Load the outliers data
                df_outliers = load_dataframe_vo2(data_outliers)

                # Get the positions of the protection bars
                limit_left = graph['layout']['shapes'][0]['x0']
                limit_right = graph['layout']['shapes'][1]['x0']

                # Get the indices for the left and right positions
                pos_left = df_outliers['t'].searchsorted(limit_left)
                pos_right = df_outliers['t'].searchsorted(limit_right)

                # Update the outliers data to the selected range
                df_outliers = df_outliers.iloc[pos_left:pos_right]

                # Filter the main dataframe to exclude the outliers
                df_filtered = df[~df['t'].isin(df_outliers['t'])].reset_index(drop = True).to_json(orient="records", date_format="iso")

            else:
                # No outliers, proceed with the original data
                df_filtered = df.to_json(orient="records", date_format="iso")

            # Update the test data in the database
            test = update(
                db=session, 
                model=Test, 
                object_id=test_id, 
                obj_in={
                    'computed_dataframe': df_filtered,
                }
            )

            return "/second_linear_analysis", df_filtered, False, False, ""
        
        return no_update

    except Exception as e:
        logging.error(f"Error in change_analysis_step_first_linear : {e}\n{traceback.format_exc()}")
        return (
            no_update, no_update, no_update,
            True, "Une erreur s'est produite. Consultez les journaux pour plus de détails."
        )