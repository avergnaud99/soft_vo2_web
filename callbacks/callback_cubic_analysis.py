"""
This module contains callback functions that handle the cubic modelization of the analysis.

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
    Output("graph-cubic", "figure", allow_duplicate=True),
    Output("title_analysis_cubic", "children"),
    Output('slider-div-outlier-cubic', "children"),
    Output("data_vo2", "data", allow_duplicate=True),
    Output('data_outliers', "data", allow_duplicate=True),
    Output('back', 'data', allow_duplicate=True),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input("div-cubic", "children"),
    State("test", "data"),
    State("back", "data"),
    State("initial_df_first_modelization", "data"),
    prevent_initial_call='initial_duplicate'
)
def display_figure_cubic(graph, test_id, bool_back, initial_df):
    """
    Displays a cubic model analysis graph based on the given test data.

    Parameters
    ----------
    graph : str
        Placeholder for the callback input (not used directly).
    test_id : str
        Identifier of the test to retrieve data.
    bool_back : bool
        Flag indicating whether to load data from the initial DataFrame or fetch it based on the test ID.
    initial_df : dict
        Serialized representation of the initial DataFrame for modelization.

    Returns
    -------
    tuple
        - A Plotly figure showing the cubic model.
        - Title of the analysis.
        - A Dash slider component for outlier management.
        - Serialized data of the main VO2 DataFrame in JSON format.
        - Serialized data of outliers in JSON format.
        - Reset flag for 'back' state (False).
        - Error toast visibility state (False in case of success).
        - Error toast message (empty in case of success).
    """
    try:
        # Load the data
        df = load_dataframe_vo2(initial_df) if bool_back else load_dataframe_from_id(test_id)
    
        # Create the cubic model graph and identify outliers
        fig, df_outliers = plot_analysis(df, "cubic")

        # Add vertical protective bars
        fig['layout']['shapes'] = create_protect_bars(df)

        # Create a slider for managing outliers
        slider_outlier = dcc.Slider(
            id="slider-outlier-cubic",
            min=0, 
            max=200, 
            step=1, 
            marks={i: str(i) for i in range(0, 201, 25)},
            value=100
        )

        return (
            fig, 
            "Modélisation cubique des données", 
            slider_outlier, 
            df.to_json(orient="records", date_format="iso"), 
            df_outliers.to_json(orient="records", date_format="iso"), 
            False,
            False, 
            ""
        )
    
    except Exception as e:
        logging.error(f"Error in display_figure_cubic : {e}/n{traceback.format_exc()}")
        return (
            no_update, no_update, no_update, no_update, no_update, no_update,
            True, "Une erreur s'est produite. Consultez les journaux pour plus de détails."
        )
    
# --- SECTION 2 : Callback for the type of modelization selection and points deletion ---
@callback(
    Output('graph-cubic', 'figure', allow_duplicate=True),
    Output('data_outliers', "data", allow_duplicate=True),
    Output('data_vo2', 'data', allow_duplicate=True),
    Output("slider-div-protect-left-cubic", "children"),
    Output("slider-div-protect-right-cubic", "children"),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input('graph-cubic', 'clickData'),
    Input("dropdown-model-cubic", "value"),
    State('graph-cubic', 'figure'),
    State('test', 'data'),
    State('data_vo2', 'data'),
    State('n_outliers', 'data'),
    State("initial_df_first_modelization", "data"),
    State("back", "data"),
    prevent_initial_call=True
)
def update_figure_cubic(click_data, modelization_type, figure, test_id, data_vo2, n_outliers, initial_df, bool_back):
    """
    Updates the cubic model graph based on user interactions, such as clicking on points or selecting a model type.

    Parameters
    ----------
    click_data : dict
        Data of the clicked point on the graph.
    modelization_type : str
        Selected type of cubic modelization ("first-model-cubic" or "second-model-cubic").
    figure : dict
        The current figure data of the graph.
    test_id : str
        Identifier of the test for retrieving data.
    data_vo2 : str
        Serialized VO2 data in JSON format.
    n_outliers : int
        Number of outliers for modelization.
    initial_df : dict
        Serialized initial DataFrame for modelization.
    bool_back : bool
        Flag indicating whether to load data from the initial DataFrame or fetch it based on the test ID.

    Returns
    -------
    tuple
        - Updated figure for the cubic model graph.
        - Serialized data of outliers in JSON format.
        - Serialized data of the VO2 DataFrame in JSON format.
        - Slider component for the left protective bar.
        - Slider component for the right protective bar.
        - Error toast visibility state (False in case of success).
        - Error toast message (empty in case of success).
    """
    try:
        # Load data
        if not data_vo2:
            df = load_dataframe_vo2(initial_df) if bool_back else load_dataframe_from_id(test_id)
        else:
            df = load_dataframe_vo2(data_vo2)

        # Handle click on a data point
        if click_data:
            point = click_data['points'][0]
            clicked_time = point['x']

            # Ignore clicks on modelization points
            if point.get('curveNumber') == 1:
                raise PreventUpdate

            # Remove the clicked point
            df = remove_clicked_point(df, clicked_time)
            if df.empty:
                logging.error("Toutes les données ont été supprimées.")
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
                "slider-bar-protect-left-cubic", 
                3, 
                time=True
            )
            slider_right = create_slider(
                second_part_df, 
                second_part_df.searchsorted(pos_right), 
                "slider-bar-protect-right-cubic", 
                3, 
                time=True
            )

            # Single-segment model
            if modelization_type == "first-model-cubic":
                figure, df_outliers = update_figure_modelization(df, figure, n_outliers, pos_left, pos_right, 0)
                return (
                    figure, 
                    df_outliers.to_json(orient="records", date_format="iso"), 
                    df.to_json(date_format='iso', orient="records"), 
                    slider_left, 
                    slider_right,
                    False,
                    ""
                )

            # Two-segment model
            elif modelization_type == "second-model-cubic":
                figure, df_outliers = update_figure_modelization(df, figure, n_outliers, pos_left, pos_right, 1)
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
        slider_left = create_slider(first_part_df, 0, "slider-bar-protect-left-cubic", 3, time=True)
        slider_right = create_slider(second_part_df, len(second_part_df)-1, "slider-bar-protect-right-cubic", 3, time=True)

        return no_update, no_update, no_update, slider_left, slider_right, False, ""
        
    except Exception as e:
        logging.error(f"Error in update_figure_cubic : {e}/n{traceback.format_exc()}")
        return (
            no_update, no_update, no_update, no_update, no_update, 
            True, "Une erreur s'est produite. Consultez les journaux pour plus de détails."
        )
    
# --- SECTION 3 : Callback handling the vertical bars of data protection ---
@callback(
    Output('graph-cubic', 'figure', allow_duplicate=True),
    Output('data_outliers', "data", allow_duplicate=True),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input("slider-bar-protect-left-cubic", 'value'),
    Input("slider-bar-protect-right-cubic", 'value'),
    State('graph-cubic', 'figure'),
    State('data_vo2', 'data'),
    State('data_outliers', 'data'),
    prevent_initial_call=True
)
def display_bar_cubic(slider_value_left, slider_value_right, graph, data_vo2, data_outliers):
    """
    Updates the cubic graph's left and right protection bars based on slider values, 
    and processes outliers based on selected regions.

    Parameters
    ----------
    slider_value_left : int
        The slider value for the left protection bar, which corresponds to a position in the data.
    slider_value_right : int
        The slider value for the right protection bar, which corresponds to a position in the data.
    graph : dict
        The current graph figure dictionary, containing layout and data.
    data_vo2 : list
        The data representing VO2 values for plotting and analysis.
    data_outliers : list
        The data representing outliers in the VO2 dataset.

    Returns
    -------
    tuple
        - Updated graph figure dictionary with modified protection bars.
        - JSON string of processed outlier data.
        - Error toast visibility state (False in case of success).
        - Error toast message (empty in case of success).
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
            if trigger == "slider-bar-protect-left-cubic":
                graph['layout']['shapes'][0]['x0'] = first_part_df.iloc[slider_value_left]
                graph['layout']['shapes'][0]['x1'] = first_part_df.iloc[slider_value_left]

            # Modify the position of the right protection bar
            if trigger == "slider-bar-protect-right-cubic":
                graph['layout']['shapes'][1]['x0'] = second_part_df.iloc[slider_value_right]
                graph['layout']['shapes'][1]['x1'] = second_part_df.iloc[slider_value_right]

            return graph, df_outliers.to_json(orient="records", date_format="iso"), False, ""
        
        return no_update

    except Exception as e:
        logging.error(f"Error in display_bar_cubic : {e}\n{traceback.format_exc()}")
        return (
            no_update, no_update,
            True, "Une erreur s'est produite. Consultez les journaux pour plus de détails."
        )
    
# --- SECTION 4 : Callback that handle the outliers ---
@callback(
    Output('graph-cubic', 'figure', allow_duplicate=True),
    Output('data_outliers', "data", allow_duplicate=True),
    Output("n_outliers", "data"),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input('slider-outlier-cubic', 'value'),
    State('graph-cubic', 'figure'),
    State('data_vo2', 'data'),
    State("dropdown-model-cubic", "value"),
    prevent_initial_call=True
)
def outliers_cubic(slider_value_outlier, graph, data_vo2, model_type):
    """
    Identifies and visualizes outliers in the cubic model based on slider input, 
    while updating the graph with the outlier data.

    Parameters
    ----------
    slider_value_outlier : int
        The slider value used to control the sensitivity of outlier detection.
    graph : dict
        The current graph figure dictionary, containing layout and data.
    data_vo2 : list
        The data representing VO2 values for analysis and plotting.
    model_type : str
        The type of model used for outlier detection, influencing the method of modelization.

    Returns
    -------
    tuple
        - Updated graph figure dictionary with the visualized outliers.
        - JSON string of outlier data.
        - The current slider value used for the outlier detection.
        - Visibility and message of the error toast.
    """
    try:
        if graph:
            
            # Get the data
            df = load_dataframe_vo2(data_vo2)

            # Retrieve the positions of the protection bars
            limit_left = graph['layout']['shapes'][0]['x0']
            limit_right = graph['layout']['shapes'][1]['x0']
                
            # Modelization based on the number of outliers
            n = 0 if 'first' in model_type else 1
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
        logging.error(f"Error in outliers_cubic : {e}\n{traceback.format_exc()}")
        return (
            no_update, no_update, no_update,
            True, "Une erreur s'est produite. Consultez les journaux pour plus de détails."
        )

# --- SECTION 5 : Callback that handle the change of analysis step ---
@callback(
    Output("url", "pathname", allow_duplicate=True),
    Output("initial_df_second_modelization", "data"),
    Output('back', 'data', allow_duplicate=True),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input('prev-button-cubic', 'n_clicks'),
    Input('next-button-cubic', 'n_clicks'),
    State('test', 'data'),
    State("data_vo2", "data"),
    State("data_outliers", "data"),
    State('graph-cubic', 'figure'),
    prevent_initial_call=True
)
def change_analysis_step_cubic(prev_btn, next_btn, test_id, data_vo2, data_outliers, graph):
    """
    Changes the analysis step in a cubic modelization process based on button clicks. 
    It processes the data, updates the database, and navigates to the appropriate analysis step.

    Parameters
    ----------
    prev_btn : int
        The number of times the "previous" button has been clicked.
    next_btn : int
        The number of times the "next" button has been clicked.
    test_id : str
        The identifier of the test data in the database.
    data_vo2 : list
        The data representing VO2 values for the analysis.
    data_outliers : list
        The data representing outliers in the VO2 dataset.
    graph : dict
        The current graph figure dictionary, containing layout and data.

    Returns
    -------
    tuple
        - The URL path for the next analysis step or previous step.
        - The filtered data for the second modelization.
        - A flag indicating whether the "back" button is activated.
    """
    try:

        # Detect the triggering element of the callback
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        if trigger == "prev-button-cubic" and prev_btn:

            return "/time_analysis", None, False, False, ""
        
        if trigger == "next-button-cubic" and next_btn:
                
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
                    'computed_dataframe': df_filtered
                }
            )

            return "/first_linear_analysis", df_filtered, False, False, ""
        
        return no_update

    except Exception as e:
        logging.error(f"Error in change_analysis_step_cubic : {e}\n{traceback.format_exc()}")
        return (
            no_update, no_update, no_update,
            True, "Une erreur s'est produite. Consultez les journaux pour plus de détails."
        )