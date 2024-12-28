"""
This module handles the callbacks related to the time analysis.

Features :
- Display the figure.
- Remove the points.
- Manage the interactions with the sliders and the bars.
- Change the step of the analysis.

"""

### Imports ###

from dash import Input, Output, State, callback, callback_context, no_update
import pandas as pd
from utils.crud import *
import plotly.graph_objs as go
from datetime import date
from dash.exceptions import PreventUpdate
from utils.data_processing import *
from utils.graph_updating import *
import logging
import copy

### Callbacks ###

# --- SECTION 1 : Callback related to the graph visualization ---
@callback(
    Output("graph-time", "figure", allow_duplicate=True),
    Output("title_analysis_time", "children"),
    Output('data_vo2', 'data', allow_duplicate=True),
    Output("pos_start_cut", "data"),
    Output("pos_end_cut", "data"),
    Output("pos_start_plateau", "data"),
    Output("pos_vo2max", "data"),
    Output("dropdown-vertical-bars", "value"),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input("graph-time", "figure"),
    State("test", "data"),
    State("pos_start_cut", "data"),
    State("pos_end_cut", "data"),
    State("pos_start_plateau", "data"),
    State("pos_vo2max", "data"),
    prevent_initial_call="initial_duplicate"
)
def display_figure_time(graph, test_id, pos_start, pos_end, pos_plateau, pos_vo2_max):
    """
    Callback function to generate and display the time-based VO2 analysis graph.

    Parameters
    ----------
    graph : dict or None
        The current state of the time graph figure. If None, a new graph is generated.
    test_id : str
        The ID of the test for which data is being analyzed.
    pos_start : int or None
        Position of the start cut. If None, it is computed.
    pos_end : int or None
        Position of the end cut. If None, it is computed.
    pos_plateau : int or None
        Position of the plateau. If None, it is computed.
    pos_vo2_max : int or None
        Position of VO2 max. If None, it is computed.

    Returns
    -------
    tuple
        A tuple containing the updated graph, analysis title, VO2 data, 
        positions of start cut, end cut, plateau, VO2 max, dropdown value
        and the visibility and message of the error toast.
    """
    try:
        if not graph:

            # Retrieve the test object from the database
            session = next(get_db())
            test = get_by_id(session, Test, test_id)

            # Load VO2 data from the test
            df = load_dataframe_vo2(test.source_vo2)

            # Compute positions if they are not already defined
            pos_start = pos_start or modelization_time(df, "start")
            pos_end = pos_end or modelization_time(df, "end")
            pos_plateau = pos_plateau or modelization_time(df, "plateau")
            pos_vo2_max = pos_vo2_max or modelization_time(df, "max")

            # Define vertical bar properties for the graph
            dict_association = {
                'start-cut': ["gray", pos_start],
                'end-cut': ["black", pos_end],
                "plateau": ["blue", pos_plateau],
                "max": ["red", pos_vo2_max]
            }

            # Generate the initial plot for the analysis
            figure = plot_analysis(df, "time")
            fig = go.Figure(figure)

            # Add vertical bars to the graph
            for id, values in dict_association.items():
                fig.add_vline(
                    x=df["t"].iloc[values[1]], 
                    line=dict(color=values[0], width=2, dash="dash"),
                    name=id
                )

            # Title for the analysis step
            title = "Analyse temporelle des données"

            # Convert the dataframe to JSON format
            df = df.to_json(orient="records", date_format="iso")

            return fig, title, df, pos_start, pos_end, pos_plateau, pos_vo2_max, "start-cut", False, ""
        
        else:
            return no_update
    
    except Exception as e:
        logging.error(f"Error in display_figure_time: {e}\n{traceback.format_exc()}")
        return (
            no_update, no_update, no_update, no_update, no_update,
            no_update, no_update, no_update, True, "Une erreur s'est produite. Consultez les journaux pour plus de détails."
        )

# --- SECTION 2 : Callback related to the deletion of points ---
@callback(
    Output("graph-time", "figure", allow_duplicate=True),
    Output('data_vo2', 'data', allow_duplicate=True),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input('graph-time', 'clickData'),
    State('data_vo2', 'data'),
    State("graph-time", "figure"),
    prevent_initial_call=True
)
def remove_point(click_data, data_vo2, graph_data):
    """
    Callback to remove a clicked point from the graph and update the data.

    Parameters
    ----------
    click_data : dict
        Data from the graph indicating the point clicked by the user.
    data_vo2 : str
        JSON string of the VO2 data used for plotting the graph.
    graph_data : dict
        The current figure data of the time graph.

    Returns
    -------
    tuple
        Updated graph figure, updated VO2 data, error toast state, and error toast content.
    """

    # Check if no point was clicked
    if not click_data:
        raise PreventUpdate

    try:
        # Retrieve information about the selected point
        point = click_data['points'][0]
        clicked_time = point['x']

        # Ensure the clicked point belongs to the main curve
        if point.get('curveNumber') != 0:
            raise PreventUpdate

        # Load the VO2 data
        df = load_dataframe_vo2(data_vo2)

        # Remove the clicked point
        df = remove_clicked_point(df, clicked_time)

        # Check if all data points have been removed
        if df.empty:
            logging.warning("Toutes les données ont été supprimées.")
            raise PreventUpdate

        # Update the graph data
        graph_data = update_graph_data_time(graph_data, df)

        # Convert the updated data to JSON
        data_vo2_updated = df.to_json(date_format='iso', orient="records")

        return graph_data, data_vo2_updated, False, ""
    
    except Exception as e:
        logging.error(f"Error in remove_point: {e}\n{traceback.format_exc()}")
        return no_update, no_update, True, "Une erreur s'est produite. Consultez les journaux pour plus de détails."

# --- SECTION 3 : Callback related to slider management ---
@callback(
    Output("slider-div-time", "children"),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input("dropdown-vertical-bars", "value"),
    State("pos_start_cut", "data"),
    State("pos_end_cut", "data"),
    State("pos_start_plateau", "data"),
    State("pos_vo2max", "data"),
    State("data_vo2", "data"),
    State("test", "data"),
    prevent_initial_call=True
)
def display_slider(dropdown_value, pos_start, pos_end, pos_plateau, pos_vo2_max, data_vo2, test_id):
    """
    Callback to display a slider corresponding to the selected vertical bar in the dropdown.

    Parameters
    ----------
    dropdown_value : str
        The selected value from the dropdown, identifying the vertical bar type.
    pos_start : int
        The position of the start-cut vertical bar.
    pos_end : int
        The position of the end-cut vertical bar.
    pos_plateau : int
        The position of the plateau vertical bar.
    pos_vo2_max : int
        The position of the VO2 max vertical bar.
    data_vo2 : str
        JSON string containing the VO2 data.
    test_id : int
        The ID of the test to retrieve data from if `data_vo2` is unavailable.

    Returns
    -------
    tuple
        - The slider component for the selected bar type.
        - Boolean indicating if the error toast should be shown.
        - Visibility and string message for the error toast.
    """
    try:
        
        # Retrieve VO2 data
        if not data_vo2:

            # Fetch the test from the database
            session = next(get_db())
            test = get_by_id(session, Test, test_id)

            # Load the data from the test
            df = load_dataframe_vo2(test.source_vo2)

        else:
            # Load the data from the provided JSON
            df = load_dataframe_vo2(data_vo2)

        # Dictionary of bar positions
        dict_values = {
            "start-cut": pos_start,
            "end-cut": pos_end,
            "plateau": pos_plateau,
            "max": pos_vo2_max
        }

        # Get the current slider value or initialize it
        slider_value = dict_values.get(dropdown_value, 0)

        # Create the slider
        slider = create_slider(df["t"], slider_value, "slider-bar-time", 5, time=True)

        return slider, False, ""

    except Exception as e:
        logging.error(f"Error in display_slider: {e}\n{traceback.format_exc()}")
        return no_update, True, "Une erreur s'est produite. Consultez les journaux pour plus de détails."

# --- SECTION 4 : Callback related to vertical bar management ---
@callback(
    Output("vo2-value", "children"),
    Output('time-value', 'children'),
    Output('graph-time', 'figure', allow_duplicate=True),
    Output("pos_start_cut", "data", allow_duplicate=True),
    Output("pos_end_cut", "data", allow_duplicate=True),
    Output("pos_start_plateau", "data", allow_duplicate=True),
    Output("pos_vo2max", "data", allow_duplicate=True),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input('slider-bar-time', 'value'),
    State('graph-time', 'figure'),
    State("dropdown-vertical-bars", "value"),
    State("pos_start_cut", "data"),
    State("pos_end_cut", "data"),
    State("pos_start_plateau", "data"),
    State("pos_vo2max", "data"),
    State("test", "data"),
    prevent_initial_call=True
)
def display_bar(slider_value, graph, value_dropdown, pos_start, pos_end, pos_plateau, pos_vo2_max, test_id):
    """
    Callback to update VO2 and time values, as well as adjust graph shapes based on slider input.

    Parameters
    ----------
    slider_value : int
        The current value of the slider.
    graph : dict
        The current graph figure data.
    value_dropdown : str
        The selected dropdown value (e.g., "start-cut", "end-cut").
    pos_start : int
        The position of the start-cut bar.
    pos_end : int
        The position of the end-cut bar.
    pos_plateau : int
        The position of the plateau bar.
    pos_vo2_max : int
        The position of the VO2 max bar.
    test_id : int
        The ID of the test.

    Returns
    -------
    tuple
        - Updated VO2 value.
        - Updated time value.
        - Updated graph figure.
        - Updated positions for start-cut, end-cut, plateau, and VO2 max.
        - Visibility and message of the error toast.
    """
    try:
        if not graph:

            # Fetch the test
            session = next(get_db())
            test = get_by_id(session, Test, test_id)

            # Load VO2 data
            df = load_dataframe_vo2(test.source_vo2)

            # Get the VO2 value at the slider position
            value_vo2 = round(df['VO2'].iloc[slider_value], 2)

            # Get the time value at the slider position
            value_time = date.strftime(pd.to_datetime(df['t'].iloc[slider_value]), "%-Mmin%-Ss")

            return (
                value_vo2, value_time, no_update, no_update, 
                no_update, no_update, no_update, False, ""
            )

        else:

            # Extract the x-axis data
            axis = graph['data'][0]['x']

            # Update the position based on the dropdown selection
            if value_dropdown == "start-cut":
                pos_start = slider_value
            if value_dropdown == "end-cut":
                pos_end = slider_value
            if value_dropdown == "plateau":
                pos_plateau = slider_value
            if value_dropdown == "max":
                pos_vo2_max = slider_value
                
            # Get the VO2 value based on the dropdown
            if value_dropdown == "max":
                value_vo2 = round(graph['data'][1]['y'][slider_value], 2)
            else:
                value_vo2 = round(graph['data'][0]['y'][slider_value], 2)

            # Get the time value
            value_time = date.strftime(pd.to_datetime(axis[slider_value]), "%-Mmin%-Ss")
            
            # Update graph shapes with the new vertical bar position
            graph = update_graph_shapes(graph, axis[slider_value], value_dropdown)

            return (
                value_vo2, value_time, graph, pos_start, 
                pos_end, pos_plateau, pos_vo2_max, False, ""
            )
    
    except Exception as e:
        logging.error(f"Une erreur est survenue lors de la mise à jour des barres : {e}")
        return (
            no_update, no_update, no_update, no_update,
            no_update, no_update, no_update, True, 
            "Une erreur s'est produite. Consultez les journaux pour plus de détails."
        )

# --- SECTION 5 : Callback related to the change of step in the analysis ---
@callback(
    Output("url", "pathname", allow_duplicate=True),
    Output("initial_df_first_modelization", "data"),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input('prev-button-time', 'n_clicks'),
    Input('next-button-time', 'n_clicks'),
    State('test', 'data'),
    State("data_vo2", "data"),
    State("pos_start_cut", "data"),
    State("pos_end_cut", "data"),
    State("pos_start_plateau", "data"),
    State("pos_vo2max", "data"),
    State('graph-time', 'figure'),
    prevent_initial_call=True
)
def change_analysis_step_time(prev_btn, next_btn, test_id, data_vo2, pos_start, pos_end, pos_plateau, pos_vo2_max, graph):
    """
    Callback to handle navigation between analysis steps and update test data.

    Parameters
    ----------
    prev_btn : int
        Number of clicks on the "Previous" button.
    next_btn : int
        Number of clicks on the "Next" button.
    test_id : int
        ID of the test being analyzed.
    data_vo2 : str
        JSON-encoded VO2 data.
    pos_start : int
        Start position for the analysis.
    pos_end : int
        End position for the analysis.
    pos_plateau : int
        Position for the plateau data.
    pos_vo2_max : int
        Position of the VO2 max point.
    graph : dict
        Current graph data.

    Returns
    -------
    tuple
        - Pathname for the URL redirection.
        - Initial dataframe for the next analysis step.
        - Toast notification state (open/closed).
        - Toast notification message.
    """
    try:
        # Determine which button triggered the callback
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        if trigger == "prev-button-time" and prev_btn:

            return "/overview", None, False, ""
        
        if trigger == "next-button-time" and next_btn:

            # Load the VO2 data
            session = next(get_db())
            df = load_dataframe_vo2(data_vo2)

            # Slice the dataframe based on start and end positions
            new_df = copy.deepcopy(df)
            new_df = new_df.iloc[pos_start:pos_end].reset_index(drop = True).to_json(orient="records", date_format="iso")

            # Create the plateau dataframe
            df_plateau = df.iloc[pos_plateau:].reset_index(drop = True).to_json(orient="records", date_format="iso")

            # Update the test data in the database
            test = update(
                db=session, 
                model=Test, 
                object_id=test_id, 
                obj_in={
                    'computed_dataframe': new_df,
                    'vo2_max': round(graph['data'][1]['y'][pos_vo2_max], 2),
                    'plateau_dataframe': df_plateau
                }
            )

            return "/cubic_analysis", new_df, False, ""
        
        return no_update

    except Exception as e:
        logging.error(f"Error in change_analysis_step_time: {e}\n{traceback.format_exc()}")
        return no_update, no_update, True, "Une erreur s'est produite. Consultez les journaux pour plus de détails."
