'''
This module handles the methods used for graph updating.

Features:
- Update the graphs, plots, sliders and vertical bars.

'''

### Imports ###

import logging
from utils.data_processing import *
from dash import dcc, html
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def update_graph_data_time(graph_data, df):
    """ Update the graph data with new time and rolling average values.

    Parameters
    ----------
    graph_data : dict
        A dictionary representing the current data of the graph.
    df : pandas.DataFrame
        DataFrame containing 't' (time) and 'VO2' (the VO2 measurements).
    
    Returns
    -------
    dict
        The updated graph_data dictionary with new x and y values for the graph traces.
    """
    try:
        
        # Update the graph data with the new values from the DataFrame
        graph_data['data'][0]['x'] = df['t']
        graph_data['data'][0]['y'] = df['VO2']
        graph_data['data'][1]['x'] = df['t']
        graph_data['data'][1]['y'] = calculate_rolling_average(df)

        return graph_data
    
    except Exception as e:
        logging.error(f"Error in update_graph_data_time: {e}\n{traceback.format_exc()}")
        raise

def create_slider(axis, value, id_slider, n, time=False, disabled=False, step=1):
    """ Create a slider widget with custom marks for a given axis.

    Parameters
    ----------
    axis : pandas.Series or list
        The data axis (e.g., time or value) to be used for the slider.
    value : int
        The initial value (position) of the slider.
    id_slider : str
        The unique identifier for the slider.
    n : int
        The number of marks to display on the slider.
    time : bool, optional
        Whether the axis represents time (default is False).
    disabled : bool, optional
        Whether the slider is disabled (default is False).
    step : int, optional
        The step size for the slider (default is 1).

    Returns
    -------
    dcc.Slider
        A Dash Slider component with the specified properties.
    """
    try:
        # Generate marks for the slider
        range_axis = np.linspace(0, len(axis)-1, n, dtype=int)
        if time:
            marks_slider = {int(i): str(axis.iloc[i].strftime("%-Mmin%-Ss")) for i in range_axis}
        else:
            marks_slider = {int(i): str(round(axis[i], 2)) for i in range_axis}
        
        # Create the slider
        slider = dcc.Slider(
            id=id_slider,
            min=0, 
            max=len(axis)-1, 
            step=step, 
            marks=marks_slider,
            value=value if value else 0,
            disabled=disabled
        )
        return slider
    
    except Exception as e:
        logging.error(f"Error in create_slider: {e}\n{traceback.format_exc()}")
        raise

def update_figure_modelization(df, figure, n_outliers, pos_left, pos_right, n):
    """Update the figure with modelized data and adjust the outliers' visualization.

    Parameters
    ----------
    df : pandas.DataFrame
        The main dataframe containing time (`t`) and `VO2` data.
    figure : dict
        The figure (likely a Plotly figure) containing traces to be updated.
    n_outliers : int
        The number of outliers to include in the modelization.
    pos_left : float
        The left position of the data range to be considered.
    pos_right : float
        The right position of the data range to be considered.
    n : int
        The parameter for the cubic-linear modelization.

    Returns
    -------
    tuple
        Updated figure and outliers dataframe (`df_outliers`).
    """
    try:
        # Update the main figure with the new data from modelization
        figure['data'][1]['x'] = df['t']
        figure['data'][1]['y'], df_outliers = modelization_cubic_linear(df, n_outliers, n)

        # Get the indices of the outliers within the specified range
        limit_left = df_outliers['t'].searchsorted(pos_left)
        limit_right = df_outliers['t'].searchsorted(pos_right)

        # Update the outliers plot with the filtered data
        figure['data'][2]['x'] = df_outliers['t'].iloc[limit_left:limit_right]
        figure['data'][2]['y'] = df_outliers['VO2'].iloc[limit_left:limit_right]

        return figure, df_outliers
    
    except Exception as e:
        logging.error(f"Error in update_figure_modelization: {e}\n{traceback.format_exc()}")
        raise

def create_protect_bars(df):
    """Create protective boundary bars for a Plotly figure.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing time (`t`) values.

    Returns
    -------
    list
        A list of two dictionary elements representing left and right boundary lines.
    """
    return [
        {
            'line': {'color': 'black', 'dash': 'dash', 'width': 2}, 
            'name': 'left-boundary', 
            'type': 'line', 
            'x0': df['t'].iloc[0], 
            'x1': df['t'].iloc[0], 
            'xref': 'x', 
            'y0': 0, 
            'y1': 1, 
            'yref': 'y domain'
        },
        {
            'line': {'color': 'black', 'dash': 'dash', 'width': 2}, 
            'name': 'right-boundary', 
            'type': 'line', 
            'x0': df['t'].iloc[-1], 
            'x1': df['t'].iloc[-1], 
            'xref': 'x', 
            'y0': 0, 
            'y1': 1, 
            'yref': 'y domain'
        }
    ]

def plot_analysis(df, step):
    """Create a plot of VO2 data with cubic or linear modelizations and outliers.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing time (`t`) and VO2 (`VO2`) data.
    step : str
        The analysis step (`'time'`, `'cubic'`, or `'linear'`).

    Returns
    -------
    dict
        A Plotly figure containing the data and modelizations.
    pandas.DataFrame, optional
        A dataframe containing outliers if applicable.
    """
    try:
        # Create the elements regarding the step
        if step == 'time':
            y_model = calculate_rolling_average(df)
            name_model = "Modélisation cubique"

        elif step == "cubic":
            y_model, df_outliers = modelization_cubic_linear(df, 100, 0)
            name_model = "Modélisation cubique"

        elif step == "linear":
            y_model, df_outliers = modelization_cubic_linear(df, 100, 2)
            name_model = "Modélisation linéaire"

        # Scatter plot of the VO2 data
        fig = {
            'data': [
                {
                    'x': df['t'], 
                    'y': df['VO2'], 
                    'mode': 'markers',
                    'name': 'VO2', 
                    'marker': dict(color='#3498DB')
                },
                {
                    'x': df['t'], 
                    'y': y_model, 
                    'mode': 'lines', 
                    'name': name_model, 
                    'line': dict(color='#F1C40F', width=3)
                },
            ],
            'layout': {
                'title': {
                    'text': 'VO2 en fonction du temps',
                    'x': 0.5
                },
                'xaxis': {
                    'title': 'Temps'
                },
                'yaxis': {
                    'title': 'VO2 (L/min)'
                },
                'legend': {
                    'x': 0,
                    'y': 1,
                    'xanchor': 'left',
                    'yanchor': 'top',
                    'bgcolor': 'white',
                    'bordercolor': 'black',
                    'borderwidth': 1
                }
            }
        }

        # Outliers plot
        if step != "time":
            plot_outliers = {
                'x': df_outliers['t'], 
                'y': df_outliers['VO2'], 
                'mode': 'markers',
                'name': 'Outliers', 
                'marker': dict(color='#DC7633')
            }
            fig["data"].append(plot_outliers)

            return fig, df_outliers
        
        return fig
    
    except Exception as e:
        logging.error(f"Error in plot_analysis: {e}\n{traceback.format_exc()}")
        raise

def plot_modelizations(df_vo2, df_lactate=None):
    """
    Generate plots for various modelizations based on VO2 and optionally lactate data.

    Parameters
    ----------
    df_vo2 : pandas.DataFrame
        The dataframe containing VO2 and other physiological data.
    df_lactate : pandas.DataFrame, optional
        The dataframe containing lactate data. Default is None.

    Returns
    -------
    tuple
        A tuple containing the children components for the graph (including the plot),
        the models dictionary, and any error related to RER if it occurred.
    """
    try:
        # Retrieve modelizations and any error related to RER
        models, error = final_modelizations(df_vo2, df_lactate)

        # Determine the number of rows for the subplots
        if len(models) < 9:
            n_rows = 3
            subplot_titles = ['RER', 'VCO2', 'VE', 'VE/VO2', 'VE/VCO2', 'Lactate']
        else:
            n_rows = 4
            subplot_titles = ['RER', 'VCO2', 'VE', 'PetCO2', 'PetO2', 'VE/VO2', 'VE/VCO2', 'Lactate']

        # Update subplot titles if no lactate data is provided
        if df_lactate is None:
            subplot_titles.remove("Lactate")

        # Create subplots for the graphs
        fig = make_subplots(
            rows=n_rows, 
            cols=2, 
            subplot_titles=subplot_titles,
            vertical_spacing=0.05
        )

        # Add traces for each model
        for r in range(n_rows):
            for i in range(2):
                name = list(models.keys())[i+(2*r)]

                # Plot lactate data if available
                if name == "Lactate":
                    fig.add_trace(
                        go.Scatter(
                            x=df_lactate.index, 
                            y=models[name],
                            mode='lines+markers',
                            name=name,
                            line=dict(color='#DC7633'),
                            marker=dict(color='#DC7633')
                        ), 
                        row=r+1, 
                        col=i+1
                    )
                    fig.update_xaxes(title_text="Paliers", row=r+1, col=i+1)
                    fig.update_yaxes(title_text="Lactate (mmol/L)", row=r+1, col=i+1)

                # Plot other physiological data with modelizations
                elif name in ['RER', 'VCO2', 'VE', 'PetCO2', 'PetO2', 'VE/VO2', 'VE/VCO2']:
                    fig.add_trace(
                        go.Scatter(
                            x=df_vo2['VO2'], 
                            y=df_vo2[name],
                            mode='markers',
                            name=name,
                            marker=dict(color='#3498DB')
                        ), 
                        row=r+1, 
                        col=i+1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=models[name][0], 
                            y=models[name][1],
                            mode='lines',
                            name="Modélisation linéaire",
                            line=dict(color='#F1C40F', width=3)
                        ), 
                        row=r+1, 
                        col=i+1
                    )
                    fig.update_xaxes(title_text="VO2 (L/min)", row=r+1, col=i+1)
                    fig.update_yaxes(title_text=f"{name} (L/min)" if name != 'RER' else name, row=r+1, col=i+1)

        # Add vertical line for RER if there was an error
        if error:
            fig.add_shape(
                line=dict(color='gray', dash='dash', width=2),
                name="error-RER",
                type="line",
                x0=error,
                x1=error,
                xref="x",
                y0=0,
                y1=1,
                yref="y domain",
                row=1,
                col=1,
            )

        # Update the layout of the figure
        fig.update_layout(
            title_text="Modélisations des données en fonction de la VO2",
            title_x=0.5,
            showlegend=False,
            height=2000
        )

        # Create the RER switch component
        children_graph = [
            html.Div(
                id="div-switch-rer",
                hidden=True,
                style={"textAlign": "center", "alignItems": "center"},
                children=[
                    html.B(
                        html.Label(
                            "Suppression de l'erreur de début de test",
                            style={"margin-bottom": "10px"}
                        )
                    ),
                    dbc.Switch(
                        id='switch-rer',
                        label="Désactivé",
                        value=False,
                        input_style={
                            "transform": "scale(1.5)",
                            "border-radius": "10px",
                            "position": "absolute",
                        },
                        label_style={
                            "margin-left": "10px"
                        }
                    )
                ]
            )
        ]

        # Append the graph to the children list
        children_graph.append(
            dcc.Graph(figure=fig)
        )
        
        return children_graph, models, error

    except Exception as e:
        logging.error(f"Error in plot_modelizations: {e}\n{traceback.format_exc()}")
        raise
    
def plot_results(models, results, df_vo2):
    """
    Generate plots for various modelizations and physiological results.

    Parameters
    ----------
    models : dict
        A dictionary containing modelization results for physiological data (e.g., RER, VCO2, VE, etc.).
    results : list
        A list containing specific results for thresholds and model validations.
    df_vo2 : pandas.DataFrame
        The dataframe containing VO2 and other physiological data.

    Returns
    -------
    plotly.graph_objs.Figure
        A Plotly figure containing the generated plots for the modelizations and thresholds.
    """
    try:
        # Determine the number of columns and subplot titles
        if len(models) < 9:
            n_cols = 3
            subplot_titles = ['RER', 'VCO2', 'VE', 'VE/VO2', 'VE/VCO2', 'Lactate']
        else:
            n_cols = 4
            subplot_titles = ['RER', 'VCO2', 'VE', 'PetCO2', 'PetO2', 'VE/VO2', 'VE/VCO2', 'Lactate']

        # Update subplot titles if lactate is not in the models
        if "Lactate" not in models:
            subplot_titles.remove("Lactate")

        # Create subplots
        fig = make_subplots(
            rows=2, 
            cols=n_cols, 
            subplot_titles=subplot_titles,
            vertical_spacing=0.1
        )

        # Add traces for each model
        for r in range(2):
            for i in range(n_cols):
                name = list(models.keys())[i+(n_cols*r)]

                # Handle Lactate data case
                if name == "Lactate":
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(len(models[name]))), 
                            y=models[name],
                            mode='lines+markers',
                            name=name,
                            line=dict(color='#DC7633'),
                            marker=dict(color='#DC7633')
                        ), 
                        row=r+1, 
                        col=i+1
                    )
                    fig.update_xaxes(title_text="Paliers", row=r+1, col=i+1)
                    fig.update_yaxes(title_text="Lactate (mmol/L)", row=r+1, col=i+1)

                    # Add vertical lines for lactate thresholds from results
                    fig.add_shape(
                        line=dict(color='red', dash='dash', width=2),
                        name="sl1",
                        type="line",
                        x0=results[3][0],
                        x1=results[3][0],
                        xref="x",
                        y0=0,
                        y1=1,
                        yref="y domain",
                        row=r+1,
                        col=i+1,
                    )
                    fig.add_shape(
                        line=dict(color='orange', dash='dash', width=2),
                        name="sl2",
                        type="line",
                        x0=results[3][1],
                        x1=results[3][1],
                        xref="x",
                        y0=0,
                        y1=1,
                        yref="y domain",
                        row=r+1,
                        col=i+1,
                    )

                # Handle other physiological data
                elif name in ['RER', 'VCO2', 'VE', 'PetCO2', 'PetO2', 'VE/VO2', 'VE/VCO2']:
                    fig.add_trace(
                        go.Scatter(
                            x=df_vo2['VO2'], 
                            y=df_vo2[name],
                            mode='markers',
                            name=name,
                            marker=dict(color='#3498DB')
                        ), 
                        row=r+1, 
                        col=i+1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=models[name][0], 
                            y=models[name][1],
                            mode='lines',
                            name="Modélisation linéaire",
                            line=dict(color='#F1C40F', width=3)
                        ), 
                        row=r+1, 
                        col=i+1
                    )
                    fig.update_xaxes(title_text="VO2 (L/min)", row=r+1, col=i+1)
                    fig.update_yaxes(title_text=f"{name} (L/min)" if name != 'RER' else name, row=r+1, col=i+1)
                
                    # Add vertical lines for respiratory thresholds
                    if any([not np.isnan(results[0]), not np.isnan(results[1])]):
                        if not np.isnan(results[0]):
                            fig.add_shape(
                                line=dict(color='red', dash='dash', width=2),
                                name="sv1",
                                type="line",
                                x0=results[0],
                                x1=results[0],
                                xref="x",
                                y0=0,
                                y1=1,
                                yref="y domain",
                                row=r+1,
                                col=i+1,
                            )
                        if not np.isnan(results[1]):
                            fig.add_shape(
                                line=dict(color='orange', dash='dash', width=2),
                                name="sv2",
                                type="line",
                                x0=results[1],
                                x1=results[1],
                                xref="x",
                                y0=0,
                                y1=1,
                                yref="y domain",
                                row=r+1,
                                col=i+1,
                            )

        # Update layout with general settings
        fig.update_layout(
            title_text=None,
            showlegend=False,
            height=800,
            margin=dict(r=0, t=40, l=50, b=40)
        )
        
        return fig

    except Exception as e:
        logging.error(f"Error in plot_results : {e}\n{traceback.format_exc()}")
        raise

def update_graph_shapes(graph, value, line_name):
    """
    Update the position of a specific line shape in the graph layout.

    Parameters
    ----------
    graph : dict
        The Plotly graph object, typically containing the layout and shapes.
    value : float
        The new x-position for the line shape to be updated.
    line_name : str
        The name of the line shape to update.

    Returns
    -------
    dict
        The updated graph object with the modified line shape.
    """
    try:
        # Retrieve existing shapes in the graph
        shapes = graph.get('layout', {}).get('shapes', [])

        # Iterate through the shapes to find the matching line shape
        for index, shape in enumerate(shapes):
            if (
                shape.get('type') == 'line' and 
                shape.get('x0') == shape.get('x1') and 
                shape.get('name') == line_name
            ):  
                # Update the x-position of the line shape
                graph['layout']['shapes'][index]['x0'] = value
                graph['layout']['shapes'][index]['x1'] = value

        return graph
    
    except Exception as e:
        logging.error(f"Error in update_graph_shapes: {e}\n{traceback.format_exc()}")
        raise