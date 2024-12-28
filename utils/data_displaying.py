'''
This module handles the methods used for data visualization.

Features:
- Show the results values.
- Update the results with the checkboxes.
- Create the plots of the results page.
- Create the figure of the results page.
- Format the report values.
- Create the plot of the protocol for the report.
- Show the database data in the tabs of the database page.
'''

### Import ###

import pandas as pd
from datetime import date
from utils.database import get_db
from utils.crud import *
from utils.models import *
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import traceback
from utils.data_processing import *

### Methods ###

def show_values(results, vo2_max, fc_max) :
    """
    Manages the display of the results on the final page.

    This function processes the results of a physiological analysis and determines 
    the values for the thresholds based on available data. It handles cases where 
    the data may be incomplete or contain errors, and provides formatted information 
    for display.

    Parameters
    ----------
    results : list
        A list containing the results of the analysis. The structure of the list 
        must include values that correspond to the computed thresholds and other 
        physiological measurements.
    vo2_max : float
        The maximum volume of oxygen the subject can utilize during intense exercise, 
        used to calculate the percentage values for VO2.
    fc_max : float
        The maximum heart rate during exercise, used to calculate the percentage values for heart rate (FC).

    Returns
    -------
    tuple
        A tuple containing the following values:
        - info_s1 : str
            A string describing the data used for the first threshold.
        - info_s2 : str
            A string describing the data used for the second threshold.
        - t1_vo2_percent : float or None
            The percentage of VO2 for the first threshold, or None if not computed.
        - t1_vo2_value : float or str
            The actual VO2 value for the first threshold, or 'N/A' if not computed.
        - t1_fc_percent : float or None
            The percentage of heart rate (FC) for the first threshold, or None if not computed.
        - t1_fc_value : float or str
            The actual heart rate (FC) value for the first threshold, or 'N/A' if not computed.
        - t2_vo2_percent : float or None
            The percentage of VO2 for the second threshold, or None if not computed.
        - t2_vo2_value : float or str
            The actual VO2 value for the second threshold, or 'N/A' if not computed.
        - t2_fc_percent : float or None
            The percentage of heart rate (FC) for the second threshold, or None if not computed.
        - t2_fc_value : float or str
            The actual heart rate (FC) value for the second threshold, or 'N/A' if not computed.
    """

    # Define possible data for the curves
    curves = ["RER", "VCO2", "VE", "VE/VO2", "VE/VCO2", "PetCO2", "PetO2", "Lactate"]

    try:

        # Error during computation: no data used to compute thresholds
        if (len(results[-1]) == 0 and len(results[-2]) == 0) or (np.isnan(results[5]) and np.isnan(results[6])):

            info_s1 = "Non déterminé"
            info_s2 = "Non déterminé"
            t1_vo2_percent = None
            t1_vo2_value = "N/A"
            t1_fc_percent = None
            t1_fc_value = "N/A"
            t2_vo2_percent = None
            t2_vo2_value = "N/A"
            t2_fc_percent = None
            t2_fc_value = "N/A"

        # Only the second threshold has been computed
        elif (len(results[-2]) == 0 and len(results[-1]) != 0) or (np.isnan(results[5])) :

            # Get the selected curves and display the message
            inft = curves[results[-1][0]]
            for t in results[-1][1:] :
                inft += ", "+curves[t]

            info_s1 = "Non déterminé"
            info_s2 = f"Données utilisées : {inft}"
            t1_vo2_percent = None
            t1_vo2_value = "N/A"
            t1_fc_percent = None
            t1_fc_value = "N/A"
            t2_vo2_percent = round((results[6]*100/vo2_max), 2)
            t2_vo2_value = round(results[6], 2)
            t2_fc_percent = round((results[8]*100/fc_max), 2)
            t2_fc_value = round(results[8])

        # Only the first threshold has been computed
        elif (len(results[-2]) != 0 and len(results[-1]) == 0) or (np.isnan(results[6])) :

            # Get the selected curves and display the message
            inft = curves[results[-2][0]]
            for t in results[-2][1:] :
                inft += ", "+curves[t]

            info_s1 = f"Données utilisées : {inft}"
            info_s2 = "Non déterminé"
            t1_vo2_percent = round((results[5]*100/vo2_max), 2)
            t1_vo2_value = round(results[5], 2)
            t1_fc_percent = round((results[7]*100/fc_max), 2)
            t1_fc_value = round(results[7])
            t2_vo2_percent = None
            t2_vo2_value = "N/A"
            t2_fc_percent = None
            t2_fc_value = "N/A"
        
        # Two thresholds computed
        else :

            # Get the selected curves for both thresholds and display the message
            inft = curves[results[-2][0]]
            for t in results[-2][1:] :
                inft += ", "+curves[t]
            inft2 = curves[results[-1][0]]
            for t in results[-1][1:] :
                inft2 += ", "+curves[t]

            info_s1 = f"Données utilisées : {inft}"
            info_s2 = f"Données utilisées : {inft2}"
            t1_vo2_percent = round((results[5]*100/vo2_max), 2)
            t1_vo2_value = round(results[5], 2)
            t1_fc_percent = round((results[7]*100/fc_max), 2)
            t1_fc_value = round(results[7])
            t2_vo2_percent = round((results[6]*100/vo2_max), 2)
            t2_vo2_value = round(results[6], 2)
            t2_fc_percent = round((results[8]*100/fc_max), 2)
            t2_fc_value = round(results[8])

        return (
            info_s1, info_s2, t1_vo2_percent, t1_vo2_value, 
            t1_fc_percent, t1_fc_value, t2_vo2_percent, 
            t2_vo2_value, t2_fc_percent, t2_fc_value
        )
    
    except Exception as e:
        logging.error(f"Error in show_values: {e}\n{traceback.format_exc()}")
        raise

def checkbox_update(check_sv1, check_sv2, check_sl1, check_sl2, curves, results):
    """
    Updates the results based on the user's choice of thresholds (SV1, SV2, SL1, SL2).
    
    This function updates the `temp_results` array based on the user's selection of thresholds
    (SV1, SL1, SV2, SL2). It calculates the new values for the first and second threshold (VO2, FC)
    according to the selected curves, or removes them if no threshold is selected.

    Parameters
    ----------
    check_sv1 : bool
        Whether the user selected the first threshold (SV1).
    check_sv2 : bool
        Whether the user selected the second threshold (SV2).
    check_sl1 : bool
        Whether the user selected the first lactate threshold (SL1).
    check_sl2 : bool
        Whether the user selected the second lactate threshold (SL2).
    curves : dict
        Dictionary containing the 'FC' and 'VO2' curves with coefficients for each curve.
        Expected format:
            curves = {'FC': [[...], [...], [...]], 'VO2': [[...], [...], [...]]}
    results : list
        A list containing the current results, including VO2 and FC values at different points.

    Returns
    -------
    temp_results : list
        A list containing the updated results after applying the selected thresholds.
        Contains updated values for VO2, FC, and updated threshold indices.
    """
    try:

        # Make a copy of the results to avoid modifying the original
        temp_results = results.copy()

        # User's choice : SV1 threshold
        if check_sv1 and not check_sl1:
            temp_results[5] = temp_results[0]
            index_fc_curve = 0 if temp_results[5] < curves['FC'][0][1] else 1
            temp_results[7] = curves['FC'][1][index_fc_curve] + curves['FC'][2][index_fc_curve]*temp_results[5]
            temp_results[-2].remove(7)

        # User's choice : SL1 threshold
        if not check_sv1 and check_sl1 :
            temp_results[7] = temp_results[2][0]
            index_vo2_curve = 0 if temp_results[7] < curves['VO2'][0][1] else 1
            temp_results[5] = curves['VO2'][1][index_vo2_curve] + curves['VO2'][2][index_vo2_curve]*temp_results[7]
            temp_results[-2] = [7]
        
        # User's choice : no first threshold
        if not check_sv1 and not check_sl1 :
            temp_results[5] = np.nan
            temp_results[7] = np.nan
            temp_results[-2][:] = []
        
        # User's choice : SV2 threshold
        if check_sv2 and not check_sl2 :
            temp_results[6] = temp_results[1]
            index_fc_curve = 0 if temp_results[6] < curves['FC'][0][1] else 1
            temp_results[8] = curves['FC'][1][index_fc_curve] + curves['FC'][2][index_fc_curve]*temp_results[6]
            temp_results[-1].remove(7)

        # User's choice : SL2 threshold
        if not check_sv2 and check_sl2 :
            temp_results[8] = temp_results[2][1]
            index_vo2_curve = 0 if temp_results[8] < curves['VO2'][0][1] else 1
            temp_results[6] = curves['VO2'][1][index_vo2_curve] + curves['VO2'][2][index_vo2_curve]*temp_results[8]
            temp_results[-1] = [7]

        # User's choice : no second threshold
        if not check_sv2 and not check_sl2 :
            temp_results[6] = np.nan
            temp_results[8] = np.nan
            temp_results[-1][:] = []

        return temp_results
    
    except Exception as e:
        logging.error(f"Error in checkbox_update: {e}\n{traceback.format_exc()}")
        raise

def create_plot_results(fig: go.Figure, row, col, x, y, colors, labels_curve, x_label, y_labels, title, x_levels):
    """
    Creates a plot on a given subplot of a figure with primary and secondary y-axes.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure object to which the traces will be added.
    row : int
        The row index of the subplot where the plot will be created.
    col : int
        The column index of the subplot where the plot will be created.
    x : array-like
        The x-axis data values.
    y : list of array-like
        A list of y-axis data arrays. The first element corresponds to the primary y-axis, 
        and subsequent elements correspond to secondary y-axes.
    colors : list of str
        A list of colors for each curve and vertical threshold line.
    labels_curve : list of str
        A list of labels for each curve in the plot.
    x_label : str
        The label for the x-axis.
    y_labels : list of str
        A list of labels for the y-axes. The first label is for the primary y-axis, 
        and the second label is for the secondary y-axis.
    title : str
        The title for the plot (not currently used in the function but can be implemented).
    x_levels : list of float
        A list of x-axis values where vertical threshold lines will be drawn.

    Returns
    -------
    go.Figure
        The Plotly figure object with the added plot data.
    """
    try:
        # Set the text size
        axis_size=12
        ticks_size=8
        
        # Plot main data on primary y-axis
        fig.add_trace(
            go.Scatter(x=x, y=y[0], mode='lines+markers', name=labels_curve[0],
                    marker=dict(color=colors[0], size=4), line=dict(width=1)),
            row=row, col=col, secondary_y=False
        )
        
        # Add vertical threshold lines
        for i, level in enumerate(x_levels):
            fig.add_trace(
                go.Scatter(x=[level, level], y=[min(y[0]), max(y[0])],
                        mode='lines', name=f'Seuil {i + 1}',
                        line=dict(color='#58D68D' if i == 0 else '#D35400', dash='dashdot', width=1)),
                row=row, col=col, secondary_y=False
            )

        # Plot additional data on secondary y-axis
        for index in range(1, len(y)):
            fig.add_trace(
                go.Scatter(x=x, y=y[index], mode='lines+markers', name=labels_curve[index],
                        marker=dict(color=colors[index], size=4), line=dict(width=1)),
                row=row, col=col, secondary_y=True
            )

        # Update subplot x axes
        fig.update_xaxes(
            title_text=x_label, 
            title_font=dict(size=axis_size),
            title_standoff=5,
            row=row, col=col,
            tickvals=np.linspace(min(x), max(x), len(x), dtype=int) if x_label == "FC (bpm)" else np.arange(x.min()-1, x.max()+1, 1.0),
            tickfont=dict(size=ticks_size),
            showgrid=True
        )

        # Update subplot left y axes
        if labels_curve[0] in ("Lipides", "Glucides"):
            ticktext=np.round(np.linspace(min(y[0]), max(y[0]), 10), 2)
        else:
            ticktext=np.linspace(min(y[0]), max(y[0]), 10, dtype=int)
        fig.update_yaxes(
            title_text=f'<a style="color: {colors[0]};">{y_labels[0]}</a>', 
            title_font=dict(size=axis_size),
            row=row, col=col,
            title_standoff=5,
            tickvals=np.linspace(min(y[0]), max(y[0]), 10),
            ticktext=ticktext,
            tickfont=dict(size=ticks_size),
            secondary_y=False,
            showgrid=True
        )

        # Update subplot right axes
        sub_axis_titles = y_labels[1].split(',')
        y_axis_title = ','.join([f'<a style="color: {colors[i+1]};">{sub_axis_titles[i]}</a>' for i in range(len(sub_axis_titles))])
        fig.update_yaxes(
            title_text=y_axis_title, 
            title_font=dict(size=axis_size),
            title_standoff=5,
            row=row, col=col, secondary_y=True,
            tickvals=np.linspace(np.min(y[1:]), np.max(y[1:]), 10),
            ticktext=np.linspace(np.min(y[1:]), np.max(y[1:]), 10, dtype=int),
            tickfont=dict(size=ticks_size),
            showgrid=True
        )

        return fig
    
    except Exception as e:
        logging.error(f"Error in create_plot_results: {e}\n{traceback.format_exc()}")
        raise

def create_figure_results(df_lactate, levels, results, use_lactate):
    """
    Creates a figure with multiple subplots based on the given data and parameters.

    Parameters
    ----------
    df_lactate : pandas.DataFrame
        A DataFrame containing lactate and other physiological data such as heart rate 
        (FC), VO2, VCO2, and lactate levels.
    levels : dict
        A dictionary containing the physiological data for VO2, VCO2, VE, VE/VO2, 
        lipids, and glucose. The dictionary should also contain the x-axis levels 
        for the thresholds (x levels).
    results : list
        A list containing computed results for the thresholds, including lactate levels 
        and other physiological metrics that will be used to create vertical threshold lines.
    use_lactate : bool
        A boolean flag that determines whether to include a subplot for FC vs Lactate. 
        If `True`, a subplot for lactate data will be added.

    Returns
    -------
    go.Figure
        A Plotly figure object containing the generated plots with the data from `df_lactate`, 
        `levels`, and `results`.
    """
    try:

        # Create the main figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                [{"secondary_y": True}, {"secondary_y": True}]],
            vertical_spacing=0.11,
            horizontal_spacing=0.11,
            subplot_titles=(
                "FC vs VO2, VCO2",
                "FC vs VE, VE/VO2",
                "Lipides vs Glucides",
                "FC vs Lactate" if use_lactate else None
            )
        )

        # Define data and call `plotly_plot_results`
        fig = create_plot_results(fig, 1, 1, x=df_lactate.index,
                            y=[df_lactate['fc'], levels['vo2'], levels['vco2']],
                            colors=["#48C9B0", "#85C1E9", "#E59866"],
                            labels_curve=["FC", "VO2", "VCO2"],
                            x_label="Paliers",
                            y_labels=['FC (bpm)', "VO2, VCO2 (mL/min/kg)"],
                            title="FC vs VO2, VCO2",
                            x_levels=levels['x'])
        fig = create_plot_results(fig, 1, 2, x=df_lactate.index,
                            y=[levels['vevo2'], levels['ve'], df_lactate['fc']],
                            colors=["#D6C78A", "#E59866", "#48C9B0"],
                            labels_curve=["VE/VO2", "VE", "FC"],
                            x_label="Paliers",
                            y_labels=['VE/VO2', "VE (L/min), FC (bpm)"],
                            title="FC vs VE, VE/VO2",
                            x_levels=levels['x'])
        fig = create_plot_results(fig, 2, 1, x=df_lactate['fc'],
                            y=[levels['lip'], levels['glu']],
                            colors=["#85C1E9", "#E59866"],
                            labels_curve=["Lipides", "Glucides"],
                            x_label="FC (bpm)",
                            y_labels=["Lipides (g/min)", "Glucides (g/min)"],
                            title="Lipides vs Glucides",
                            x_levels=[results[7], results[8]])
        if use_lactate:
            fig = create_plot_results(fig, 2, 2, x=df_lactate.index,
                                y=[df_lactate['fc'], df_lactate['lactate']],
                                colors=["#48C9B0", "#E59866"],
                                labels_curve=["FC", "Lactate"],
                                x_label="Paliers",
                                y_labels=["FC (bpm)", "Lactate (mmol/L)"],
                                title="FC vs Lactate",
                                x_levels=results[3])

        # Update the title    
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=15)

        # Update global layout
        fig.update_layout(
            title_text=None,
            showlegend=False,
            margin=dict(l=50, r=5, t=25, b=35),
            template="plotly_white"
        )
            
        return fig

    except Exception as e:
        logging.error(f"Error in create_figure_results: {e}\n{traceback.format_exc()}")
        raise
        
def format_report_values(df_lactate, results, curves, levels, test_id, tps_last, use_lactate):
    """
    Formats and computes a set of report values based on lactate data, results, and physiological parameters.

    Parameters
    ----------
    df_lactate : pandas.DataFrame
        A DataFrame containing lactate test data including slopes, speeds, and power values 
        for each test level.
    results : list
        A list containing computed physiological metrics from the lactate test, including VO2, 
        heart rate (HR), power (watt), and other relevant values.
    curves : dict
        A dictionary containing the test curves for VO2, VCO2, and other metrics used in 
        the report calculations.
    levels : dict
        A dictionary containing the test levels and their corresponding physiological thresholds 
        such as VO2, lactate, RER, and other data points.
    test_id : int
        The unique identifier of the test used to fetch additional test-related data such as 
        VO2 max, HR max, and weight.
    tps_last : float
        The time spent at the last test level in minutes.
    use_lactate : bool
        A boolean flag indicating whether lactate data should be included in the report. 
        If `True`, lactate values are included in the report.

    Returns
    -------
    dict
        A dictionary containing the calculated report values, including VO2, VO2/kg, heart rate 
        (HR), VE, lactate, and other physiological metrics. Each key corresponds to a specific metric, 
        and the values are lists containing the calculated results for each level of the test.
    """
    try:
        # Get the test details
        session = next(get_db())
        test = get_by_id(session, Test, test_id)

        # Compute the values
        values = compute_values(curves, results, test.vo2_max)

        # Initialize reports
        reports = {}

        # Last level maintained time
        reports['tps_last'] = tps_last

        # Slope
        reports["slope"] = [
            df_lactate.loc[round(levels['x'][i]), "pente"]
            for i in range(2)
        ] + [df_lactate["pente"].iloc[-1]]

        # Speed
        reports["speed"] = [
            df_lactate.loc[round(levels['x'][i]), "vitesse"]
            for i in range(2)
        ] + [df_lactate["vitesse"].iloc[-1]]

        # VO2
        reports["vo2"] = [round(results[i+5] * 1000, 1) for i in range(2)] + [
            round(test.vo2_max * 1000, 1)
        ]

        # VO2/kg
        reports["vo2_kg"] = [
            round(results[i+5] * 1000 / test.weight, 1)
            for i in range(2)
        ] + [round(test.vo2_max * 1000 / test.weight, 1)]

        # %VO2max
        reports["vo2_ratio"] = [
            round(results[i+5] * 100 / test.vo2_max)
            for i in range(2)
        ] + [100]

        # VE
        reports["ve"] = [round(values['ve'][i], 1) for i in range(2)] + [
            round(test.ve_max, 1)
        ]

        # HR
        reports["hr"] = [round(results[7 + i]) for i in range(2)] + [
            round(test.hr_max)
        ]

        # %HRmax
        reports["hr_ratio"] = [
            round(results[7 + i] * 100 / test.hr_max)
            for i in range(2)
        ] + [100]

        # Watt
        reports["watt"] = [
            round(df_lactate.loc[round(levels['x'][i]), "puissance"])
            for i in range(2)
        ] + [round(df_lactate["puissance"].iloc[-1])]

        # Watt/kg
        reports["watt_kg"] = [
            round(df_lactate.loc[round(levels['x'][i]), "puissance"] / test.weight, 1)
            for i in range(2)
        ] + [round(df_lactate["puissance"].iloc[-1] / test.weight, 1)]
            
        # DE
        reports["de"] = [
            round((4.471 * results[i+5] + 0.55 * values['vco2'][i]) * 60)
            for i in range(2)
        ] + [round((4.471 * test.vo2_max + 0.55 * values['vco2'][-1]) * 60)]

        # Glucids and Lipids
        for key in ["glu", "lip"]:
            reports[key] = [round(values[key][i], 1) for i in range(3)]

        # %Glucids
        reports["glu_ratio"] = [
            max(0, min(round(((r - 0.71) / 0.29) * 100), 100))
            for r in levels['rer']
        ]

        # %Lipids
        reports["lip_ratio"] = [
            max(0, min(round(((1 - r) / 0.29) * 100), 100))
            for r in levels['rer']
        ]

        # Lactate
        if use_lactate:
            reports["lactate"] = [
                round(results[4][i], 1) for i in range(2)
            ] + [round(test.lactate_max, 1)]
        else:
            reports["lactate"] = [np.nan] * 3

        return reports

    except Exception as e:
        logging.error(f"Error in format_report_values: {e}\n{traceback.format_exc()}")
        raise

def create_plot_protocol(df_lactate) :
    """
    Creates a plot visualizing the protocol's evolution over time, including the speed and slope 
    values for each time point.

    Parameters
    ----------
    df_lactate : pandas.DataFrame
        A DataFrame containing lactate test data with columns for time (`temps`), speed (`vitesse`), 
        and slope (`pente`) values for each level of the test protocol.

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly `Figure` object containing the plot with the protocol's evolution. The plot includes:
        - Speed (`vitesse`) on the primary y-axis (left)
        - Slope (`pente`) on the secondary y-axis (right)
        The x-axis represents time in HH:MM:SS format, with lines for speed and slope shown over time.
    """
    try:
        # Create the figure
        fig = go.Figure()

        # Set the text size
        axis_size=12
        ticks_size=8

        # Transform the values for the axis : add a first row for the start
        time_axis = pd.concat([pd.DataFrame(["00:00:00"], columns = ['temps']), df_lactate['temps']]).reset_index(drop = True)
        x_axis = pd.to_datetime(time_axis['temps'], format = "%H:%M:%S")
        speed_axis = pd.concat([pd.DataFrame([min(df_lactate['vitesse']) - 1], columns = ['vitesse']), df_lactate['vitesse']]).reset_index(drop = True)
        slope_axis = pd.concat([pd.DataFrame([min(df_lactate['pente']) - 1], columns = ['pente']), df_lactate['pente']]).reset_index(drop = True)

        # Add speed trace
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=speed_axis["vitesse"],
                mode="lines+markers",
                name="Vitesse (km/h)",
                line=dict(color="#DC7633", shape="hv"),
                marker=dict(size=6),
            )
        )

        # Add slope trace
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=slope_axis["pente"],
                mode="lines+markers",
                name="Pente (%)",
                line=dict(color="#3498DB", shape="hv"),
                marker=dict(size=6),
                yaxis="y2", 
            )
        )

        # Update layout
        fig.update_layout(
            title=dict(
                text="Évolution des paliers du protocole",
                font=dict(size=15),
                x=0.5,
            ),
            xaxis=dict(
                title="Temps (HH:MM:SS)",
                titlefont=dict(size=axis_size),
                tickfont=dict(size=ticks_size),
                tickvals=pd.date_range(x_axis.min(), x_axis.max(), periods = round(len(x_axis)/2)),
                tickformat="%H:%M:%S",
                showgrid=True,
            ),
            yaxis=dict(
                title="Vitesse (km/h)",
                titlefont=dict(size=axis_size),
                tickfont=dict(size=ticks_size),
                tickvals=np.round(np.linspace(min(speed_axis['vitesse']), max(speed_axis['vitesse']), max(len(speed_axis['vitesse']), len(slope_axis['pente']))), 2),
                ticktext=np.round(np.linspace(min(speed_axis['vitesse']), max(speed_axis['vitesse']), max(len(speed_axis['vitesse']), len(slope_axis['pente']))), 2),
                showgrid=True,
            ),
            yaxis2=dict(
                title="Pente (%)",
                titlefont=dict(size=axis_size),
                tickfont=dict(size=ticks_size),
                tickvals=np.round(np.linspace(min(slope_axis['pente']), max(slope_axis['pente']), max(len(speed_axis['vitesse']), len(slope_axis['pente']))), 2),
                ticktext=np.round(np.linspace(min(slope_axis['pente']), max(slope_axis['pente']), max(len(speed_axis['vitesse']), len(slope_axis['pente']))), 2),
                overlaying="y",  # Overlay on the first y-axis
                side="right",
            ),
            legend=dict(
                font=dict(size=10),
                yanchor="bottom",
                y=0.05,
                xanchor="right",
                x=0.95,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="LightGray",
                borderwidth=1,
            ),
            template="plotly_white",
            margin=dict(l=65, r=15, t=50, b=40)
        )

        # Return the figure for visualization
        return fig
    
    except Exception as e:
        logging.error(f"Error in create_plot_protocol: {e}\n{traceback.format_exc()}")
        raise

def show_database(active_tab):
    """
    Retrieves and formats the data for displaying in a dashboard, based on the selected tab. 

    Depending on the active tab, this function returns information about athletes, tests, or teams 
    from the database, including data formatting for display in a table.

    Parameters
    ----------
    active_tab : str
        The currently active tab in the dashboard. Can be one of the following:
        - "tab-athletes": Displays information about athletes.
        - "tab-tests": Displays information about tests.
        - "tab-teams": Displays information about teams.

    Returns
    -------
    data : list of dict
        A list of dictionaries containing the data to be displayed in the table. The structure of the data 
        depends on the active tab:
        - "tab-athletes": Contains athlete details (ID, name, birthdate, sex, team, weight, height).
        - "tab-tests": Contains test details (test ID, athlete's name, test date, weight, download, delete).
        - "tab-teams": Contains team details (team name, number of athletes, download, delete).
    columns : list of dict
        A list of dictionaries representing the column names for the table. Each dictionary contains 
        the 'name' and 'id' for each column.
    row_selectable : bool
        If the rows are selectable. This is set to `True` for the "tab-athletes" tab and `False` for 
        "tab-tests" and "tab-teams".
    """
    try:
        # Retrieve the database session
        session = next(get_db())

        # Data type selection
        if active_tab == "tab-athletes":
            row_selectable="single"
            athletes = get_all(session, Athlete)
            col_names = [
                "ID",
                "Nom",
                "Prénom",
                "Date de naissance",
                "Sexe",
                "Équipe",
                "Poids (kg)",
                "Taille (cm)",
            ]
            data = [
                {
                    "ID": athlete.id,
                    "Nom": athlete.last_name.capitalize(),
                    "Prénom": athlete.first_name.capitalize(),
                    "Date de naissance": date.strftime(athlete.date_of_birth, "%d/%m/%Y"),
                    "Sexe": athlete.gender.upper(),
                    "Équipe": athlete.team_name.capitalize() if athlete.team_name else athlete.team_name,
                    "Poids (kg)": athlete.weight,
                    "Taille (cm)": athlete.height,
                }
                for athlete in athletes
            ]
            
        elif active_tab == "tab-tests":
            row_selectable=False
            tests = get_all(session, Test)
            col_names = [
                "ID Test",
                "Nom",
                "Prénom",
                "Date",
                "Poids",
                "Télécharger",
                "Supprimer"
            ]
            data = [
                {
                    "ID Test": test.id,
                    "Nom": test.athlete.last_name.capitalize(),
                    "Prénom": test.athlete.first_name.capitalize(),
                    "Date": date.strftime(test.date, "%d/%m/%Y"),
                    "Poids": test.weight,
                    "Télécharger": "📥",
                    "Supprimer": "🗑"
                }
                for test in tests
            ]

        elif active_tab == "tab-teams":
            row_selectable=False
            teams = get_all(session, Team)
            col_names = [
                "Nom de l'équipe",
                "Nombre d'athlètes",
                "Télécharger",
                "Supprimer"
            ]
            data = [
                {
                    "Nom de l'équipe": team.name.capitalize(),
                    "Nombre d'athlètes": len(team.athletes),
                    "Télécharger": "📥",
                    "Supprimer": "🗑"
                }
                for team in teams
            ]

        else:
            return [], [], True

        columns = [{"name": col, "id": col} for col in col_names]

        return data, columns, row_selectable
    
    except Exception as e:
        logging.error(f"Error in show_database: {e}\n{traceback.format_exc()}")
        raise