"""
This module handles the callbacks related to the results page of the analysis.

Features :
- Results visualization and modification.
- PDF display, modification and export.

"""

### Imports ###

from dash import Input, Output, State, callback, callback_context, no_update
from utils.crud import *
from utils.data_processing import *
from utils.graph_updating import *
from utils.data_displaying import *
from utils.pdf_processing import *
import logging
import traceback
import re
from io import BytesIO

### Callbacks ###

# --- SECTION 1 : Callback related to the results visualization ---
@callback(
    Output("athlete-name", "children"),
    Output("test-date", "children"),
    Output("athlete-height", "children"),
    Output("results-vo2-max", "children"),
    Output("athlete-firstname", "children"),
    Output("athlete-birth-date", "children"),
    Output("athlete-weight", "children"),
    Output("results-fc-max", "children"),
    Output("remarks-table", "data"),
    Output("used-curves-s1", "children"),
    Output("used-curves-s2", "children"),
    Output("seuil1-vo2-percent", "value"),
    Output("seuil1-vo2-percent", "label"),
    Output("seuil1-vo2", "children"),
    Output("seuil1-fc-percent", "value"),
    Output("seuil1-fc-percent", "label"),
    Output("seuil1-fc", "children"),
    Output("seuil2-vo2-percent", "value"),
    Output("seuil2-vo2-percent", "label"),
    Output("seuil2-vo2", "children"),
    Output("seuil2-fc-percent", "value"),
    Output("seuil2-fc-percent", "label"),
    Output("seuil2-fc", "children"),
    Output('div-slider-sv1', "children"),
    Output('div-slider-sv2', "children"),
    Output('div-slider-sl1', "children"),
    Output('div-slider-sl2', "children"),
    Output("check-sv1", "label"),
    Output("check-sv2", "label"),
    Output("check-sl1", "label"),
    Output("check-sl1", "disabled"),
    Output("check-sl2", "label"),
    Output("check-sl2", "disabled"),
    Output('graph-results', "figure"),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input('div-results', "children"),
    State('results', "data"),
    State('test', 'data'),
    State("final_models", "data"),
    State("curves", "data"),
    State('use_lactate', 'data'),
    prevent_initial_call='initial_duplicate'
)
def display_results(div, results, test_id, models, curves, bool_lactate):
    """
    Generate and display analysis results for a given athlete and test.

    Parameters
    ----------
    div : Any
        Placeholder input to trigger callback execution.
    results : dict
        Analysis results containing VO2 max, heart rate, and thresholds.
    test_id : int
        ID of the test being analyzed.
    models : dict
        Data from the final models used for analysis.
    curves : dict
        Curves data for plotting and analysis.
    bool_lactate : bool
        Whether lactate data is being used in the analysis.

    Returns
    -------
    tuple
        A tuple of values for updating the Dash interface, including:
        - Athlete and test information.
        - Analysis results and remarks.
        - Configured sliders and checkboxes.
        - Graphs and error messages (if applicable).
    """
    try:

        # Retrieve athlete and test data from the database
        session = next(get_db())
        test = get_by_id(session, Test, test_id)
        athlete = get_by_id(session, Athlete, test.athlete_id)

        # Load remarks data
        df_lactate = load_dataframe_from_id(test_id, lactate=True)
        df_remarks = pd.DataFrame(columns=["Paliers", "Remarques"])
        df_remarks["Remarques"] = df_lactate[["remarques"]].replace("", np.nan).dropna()
        df_remarks["Paliers"] = [i for i in list(df_remarks.index)]

        # Retrieve maximum values
        vo2_max = test.vo2_max
        fc_max = test.hr_max

        # Process threshold values
        threshold_data = show_values(results, vo2_max, fc_max)

        # Load VO2 data
        df_vo2 = load_dataframe_vo2(test.computed_dataframe)
        sorted_vo2 = np.array(sorted(df_vo2["VO2"]))

        # Create respiratory sliders
        if not np.isnan(results[0]):
            slider_sv1 = create_slider(sorted_vo2, sorted_vo2.searchsorted(results[0]), "slider-sv1", 5)
        else:
            slider_sv1 = create_slider(sorted_vo2, sorted_vo2[-1], "slider-sv1", 5)
        if not np.isnan(results[1]):
            slider_sv2 = create_slider(sorted_vo2, sorted_vo2.searchsorted(results[1]), "slider-sv2", 5)
        else:
            slider_sv2 = create_slider(sorted_vo2, sorted_vo2[0], "slider-sv2", 5)
        
        # Create lactate sliders
        if bool_lactate:
            slider_sl1 = create_slider(df_lactate.index, results[3][0], "slider-sl1", 5, step=0.1)
            slider_sl2 = create_slider(df_lactate.index, results[3][1], "slider-sl2", 5, step=0.1)
            check_sl1, check_sl2 = False, False
        else:
            slider_sl1 = create_slider(range(10), 0, "slider-sl1", 5, disabled=True)
            slider_sl2 = create_slider(range(10), 0, "slider-sl2", 5, disabled=True)
            check_sl1, check_sl2 = True, True

        # Configure checkbox labels
        label_sv1 = compute_label_checkbox(results[0], curves["FC"]) if not np.isnan(results[0]) else "N/A"
        label_sv2 = compute_label_checkbox(results[1], curves["FC"]) if not np.isnan(results[1]) else "N/A"
        label_sl1 = f"{round(results[2][0])} bpm" if bool_lactate else "N/A"
        label_sl2 = f"{round(results[2][1])} bpm" if bool_lactate else "N/A"
        
        # Generate graphs for the results
        graphs = plot_results(models, results, df_vo2)

        return (
            athlete.last_name.capitalize(),
            date.strftime(test.date, "%d/%m/%Y"),
            athlete.height,
            round(vo2_max*1000/athlete.weight, 2),
            athlete.first_name.capitalize(),
            date.strftime(athlete.date_of_birth, "%d/%m/%Y"),
            athlete.weight,
            round(fc_max),
            df_remarks.to_dict(orient="records"),
            threshold_data[0],
            threshold_data[1],
            threshold_data[2],
            f"{threshold_data[2]} %",
            threshold_data[3],
            threshold_data[4],
            f"{threshold_data[4]} %",
            threshold_data[5],
            threshold_data[6],
            f"{threshold_data[6]} %",
            threshold_data[7],
            threshold_data[8],
            f"{threshold_data[8]} %",
            threshold_data[9],
            slider_sv1, 
            slider_sv2, 
            slider_sl1,
            slider_sl2,
            label_sv1, 
            label_sv2, 
            label_sl1, 
            check_sl1,
            label_sl2,
            check_sl2,
            graphs,
            False, 
            ""
        )

    except Exception as e:
        logging.error(f"Error in display_results : {e}\n{traceback.format_exc()}")
        return (
            no_update, no_update, no_update, no_update, no_update,
            no_update, no_update, no_update, no_update, no_update,
            no_update, no_update, no_update, no_update, no_update,
            no_update, no_update, no_update, no_update, no_update,
            no_update, no_update, no_update, no_update, no_update,
            no_update, no_update, no_update, no_update, no_update,
            no_update, no_update, no_update, no_update, True,
            "Une erreur s'est produite. Consultez les journaux pour plus de détails."
        )

# --- SECTION 2 : Callback related to the update of the threshold positions ---
@callback(
    Output("graph-results", "figure", allow_duplicate=True),
    Output("results", "data", allow_duplicate=True),
    Output("check-sv1", "label", allow_duplicate=True),
    Output("check-sv2", "label", allow_duplicate=True),
    Output("check-sl1", "label", allow_duplicate=True),
    Output("check-sl2", "label", allow_duplicate=True),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input("slider-sv1", "value"),
    Input("slider-sv2", "value"),
    Input("slider-sl1", "value"),
    Input("slider-sl2", "value"),
    State("graph-results", "figure"),
    State('test', 'data'),
    State("curves", "data"),
    State("results", "data"),
    State('use_lactate', 'data'),
    prevent_initial_call=True
)
def update_threshold_position(slider_sv1, slider_sv2, slider_sl1, slider_sl2, graphs, test_id, curves, results, bool_lactate):
    """
    Update threshold positions based on slider values and refresh related graphs and labels.

    Parameters
    ----------
    slider_sv1 : int
        Value of the first respiratory slider.
    slider_sv2 : int
        Value of the second respiratory slider.
    slider_sl1 : int
        Value of the first lactate slider.
    slider_sl2 : int
        Value of the second lactate slider.
    graphs : dict
        Current state of the results graph.
    test_id : int
        ID of the test being analyzed.
    curves : dict
        Curves data for plotting and computation.
    results : dict
        Current state of the analysis results.
    bool_lactate : bool
        Indicator if lactate data is used in analysis.

    Returns
    -------
    tuple
        Updated graph, results, checkbox labels, and error messages if applicable.
    """
    try:
        # Detect which input triggered the callback
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        # Load the required data
        df_vo2 = load_dataframe_from_id(test_id)
        df_lactate = load_dataframe_from_id(test_id, lactate=True) if bool_lactate else None
        sorted_vo2 = np.array(sorted(df_vo2["VO2"]))

        # Map slider values to their respective sliders
        threshold_values = {
            "slider-sv1": slider_sv1,
            "slider-sv2": slider_sv2,
            "slider-sl1": slider_sl1,
            "slider-sl2": slider_sl2
        }

        # Update respiratory graphs based on slider input
        if trigger in ["slider-sv1", "slider-sv2"]:
            threshold_level = 0 if trigger == "slider-sv1" else 1
            results, label = update_threshold_computation(
                sorted_vo2[threshold_values[trigger]], 
                curves,
                results,
                "respiratory",
                threshold_level,
                df_lactate
            )
            graphs = update_graph_shapes(graphs, sorted_vo2[threshold_values[trigger]], trigger[-3:])
        
        # Update lactate graphs based on slider input
        elif trigger in ["slider-sl1", "slider-sl2"]:
            threshold_level = 0 if trigger == "slider-sl1" else 1
            results, label = update_threshold_computation(
                threshold_values[trigger], 
                curves,
                results,
                "lactate",
                threshold_level,
                df_lactate
            )
            graphs = update_graph_shapes(graphs, threshold_values[trigger], trigger[-3:])

        # Update checkbox labels dynamically
        checkbox_labels = [no_update for i in range(4)]
        checkbox_labels[list(threshold_values).index(trigger)] = label

        return (
            graphs,
            results,
            checkbox_labels[0],
            checkbox_labels[1],
            checkbox_labels[2],
            checkbox_labels[3],
            False,
            ""
        )
    
    except Exception as e:
        logging.error(f"Error in update_threshold_position : {e}\n{traceback.format_exc()}")
        return (
            no_update, no_update, no_update, no_update, no_update,
            no_update, True,
            "Une erreur s'est produite. Consultez les journaux pour plus de détails."
        )

# --- SECTION 3 : Callback related to the update of the results with the checkboxes ---
@callback(
    Output("used-curves-s1", "children", allow_duplicate=True),
    Output("used-curves-s2", "children", allow_duplicate=True),
    Output("seuil1-vo2-percent", "value", allow_duplicate=True),
    Output("seuil1-vo2-percent", "label", allow_duplicate=True),
    Output("seuil1-vo2", "children", allow_duplicate=True),
    Output("seuil1-fc-percent", "value", allow_duplicate=True),
    Output("seuil1-fc-percent", "label", allow_duplicate=True),
    Output("seuil1-fc", "children", allow_duplicate=True),
    Output("seuil2-vo2-percent", "value", allow_duplicate=True),
    Output("seuil2-vo2-percent", "label", allow_duplicate=True),
    Output("seuil2-vo2", "children", allow_duplicate=True),
    Output("seuil2-fc-percent", "value", allow_duplicate=True),
    Output("seuil2-fc-percent", "label", allow_duplicate=True),
    Output("seuil2-fc", "children", allow_duplicate=True),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input("checkbox-button", "n_clicks"),
    State("results", "data"),
    State("curves", "data"),
    State("check-sv1", "value"),
    State("check-sv2", "value"),
    State("check-sl1", "value"),
    State("check-sl2", "value"),
    State("test", "data"),
    prevent_initial_call=True
)
def update_with_checkbox(checkbox_btn, results, curves, check_sv1, check_sv2, check_sl1, check_sl2, test_id):
    """
    Update threshold values and display based on checkbox button interaction.

    Parameters
    ----------
    checkbox_btn : int
        Number of clicks on the checkbox button.
    results : dict
        Current analysis results.
    curves : dict
        Curve data used for threshold computation.
    check_sv1 : bool
        Checkbox state for respiratory threshold 1.
    check_sv2 : bool
        Checkbox state for respiratory threshold 2.
    check_sl1 : bool
        Checkbox state for lactate threshold 1.
    check_sl2 : bool
        Checkbox state for lactate threshold 2.
    test_id : int
        ID of the test for which data is being analyzed.

    Returns
    -------
    tuple
        Updated threshold values and error messages if applicable.
    """
    try:
        if checkbox_btn:
            
            # Update results based on checkbox values
            temp_results = checkbox_update(check_sv1, check_sv2, check_sl1, check_sl2, curves, results)

            # Retrieve test data
            session = next(get_db())
            test = get_by_id(session, Test, test_id)

            # Extract maximum VO2 and heart rate values
            vo2_max = test.vo2_max
            fc_max = test.hr_max

            # Compute and display updated threshold data
            threshold_data = show_values(temp_results, vo2_max, fc_max)
        
            return (
                threshold_data[0],
                threshold_data[1],
                threshold_data[2],
                f"{threshold_data[2]} %",
                threshold_data[3],
                threshold_data[4],
                f"{threshold_data[4]} %",
                threshold_data[5],
                threshold_data[6],
                f"{threshold_data[6]} %",
                threshold_data[7],
                threshold_data[8],
                f"{threshold_data[8]} %",
                threshold_data[9],
                False, 
                ""
            )
        
        else:
            return no_update
        
    except Exception as e:
        logging.error(f"Error in update_with_checkbox : {e}\n{traceback.format_exc()}")
        return (
            no_update, no_update, no_update, no_update, no_update,
            no_update, no_update, no_update, no_update, no_update,
            no_update, no_update, no_update, no_update, True,
            "Une erreur s'est produite. Consultez les journaux pour plus de détails."
        )

# --- SECTION 4 : Callback related to the display of the PDF ---
@callback(
    Output("div-results-analysis", "hidden"),
    Output("div-pdf-preview", "hidden"),
    Output("report-results", "data"),
    Output("threshold-levels", "data"),
    Output("pdf-viewer", "src", allow_duplicate=True),
    Output("remark-input", "value", allow_duplicate=True),
    Output("remarks", "data", allow_duplicate=True),
    Output("pdf_report", "data", allow_duplicate=True),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input("tabs-analysis", "active_tab"),
    State("results", "data"),
    State("test", "data"),
    State("curves", "data"),
    State("use_lactate", "data"),
    State("check-sv1", "value"),
    State("check-sv2", "value"),
    State("check-sl1", "value"),
    State("check-sl2", "value"),
    State("remarks-table", "data"),
    State('tps-last', 'value'),
    running=[
        (Output("div-results-analysis", "hidden"), True, False),
        (Output("div-pdf-preview", "hidden"), True, False),
        (Output("div-progress-pdf", "hidden"), False, True),
    ],
    progress=[
        Output("progress-pdf", "value"),
        Output("progress-pdf", "max")
    ],
    prevent_initial_call=True,
)
def display_pdf(active_tab, results, test_id, curves, use_lactate, check_sv1, check_sv2, check_sl1, check_sl2, remarks, tps_last):
    """
    Generate and display PDF report or analysis results based on the selected tab.

    Parameters
    ----------
    active_tab : str
        The currently active tab in the analysis view.
    results : dict
        Analysis results data.
    test_id : int
        Identifier of the current test being analyzed.
    curves : dict
        Curve data for threshold computations.
    use_lactate : bool
        Whether lactate data is included in the analysis.
    check_sv1, check_sv2, check_sl1, check_sl2 : bool
        States of threshold-related checkboxes.
    remarks : list of dict
        Remarks provided by the user for the test.
    tps_last : str
        Last time-stamp or data point for the test.

    Returns
    -------
    tuple
        Updated visibility states, data, and error messages if applicable.
    """
    try:
        if remarks is not None:
            if active_tab == "tab-rapport":

                # Retrieve lactate data
                df_lactate = load_dataframe_from_id(test_id, lactate=True)

                # Update results with checkbox values
                results = checkbox_update(check_sv1, check_sv2, check_sl1, check_sl2, curves, results)

                # Compute thresholds
                levels = compute_levels(results, df_lactate, curves, test_id)

                # Retrieve test data
                session = next(get_db())
                test = get_by_id(session, Test, test_id)

                # Create results figure
                fig_results = create_figure_results(df_lactate, levels, results, use_lactate)

                # Format report results
                report_results = format_report_values(df_lactate, results, curves, levels, test_id, tps_last, use_lactate)

                # Create protocol and plateau figures
                fig_protocol = create_plot_protocol(df_lactate)
                fig_plateau, results_plateau = compute_plateau(test, report_results['vo2'])
                
                # Convert plots to bytes 
                bytes_image = fig_results.to_image(format = "png", scale=3, width=800)
                bytes_protocol = fig_protocol.to_image(format = "png", scale=3, width=1000)
                bytes_plateau = fig_plateau.to_image(format = "png", scale=3, width = 1000)

                # Update remarks and save plots into the database
                df_remarks = pd.DataFrame.from_records(remarks)
                test = update(
                    session, 
                    Test, 
                    test_id, 
                    {
                        'remarks': unidecode(' ; '.join(df_remarks['Remarques'])),
                        'graphics': bytes_image,
                        'protocol': bytes_protocol,
                        'plateau_study': bytes_plateau,
                    } | results_plateau
                )

                # Generate PDF
                latex_formatted_string = compute_pdf(df_lactate, test_id, report_results, levels, use_lactate)
                compile_pdf(latex_formatted_string, "temp_report")
                src_pdf = "./temp/temp_report.pdf"

                return (
                    True, False, report_results, levels, 
                    src_pdf, test.remarks, test.remarks, 
                    latex_formatted_string, False, ""
                )
            
            else:

                # Clear the temporary folder
                clear_folder("./temp")

                return (
                    False, True, no_update, no_update, 
                    no_update, None, None, no_update,
                    False, ""
                )
        
        return (
            False, True, no_update, no_update, 
            no_update, no_update, no_update, 
            no_update, False, ""
        )
    
    except Exception as e:
        logging.error(f"Error in display_pdf : {e}\n{traceback.format_exc()}")
        return (
            no_update, no_update, no_update, no_update, 
            no_update, no_update, no_update, no_update, 
            True, "Une erreur s'est produite. Consultez les journaux pour plus de détails."
        )

# --- SECTION 5 : Callback related to the management of the step of the analysis ---
@callback(
    Output("url", "pathname", allow_duplicate=True),
    Output("back", "data", allow_duplicate=True),
    Output("download_pdf", "data"),
    Output("download_excel", "data"),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input('prev-button-results', 'n_clicks'),
    Input("export-pdf-button", "n_clicks"),
    Input("upload-excel-file", "contents"),
    Input("quit-save-button", "n_clicks"),
    State("test", "data"),
    State("upload-excel-file", "contents"),
    State("threshold-levels", "data"),
    State("report-results", "data"),
    prevent_initial_call=True,
)
def previous_step(prev_btn, pdf_btn, excel_btn, save_btn, test_id, excel_file, levels, report_results):
    """
    Handles the navigation and export of results, including generating PDFs, 
    saving Excel files, and saving analysis data to the database.

    Parameters
    ----------
    prev_btn : int
        Number of clicks on the "Previous" button.
    pdf_btn : int
        Number of clicks on the "Export PDF" button.
    excel_btn : int
        Number of clicks on the "Export Excel" button.
    save_btn : int
        Number of clicks on the "Quit and Save" button.
    test_id : str
        ID of the current test.
    excel_file : str
        The contents of the uploaded Excel file.
    levels : dict
        The computed levels for the thresholds.
    report_results : dict
        The results of the report to be saved or exported.

    Returns
    -------
    tuple
        A tuple of updated states for the URL, back data, PDF download, Excel download, and error messages.
    """
    try:
        # Detect the trigger element
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        if trigger == "prev-button-results" and prev_btn:

            return "/preview", True, None, None, False, ""
        
        if trigger == "export-pdf-button" and pdf_btn:

            # Retrieve the PDF bytes from the temp folder
            with open(os.getcwd()+"/temp/temp_report.pdf", "rb") as f :
                pdf_bytes = f.read()
            
            # Retrieve the test
            session = next(get_db())
            test = get_by_id(session, Test, test_id)

            # Save the PDF in the database
            test = update(
                session, 
                Test, 
                test_id, 
                {
                    'pdf_report': pdf_bytes
                }
            )

            # Prepare PDF download for the user
            athlete = get_by_id(session, Athlete, test.athlete_id)
            prop_name_pdf = athlete.last_name.upper() + '_' + athlete.first_name.upper() + '_' + test.date.strftime("%d_%m_%Y") + '.pdf'
            download_pdf = dcc.send_bytes(
                pdf_bytes, filename=prop_name_pdf, type="application/pdf"
            )

            return no_update, False, download_pdf, None, False, ""
        
        if trigger == "upload-excel-file" and excel_file:

            # Retrieve the test and athlete data
            session = next(get_db())
            test = get_by_id(session, Test, test_id)
            athlete = get_by_id(session, Athlete, test.athlete_id)

            # Process the Excel file and update with VO2 and %VO2 data
            sheet = load_workbook(decode_contents(excel_file), data_only=True)
            k = 11
            vo2_values_excel = []
            pervo2_values_excel = []
            for v in levels["vo2"] :
                val = min(v * test.weight, test.vo2_max * 1000)
                vo2_values_excel.append(round(val, 2))
                sheet['Feuil1']['J'+str(k)] = round(val, 2)
                pervo2_values_excel.append(round(val*100/(test.vo2_max*1000), 2))
                sheet['Feuil1']['K'+str(k)] = round(val*100/(test.vo2_max*1000), 2)
                k += 1

            # Save the Excel file
            excel_bytes = BytesIO()
            sheet.save(excel_bytes)
            excel_bytes.seek(0)
            bytes_excel = excel_bytes.read()
            prop_name_excel = athlete.last_name.upper() + '_' + athlete.first_name.upper() + '_' + test.date.strftime("%d_%m_%Y") + '.xlsx'
            download_excel = dcc.send_bytes(bytes_excel, filename=prop_name_excel, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            
            # Update lactate data with the new values
            df_lactate = load_dataframe_from_id(test_id, lactate=True)
            df_lactate['vo2'] = vo2_values_excel
            df_lactate['%vo2max'] = pervo2_values_excel
            test = update(
                session, 
                Test, 
                test_id, 
                {
                    'source_excel': df_lactate.to_json(orient="records")
                }
            )
            
            return no_update, False, None, download_excel, False, ""
            
        if trigger == "quit-save-button" and save_btn:

            # Save analysis results to the database
            session = next(get_db())
            attributes = ['slope', 'speed', 'vo2', 'vo2_kg', 'vo2_ratio', 've', 'hr',
                            'hr_ratio', 'watt', 'watt_kg', 'de', 'glu', 'lip', 'lactate']
            index_attributes = ["s1", "s2", "max"]
            obj_in = {}
            for s in range(3) :
                for i in range(14) :
                    obj_in[f"{attributes[i]}_{index_attributes[s]}"] = report_results[attributes[i]][s]
            obj_in["tps_max"] = report_results['tps_last']
            test = update(
                session, 
                Test, 
                test_id, 
                obj_in
            )

            # Clear temporary files
            clear_folder("./temp")

            return "/upload", False, None, None, False, ""

        return no_update
    
    except Exception as e:
        logging.error(f"Error in previous_step : {e}\n{traceback.format_exc()}")
        return (
            no_update, no_update, no_update, no_update, 
            True, "Une erreur s'est produite. Consultez les journaux pour plus de détails."
        )

# --- SECTION 6 : Callback related to the modification of the remarks ---
@callback(
    Output("pdf-viewer", "key"),
    Output("remark-input", "value", allow_duplicate=True),
    Output("remarks", "data", allow_duplicate=True),
    Output("pdf_report", "data", allow_duplicate=True),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input("remark-submit", "n_clicks"),
    State("pdf_report", "data"),
    State("remark-input", "value"),
    State("remarks", "data"),
    prevent_initial_call=True,
)
def modify_remarks(remark_btn, latex_string, new_remarks, past_remarks):
    """
    Handles the modification of remarks in the test report. When the user submits new remarks,
    it updates the LaTeX string, regenerates the PDF, and updates the UI.

    Parameters
    ----------
    remark_btn : int
        The number of times the "Submit Remarks" button has been clicked.
    latex_string : str
        The current LaTeX string of the report.
    new_remarks : str
        The new remarks entered by the user.
    past_remarks : str
        The previous remarks from the test.

    Returns
    -------
    tuple
        A tuple containing updated values for the PDF viewer, remark input, 
        remarks, PDF data, visibility and message of the error toast.
        If no update is necessary, `no_update` is returned.
    """
    try:
        # Detect the trigger element
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        if trigger == "remark-submit" and remark_btn:

            # Update the LaTeX string with the new remarks
            match = re.search(f"Remarques du testeur : }} {past_remarks}", latex_string)
            if match:
                new_latex_string = f"{latex_string[:match.start()]}Remarques du testeur : }} {new_remarks} {latex_string[match.end():]}"
            else:
                new_latex_string = f"{latex_string}Remarques du testeur : }} {new_remarks}"

            new_latex_string = f"{latex_string[:match.start()]}Remarques du testeur : }} {new_remarks} {latex_string[match.end():]}"

            # Compile the PDF with the updated LaTeX string
            compile_pdf(new_latex_string, "temp_report")

            return str(time.time()), new_remarks, new_remarks, new_latex_string, False, ""
        
        return no_update
    
    except Exception as e:
        logging.error(f"Error in modify_remarks : {e}\n{traceback.format_exc()}")
        return (
            no_update, no_update, no_update, no_update, 
            True, "Une erreur s'est produite. Consultez les journaux pour plus de détails."
        )