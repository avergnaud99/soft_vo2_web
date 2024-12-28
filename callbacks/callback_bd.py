"""
This module contains callback functions that handle the database.

Features :
- Database visualisation.
- Interaction with the tables.
- Interaction with the database : addition/deletion of a team, deletion of a test.
- Export reports (tests, followup, team)

"""

### Imports ###
from dash import Input, Output, callback, dcc, no_update, State, callback_context
from datetime import date, datetime
from utils.database import get_db
from utils.models import *
from utils.crud import *
from utils.data_displaying import *
from utils.pdf_processing import *

### Callbacks ###

# --- SECTION 1 : Callback for data visualisation ---
@callback(
    Output("bd-table", "data"), 
    Output("bd-table", "columns"),
    Output("bd-table", "row_selectable"),
    Output("add-team-div", "hidden"),
    Output("export-followup-bd", "hidden"),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input("tabs", "active_tab"),
    prevent_initial_call=True
)
def render_database_content(active_tab):
    """
    Update the table and visibility settings based on the active tab.

    Parameters
    ----------
    active_tab : str
        The identifier of the currently active tab.

    Returns
    -------
    tuple
        - data (list of dict): The data to display in the table.
        - columns (list of dict): The column definitions for the table.
        - row_selectable (str): Row selection mode ("single", "multi", or None).
        - hidden_team (bool): Visibility of the team addition division.
        - hidden_followup (bool): Visibility of the follow-up export division.
        - Visibility and content of the error toast.
    """
    try:
        # Calculate data for the table based on the active tab
        data, columns, row_selectable = show_database(active_tab)

        # Configure the visibility of the extra divisions
        hidden_followup = False if active_tab == "tab-athletes" else True
        hidden_team = False if active_tab == "tab-teams" else True

        # Validate results
        if any([data is None, columns is None, hidden_team is None]):
            raise ValueError("One or more required outputs are None.")

        return data, columns, row_selectable, hidden_team, hidden_followup, False, ""

    except Exception as e:
        logging.error(f"Error in render_database_content: {e} \n {traceback.format_exc()}")
        return (
            no_update, no_update, no_update, no_update, no_update, True,
            "Une erreur s'est produite. Consultez les journaux pour plus de détails."
        )
    
# --- SECTION 2 : Callback associated with a click on the table ---
@callback(
    Output("bd-table", "data", allow_duplicate=True), 
    Output("bd-table", "columns", allow_duplicate=True), 
    Output("download-report-bd", "data"),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Output("confirmation-modal-team", "is_open", allow_duplicate=True),
    Output("modal-body-team", "children"),
    Output("confirmation-modal-delete", "is_open", allow_duplicate=True),
    Output("modal-body-delete", "children"),
    Output("slope-flat-tests-team", "data"),
    Input("bd-table", "active_cell"),
    State("tabs", "active_tab"),
    State("bd-table", "derived_viewport_data"),
    prevent_initial_call=True,
)
def click_table(active_cell, active_tab, data_table):
    """
    Handle actions triggered by a click on a cell in the table.

    Parameters
    ----------
    active_cell : dict
        Information about the clicked cell, including row and column IDs.
    active_tab : str
        The identifier of the currently active tab.
    data_table : list of dict
        The data displayed in the current table.

    Returns
    -------
    tuple
        - Updated data or columns for the table.
        - Data for downloading reports.
        - Visibility and content of toasts or modals.
        - Slope and flat tests for teams.
    """
    if active_cell:
        try:

            # Click on a download cell
            if active_cell["column_id"] == "Télécharger":

                # Test table
                if active_tab == "tab-tests":

                    # Retrieve the test ID
                    row_id = active_cell["row"]
                    session = next(get_db())
                    test_id = data_table[row_id]["ID Test"]
                    test = get_by_id(session, Test, test_id)

                    # Check if the PDF report exists
                    if not test.pdf_report:
                        return (
                            no_update, no_update, no_update,
                            True, "Aucun rapport PDF n'est lié à ce test.",
                            False, "", False, "", no_update
                        )
                    
                    # Bytes of the PDF file
                    pdf_bytes = test.pdf_report

                    # File name
                    filename = 'Rapport_' + test.athlete.last_name.capitalize() + '_' + test.athlete.first_name.capitalize() + '_' + date.strftime(test.date, "%d/%m/%Y") + '.pdf'

                    return (
                        no_update, no_update, dcc.send_bytes(pdf_bytes, filename=filename),
                        no_update, no_update, no_update, no_update, no_update, no_update, no_update
                    )
                
                # Team table
                elif active_tab == "tab-teams":

                    # Retrieve the Team
                    row_id = active_cell["row"]
                    session = next(get_db())
                    team_name = data_table[row_id]["Nom de l'équipe"].lower()
                    team = session.query(Team).get(team_name)

                    # Get the athletes
                    team_athletes = team.athletes
                    if not team_athletes:
                        return (
                            no_update, no_update, no_update,
                            True, "Aucun athlète associé à cette équipe.",
                            no_update, no_update, no_update,
                            no_update, no_update
                        )
                    
                    # Get the last tests of the team for the current or last year
                    last_year = [t.tests[-1].date.year for t in team_athletes if t.tests]
                    if not last_year:
                        return (
                            no_update, no_update, no_update,
                            True, "Aucun test associé à cette équipe.",
                            no_update, no_update, no_update,
                            no_update, no_update
                        )
                    last_year = max(last_year)
                    team_tests = [t.tests[-1] for t in team_athletes if (t.tests and t.tests[-1].date.year == last_year)]

                    # Test the homogeneity of the slopes
                    non_zero_tests, zero_tests = test_slope_report(team_tests)

                    # Absence of tests
                    if not non_zero_tests and not zero_tests:
                        return (
                            no_update, no_update, no_update,
                            True, f"Aucun test associé à cette équipe en {last_year}.",
                            no_update, no_update, no_update, no_update,
                            {"slope": None, "flat": None}
                        )
                    
                    # Heterogeneity of the tests
                    elif non_zero_tests and zero_tests:
                        return (
                            no_update, no_update, no_update,
                            no_update, no_update, True,
                            "Quels tests voulez-vous choisir ?",
                            no_update, no_update,
                            {"slope": [t.id for t in non_zero_tests], "flat": [t.id for t in zero_tests]}
                        )
                    
                    # Homogeneity of the tests
                    else:
                        return (
                            no_update, no_update, no_update,
                            no_update, no_update, no_update,
                            no_update, no_update, no_update,
                            {"slope": [t.id for t in non_zero_tests], "flat": [t.id for t in zero_tests]}
                        )

            # Click on a delete cell
            elif active_cell["column_id"] == "Supprimer":

                # Test table
                if active_tab == "tab-tests":

                    # Retrieve the test ID
                    row_id = active_cell["row"]
                    session = next(get_db())
                    test_id = data_table[row_id]["ID Test"]
                    test = get_by_id(session, Test, test_id)

                    # Create the message
                    message_info = f"""
                        Vous vous apprêtez à supprimer le test n°{test.id} 
                        de l'athlète {test.athlete.first_name.capitalize()} {test.athlete.last_name.capitalize()}
                        du {date.strftime(test.date, "%d/%m/%Y")}.
                        Êtes-vous sûr de vouloir continuer ? 
                    """

                # Team table
                elif active_tab == "tab-teams":

                    # Retrieve the Team
                    row_id = active_cell["row"]
                    session = next(get_db())
                    team_name = data_table[row_id]["Nom de l'équipe"].lower()
                    team = session.query(Team).get(team_name)

                    # Create the message
                    message_info = f"""
                        Vous vous apprêtez à supprimer la Team {team.name.upper()} 
                        comprenant {len(team.athletes)} athlètes. 
                        Êtes-vous sûr de vouloir continuer ? 
                    """

                return (
                    no_update, no_update, no_update,
                    no_update, no_update, no_update, 
                    no_update, True, message_info, no_update
                )
            
        except Exception as e:
            logging.error(f"Error in click_table : {e}\n{traceback.format_exc()}")
            return (
                no_update, no_update, no_update,
                True, "Une erreur s'est produite. Consultez les journaux pour plus de détails.",
                no_update, no_update, no_update, no_update, no_update
            )
    else:
        return no_update
    
# --- SECTION 3 : Callback for the deletion of an item ---
@callback(
    Output("bd-table", "data", allow_duplicate=True), 
    Output("bd-table", "columns", allow_duplicate=True), 
    Output("confirmation-modal-delete", "is_open", allow_duplicate=True),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input("confirm-delete", "n_clicks"),
    Input("cancel-delete", "n_clicks"),
    State("bd-table", "active_cell"),
    State("tabs", "active_tab"),
    State("bd-table", "derived_viewport_data"),
    prevent_initial_call=True
)
def delete_item(confirm_btn, cancel_btn, active_cell, active_tab, data_table):
    """
    Handle deletion of items (tests or teams) based on user confirmation.

    Parameters
    ----------
    confirm_btn : int
        Number of times the confirm delete button has been clicked.
    cancel_btn : int
        Number of times the cancel delete button has been clicked.
    active_cell : dict
        Information about the currently active cell in the table.
    active_tab : str
        Identifier of the currently active tab.
    data_table : list of dict
        Data currently displayed in the table.

    Returns
    -------
    tuple
        - Updated table data and columns.
        - Boolean indicating whether the delete confirmation modal is open.
        - Visibility and content of the error toast.
    """
    try:
        # Identify the triggering input
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        # Confirmation of the delete action
        if trigger == "confirm-delete" and confirm_btn:

            # Ensure an active cell is selected
            if not active_cell:
                logging.error("No active cell selected for deletion.")
                return (
                    no_update, no_update, False, True, 
                    "Une erreur s'est produite. Consultez les journaux pour plus de détails."
                )
            
            # Get the selected row
            row_id = active_cell["row"]
            session = next(get_db())

            # Test table
            if active_tab == "tab-tests":

                # Retrieve and delete the test
                test_id = data_table[row_id]["ID Test"]
                test = get_by_id(session, Test, test_id)
                delete(session, Test, test.id)

            # Team table
            elif active_tab == "tab-teams":
                
                # Retrieve and delete the team
                team_name = data_table[row_id]["Nom de l'\u00e9quipe"].lower()
                team = session.query(Team).get(team_name)
                session.delete(team)
                session.commit()
            
            else:
                logging.error(f"Unsupported tab for deletion: {active_tab}")
                return (
                    no_update, no_update, False, True, 
                    "Une erreur s'est produite. Consultez les journaux pour plus de détails."
                )
            
            # Refresh table data
            data, columns, _ = show_database(active_tab)
            return data, columns, False, False, ""
        
        # Cancelation of the delete action
        elif trigger == "cancel-delete" and cancel_btn:
            # Close the delete confirmation modal without changes
            return no_update, no_update, False, False, ""

        # Default no-op return
        return no_update

    except Exception as e:
        logging.error(f"An error in delete_item: {e}\n{traceback.format_exc()}")
        return (
            no_update, no_update, False, True, 
            "Une erreur s'est produite. Consultez les journaux pour plus de détails."
        )

# --- SECTION 4 : Callback for the insertion of a team ---
@callback(
    Output("bd-table", "data", allow_duplicate=True), 
    Output("bd-table", "columns", allow_duplicate=True), 
    Output("error-toast-app", "is_open"),
    Output("error-toast-app", "children"),
    Input("add-team-button", "n_clicks"),
    State("team-name-input", "value"),
    State("tabs", "active_tab"),
    prevent_initial_call=True
)
def add_team(add_btn, team_name, active_tab):
    """
    Handle the addition of a new team via user input.

    Parameters
    ----------
    add_btn : int
        The number of times the "Add Team" button has been clicked.
    team_name : str
        The name of the team entered by the user.
    active_tab : str
        The identifier of the currently active tab.

    Returns
    -------
    tuple
        - Updated data for the table.
        - Updated columns for the table.
        - Boolean indicating if the error toast should be displayed.
        - Visibility and content of the error toast.
    """
    try:
        if add_btn:
            if team_name:

                # Format the name
                name = team_name.lower().strip()

                # Check if the team already exists in the database
                session = next(get_db())
                if session.query(Team).get(name) is None :

                    # Create a new team entry
                    new_team = create(session, Team, {'name': name})

                    # Update the table data
                    data, columns, _ = show_database(active_tab)

                    return data, columns, False, ""
                
                else:
                    logging.warning(f"Attempt to add a team with an existing name: {name}")
                    return (
                        no_update, no_update, True, 
                        "Attention, le nom que vous avez rentr\u00e9 semble d\u00e9j\u00e0 \u00eatre li\u00e9 \u00e0 une \u00e9quipe existante."
                    )
            
            else:
                logging.warning("Attempt to add a team with an empty name field.")
                return (
                    no_update, no_update, True, 
                    "Le champ 'Nom de l'\u00e9quipe' ne peut pas \u00eatre vide."
                )
        
        else:
            return no_update
        
    except Exception as e:
        logging.error(f"An error occurred while adding a team: {e}\n{traceback.format_exc()}")
        return (
            no_update, no_update, True, 
            "Une erreur s'est produite. Consultez les journaux pour plus de détails."
        )

# --- SECTION 5 : Callback for updating the dropdown of the tests linked to an athlete ---
@callback(
    Output("dropdown-tests-followup", "options"),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input("bd-table", "selected_rows"),
    State("bd-table", "data"),
    prevent_initial_call=True
)
def update_dropdown_followup(rows, data_table):
    """
    Update the options of the dropdown for follow-up tests based on the selected athlete.

    Parameters
    ----------
    rows : list of int
        The indices of the rows selected in the table.
    data_table : list of dict
        The data currently displayed in the table.

    Returns
    -------
    tuple
        - List of dropdown options for the selected athlete's tests.
        - Boolean indicating whether the error toast should be open.
        - String containing the error message if applicable.
    """
    try:
        if rows:

            # Get the athlete for the selected row
            session = next(get_db())
            athlete_id = data_table[rows[0]]["ID"]
            athlete = get_by_id(session, Athlete, athlete_id)

            if athlete:
                # Generate dropdown options from the athlete's tests
                tests = [
                    {"label": date.strftime(t.date, "%d/%m/%Y"), "value": t.id}
                    for t in athlete.tests
                ]

                return tests, False, ""
            else:
                logging.error(f"Athlete with ID {athlete_id} not found.")
                return no_update, True, "Athlète non trouvé dans la base de données."
        
        else:
            return no_update, False, ""
        
    except Exception as e:
        logging.error(f"Error in update_dropdown_followup: {e}\n{traceback.format_exc()}")
        return no_update, True, "Une erreur s'est produite. Consultez les journaux pour plus de détails."

# --- SECTION 6 : Callback for the export of a team report ---
@callback(
    Output("download-report-bd", "data", allow_duplicate=True),
    Output("confirmation-modal-team", "is_open", allow_duplicate=True),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input("flat-report-team", "n_clicks"),
    Input("slope-report-team", "n_clicks"),
    Input("slope-flat-tests-team", "data"),
    State("bd-table", "active_cell"),
    State("bd-table", "derived_viewport_data"),
    prevent_initial_call=True
)
def export_team_report(flat_btn, slope_btn, all_tests_team, active_cell, data_table):
    """
    Export a team performance report as a PDF based on the selected type of tests.

    Parameters
    ----------
    flat_btn : int
        Number of clicks on the "flat report" button.
    slope_btn : int
        Number of clicks on the "slope report" button.
    all_tests_team : dict
        A dictionary containing test IDs for slope and flat tests.
    active_cell : dict
        Information about the currently active cell in the table.
    data_table : list of dict
        Data displayed in the table.

    Returns
    -------
    tuple
        - PDF file for download.
        - State of the confirmation modal.
        - Visibility and content of the error toast.
    """
    try:

        # Identify the triggering input
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        if all_tests_team and any(list(all_tests_team.values())):
            bool_flat = False
            team_tests = []

            # Determine the test type based on the trigger
            if trigger == "flat-report-team" and flat_btn:
                team_tests = all_tests_team['flat']
                bool_flat = True
            elif trigger == "slope-report-team" and slope_btn:
                team_tests = all_tests_team['slope']
            elif trigger == "slope-flat-tests-team" and all_tests_team:
                bool_flat = True if not all_tests_team["slope"] else False
                team_tests = [test for subtests in all_tests_team.values() for test in subtests]

            # Retrieve team information
            session = next(get_db())
            team_athletes = [get_by_id(session, Test, t).athlete for t in team_tests]
            row_id = active_cell["row"]
            team_name = data_table[row_id]["Nom de l'équipe"].lower()
            team = session.query(Team).get(team_name)

            # Generate report figures
            fig_max, fig_threshold = create_team_report(team_tests, team_athletes, bool_flat)

            # Convert figures to images
            bytes_max = fig_max.to_image(format = "png", scale=3, width=1000)
            bytes_threshold = fig_threshold.to_image(format = "png", scale=3, width=1000)
            bytes_images = [bytes_max, bytes_threshold]
            paths = ['./temp/temp_team_max.png', './temp/temp_team_threshold.png']
            for i in range(2):
                save_temp_files(bytes_images[i], paths[i])

            # Generate LaTeX report and compile it
            latex_formatted_team_report = generate_latex_team(team)
            compile_pdf(latex_formatted_team_report, "temp_team_report")

            # Retrieve PDF bytes
            prop_name = team.name.upper() + '_' + datetime.now().date().strftime("%d_%m_%Y") + '.pdf'
            with open("./temp/temp_team_report.pdf", "rb") as f:
                pdf_bytes = f.read()

            # Clean up temporary files
            clear_folder("./temp")

            return dcc.send_bytes(pdf_bytes, filename=prop_name), False, False, ""

        return no_update
    
    except Exception as e:
        logging.error(f"Error in export_team_report: {e}\n{traceback.format_exc()}")
        return no_update, True, True, "Une erreur s'est produite. Consultez les journaux pour plus de détails."

# --- SECTION 7 : Callback for the export of a followup report ---
@callback(
    Output("download-report-bd", "data", allow_duplicate=True),
    Output("confirmation-modal-followup", "is_open", allow_duplicate=True),
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Input("flat-report-followup", "n_clicks"),
    Input("slope-report-followup", "n_clicks"),
    Input("slope-flat-tests-followup", "data"),
    State("bd-table", "selected_rows"),
    State("bd-table", "data"),
    prevent_initial_call=True
)
def export_followup_report(flat_btn, slope_btn, all_tests_followup, rows, data_table):
    """
    Export an athlete's follow-up performance report as a PDF based on the selected type of tests.

    Parameters
    ----------
    flat_btn : int
        Number of clicks on the "flat report" button.
    slope_btn : int
        Number of clicks on the "slope report" button.
    all_tests_followup : dict
        A dictionary containing test IDs for slope and flat tests.
    rows : list
        List of selected rows in the table.
    data_table : list of dict
        Data displayed in the table.

    Returns
    -------
    tuple
        - PDF file for download or `no_update`.
        - State of the confirmation modal or `no_update`.
        - Visibility and content of the error toast.
    """
    try:

        # Identify the triggering input
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        if all_tests_followup and any(list(all_tests_followup.values())):
            bool_flat = False
            tests = []

            # Retrieve athlete information
            session = next(get_db())
            athlete_id = data_table[rows[0]]["ID"]
            athlete = get_by_id(session, Athlete, athlete_id)

            # Determine the test type based on the trigger
            if trigger == "flat-report-followup" and flat_btn:
                tests = all_tests_followup['flat']
                bool_flat = True
            elif trigger == "slope-report-followup" and slope_btn:
                tests = all_tests_followup['slope']
            elif trigger == "slope-flat-tests-followup" and all_tests_followup:
                bool_flat = True if not all_tests_followup["slope"] else False
                tests = [test for subtests in all_tests_followup.values() for test in subtests]

            # Fetch detailed test data
            tests = [get_by_id(session, Test, t) for t in tests]

            # Generate report components
            fig_values, fig_thresholds, latex_tabular_values, levels_labels, levels_units = create_followup_report(
                tests, bool_flat
            )
            
            # Convert figures to images
            bytes_max = fig_values.to_image(format = "png", scale=3, width=1000)
            bytes_threshold = fig_thresholds.to_image(format = "png", scale=3, width=1000)
            bytes_images = [bytes_max, bytes_threshold]
            paths = ['./temp/temp_followup_curves.png', './temp/temp_followup_threshold.png']
            for i in range(2):
                save_temp_files(bytes_images[i], paths[i])

            # Generate LaTeX report and compile
            latex_formatted_followup_report = generate_latex_followup(
                athlete, latex_tabular_values, levels_labels, levels_units
            )
            compile_pdf(latex_formatted_followup_report, "temp_followup_report")

            # Retrieve PDF bytes
            prop_name = athlete.last_name.upper() + '_' + athlete.first_name.upper() + '_' + datetime.now().date().strftime("%d_%m_%Y") + '_suivi' + '.pdf'
            with open("./temp/temp_followup_report.pdf", "rb") as f:
                pdf_bytes = f.read()

            # Clean up temporary files
            clear_folder("./temp")

            return dcc.send_bytes(pdf_bytes, filename=prop_name), False, False, ""
        
        return no_update
            
    except Exception as e:
        logging.error(f"Error in export_followup_report : {e}, {traceback.format_exc()}")
        return no_update, no_update, True, "Une erreur s'est produite. Consultez les journaux pour plus de détails."

# --- SECTION 8 : Callback for the precomputation of a followup report ---
@callback(
    Output("error-toast-app", "is_open", allow_duplicate=True),
    Output("error-toast-app", "children", allow_duplicate=True),
    Output("confirmation-modal-followup", "is_open", allow_duplicate=True),
    Output("modal-body-followup", "children", allow_duplicate=True),
    Output("slope-flat-tests-followup", "data"),
    Input("export-followup-report", "n_clicks"),
    State("dropdown-tests-followup", "value"),
    prevent_initial_call=True
)
def precompute_followup_report(export_btn, tests):
    """
    Precomputes data for the follow-up report based on selected tests.

    Parameters
    ----------
    export_btn : int
        Number of clicks on the "export follow-up report" button.
    tests : list
        List of selected test IDs from the dropdown menu.

    Returns
    -------
    tuple
        - Error toast visibility state.
        - Error toast message.
        - Confirmation modal visibility state.
        - Modal body content.
        - Slope and flat test data in a dictionary.
    """
    try:
        # Detect the triggering input
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        if trigger == "export-followup-report" and export_btn:

            if not tests:
                return True, "Aucun test sélectionné", no_update, no_update, no_update
            
            # Retrieve tests from the database
            session = next(get_db())
            chosen_tests = [get_by_id(session, Test, t) for t in tests]

            # Check the homogeneity of slopes
            non_zero_tests, zero_tests = test_slope_report(chosen_tests)

            # Heterogeneous tests
            if non_zero_tests and zero_tests:
                return (
                    no_update, no_update, True,
                    "Quels tests voulez-vous choisir ?",
                    {"slope": [t.id for t in non_zero_tests], "flat": [t.id for t in zero_tests]}
                )
            
            # Homogeneous tests
            else:
                return (
                    no_update, no_update, no_update, no_update,
                    {"slope": [t.id for t in non_zero_tests], "flat": [t.id for t in zero_tests]}
                )

        return no_update
    
    except Exception as e:
        logging.error(f"Error in precompute_followup_report : {e}, {traceback.format_exc()}")
        return (
            True, "Une erreur s'est produite. Consultez les journaux pour plus de détails.", 
            no_update, no_update, no_update
        )