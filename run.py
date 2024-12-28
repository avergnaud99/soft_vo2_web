'''
This module launches the Dash application and manages its pages.
'''

### Imports ###

from app import app, create_navbar, navbar_components_analysis, navbar_components_normal
from dash import Input, Output, State, html, callback_context, dcc
import dash_bootstrap_components as dbc
from flask import session
from pages import bd, login, accueil, modelization_analysis, time_analysis, upload, overview, preview, results
from callbacks import callback_log, callback_upload, callback_overview, callback_quit_analysis, callback_bd, callback_accueil
from callbacks import callback_time_analysis, callback_cubic_analysis, callback_first_linear_analysis, callback_second_linear_analysis
from callbacks import callback_preview, callback_results

### Main ###

# Pages registration in the application
app.validation_layout = html.Div([login.login_page, 
                                  upload.upload_page, 
                                  bd.bd_page, accueil.accueil_page, 
                                  overview.overview_page,
                                  time_analysis.time_analysis_page,
                                  modelization_analysis.cubic_analysis_page,
                                  modelization_analysis.first_linear_analysis_page,
                                  modelization_analysis.second_linear_analysis_page,
                                  preview.preview_page,
                                  results.results_page])

# Define possible pages
PAGES_NAV = {
    '/': accueil.accueil_page,
    '/upload': upload.upload_page,
    '/bd': bd.bd_page
}
PAGES_ANALYSIS_NAV = {
    '/overview': overview.overview_page,
    '/time_analysis': time_analysis.time_analysis_page,
    '/cubic_analysis': modelization_analysis.cubic_analysis_page,
    '/first_linear_analysis': modelization_analysis.first_linear_analysis_page,
    '/second_linear_analysis': modelization_analysis.second_linear_analysis_page,
    '/preview': preview.preview_page,
    '/results':results.results_page
}

# Callback for navigation
@app.callback(
    Output("navbar-container", "children"),
    Output("page-content", "children"),
    Input("url", "pathname"),
    Input("login-status", "data"),
    prevent_initial_call=True
)
def display_page(pathname, login_status):
    """
    Manages the display of pages based on the URL and the user's login status.

    This callback determines which page to display and updates the URL accordingly, 
    considering the user's login status.

    Parameters
    ----------
    pathname : str
        The current URL path corresponding to the page requested by the user.
    login_status : bool
        Indicates whether the user is logged in or not:
        - `True` : The user is logged in.
        - `False` : The user is logged out.

    Returns
    -------
    tuple
        - page_content : html.Div
            The HTML content of the page to be displayed.
        - url : str
            The URL path corresponding to the displayed page.
    """

    if login_status:
        if pathname in PAGES_NAV:
            navbar = create_navbar(navbar_components_normal)
            return navbar, PAGES_NAV[pathname]
        elif pathname in PAGES_ANALYSIS_NAV:
            navbar = create_navbar(navbar_components_analysis)
            return navbar, PAGES_ANALYSIS_NAV[pathname]
        else:
            return None, html.H1("Page 404 - Page non trouvée")
    else:
        return None, login.login_page

# Launch the application
if __name__ == '__main__':
    app.run(debug=True)
