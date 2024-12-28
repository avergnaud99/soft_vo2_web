"""
This module contains callback functions that handle interactions on the homepage.

Features:
- Dynamic update of the welcome message.
- Error handling for invalid inputs.
- Error logging.
- Displaying errors via a dbc.Toast component.
"""

### Imports ###

from dash import Input, Output, callback

### Callbacks ###

@callback(
    Output("welcome-message", "children"), 
    Input("login-data", "data"),
)
def welcome(user_info):
    """
    Dynamically updates the welcome message displayed to the user.

    Parameters
    ----------
    user_info : dict or None
        User information or `None` if no data is available.

    Returns
    -------
    str
        A personalized welcome message if the user data is valid.
    """

    if not user_info:
        raise ValueError("User data is missing.")

    return f"Bienvenue, {user_info} !", False, ""