"""
This module handles the callbacks for the login and logout functionalities.

Features :
- Logout and login.

"""

### Imports ###
from dash import Input, Output, State, callback
from utils.database import verify_user, get_user_info

### Callbacks ###

# --- SECTION 1 : Callback related to user logout ---
@callback(
    Output("login-status", "data", allow_duplicate=True),
    Input("logout-button", "n_clicks"),
    prevent_initial_call=True
)
def logout(n_clicks):
    """
    Handles user logout when the "logout-button" is clicked.

    This function updates the user's login status by modifying the `data` property
    of the `dcc.Store` component with the identifier `login-status`.

    Parameters
    ----------
    n_clicks : int
        The number of times the "logout-button" has been clicked. Each click triggers
        the callback.

    Returns
    -------
    login-status : bool
        - `False`: Indicates the user is logged out (button clicked).
        - `True`: Default value if the callback is triggered without a click
          (normally prevented by `prevent_initial_call=True`).
    """
    if n_clicks:
        return False
    return True

# --- SECTION 2 : Callback related to user login ---
@callback(
    Output("login-status", "data", allow_duplicate=True),
    Output("login-message", "children"),
    Output("login-data", "data"),
    Input("login-button", "n_clicks"),
    State("username", "value"),
    State("password", "value"),
    State("login-message", "children"),
    prevent_initial_call=True
)
def login(n_clicks, username, password, message):
    """
    Callback to manage user login.

    Parameters
    ----------
    n_clicks : int
        The number of clicks on the login button.
    username : str
        The entered username.
    password : str
        The entered password.
    
    Returns
    -------
    bool, str 
        Connection status and user message.
    """
    if n_clicks:
        
        # Missing field value
        if not username or not password:
            return False, "Veuillez remplir tous les champs.", ""
        
        # Verify the user
        if verify_user(username, password):
            return True, "", get_user_info(username)
        
        return False, "Nom d'utilisateur ou mot de passe incorrect", ""
    
    return False, message, ""
