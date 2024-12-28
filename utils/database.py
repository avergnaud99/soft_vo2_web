'''
This module handles the database.
'''

### Imports ###

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import hashlib
from utils.models import Base

### Methods ###

def verify_user(username, password):
    """
    Verifies that the user is registered in the database and that their password is correct.

    Parameters
    ----------
    username : str
        The username to verify.
    password : str
        The password associated with the username.

    Returns
    -------
    bool
        - `True` if the username exists in the database and the password matches.
        - `False` otherwise.
    """

    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    return USERS.get(username)[0] == hashed_password

def get_user_info(username):
    """
    Retrieves the user information associated with the given username.

    Parameters
    ----------
    username : str
        The username for which to fetch the associated user information.

    Returns
    -------
    str
        The user information (typically stored in the second element of the value) for the specified username.
        Returns None if the username does not exist in the USERS dictionary.
    """

    return USERS.get(username)[1]

def get_db():
    """
    Provides a database session using a managed context, ensuring that 
    resources are properly released once used.

    This function is often used as a dependency in frameworks like FastAPI 
    to interact with a database via SQLAlchemy.

    Yields
    ------
    sqlalchemy.orm.Session
        An instance of a SQLAlchemy session to perform operations on the database.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

### Main ###

# Type and localisation of the database
DATABASE_URL = "sqlite:///VO2.db"

# Create the session of the database
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# Users and passwords of the database
USERS = {
    "avergnaud": [hashlib.sha256("motdepasse".encode()).hexdigest(), "Alexandre Vergnaud"],
}

