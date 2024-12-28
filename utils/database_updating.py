'''
This module handles the objects update in the database.
'''

### Import ###

import pandas as pd
import Levenshtein
from datetime import date
from utils.database import get_db
from utils.crud import *
from utils.models import *
import logging
import traceback

### Methods ###

def normalize_string(s, name):
    """
    Normalize a string by converting it to lowercase, stripping whitespace, and replacing spaces with hyphens.

    This function takes an input string, ensures it is not null, and performs the following operations:
    - Converts the string to lowercase.
    - Strips leading and trailing whitespaces.
    - Replaces all spaces with hyphens.

    Parameters
    ----------
    s : str
        The string to be normalized. It should be a non-null string.
    name : str
        The name of the field or entity to be used in the error message if the string is missing.
    
    Returns
    -------
    str
        The normalized string, where spaces are replaced by hyphens, and the string is in lowercase with no leading/trailing spaces.
    """
    try:
        if pd.isnull(s):
            raise ValueError(f"Le {name} est manquant.")
        return "-".join(s.lower().strip().split())
    
    except Exception as e:
        logging.error(f"Error in normalize_string: {e}\n{traceback.format_exc()}")
        raise

def check_and_save(name, dataframe, i, j):
    """
    Checks if a value in a DataFrame is present. If it is missing, an exception is raised.

    This function checks the cell of a DataFrame specified by the row and column indices. 
    If the cell is empty (NaN), a `ValueError` exception is raised, indicating that the value is missing.

    Parameters
    ----------
    name : str
        The name of the field for customizing the error message, e.g., "age" or "score".
    dataframe : pandas.DataFrame
        The DataFrame containing the data in which the cell is checked.
    i : int
        The row index in the DataFrame where the cell to be checked is located.
    j : int
        The column index in the DataFrame where the cell to be checked is located.
    
    Returns
    -------
    value : type depending on the cell
        The value of the cell at the intersection of row index `i` and column index `j` in the DataFrame.
    """
    try:
        if pd.isnull(dataframe.iloc[i, j]):
            raise ValueError(f"Le/la {name} est manquant(e).")
        return dataframe.iloc[i, j]

    except Exception as e:
        logging.error(f"Error in check_and_save: {e}\n{traceback.format_exc()}")
        raise

def load_athlete_profile(df_lactate):
    """
    Loads athlete information from an Excel file and prepares the data for creation or update.

    This function checks if an athlete already exists in the database and generates a confirmation message.
    It also returns the necessary information for the operation.

    Parameters
    ----------
    df_lactate : pandas.DataFrame
        Data loaded from an Excel file, containing information about the athlete.

    Returns
    -------
    dict
        A dictionary containing the athlete's information and a confirmation message:
        - "text" (str) : Message to display to the user.
        - "mode" (str) : Operation mode ("new" for creation or "update" for update).
        - "first_name" (str) : Athlete's first name.
        - "last_name" (str) : Athlete's last name.
        - "date_of_birth" (date) : Athlete's date of birth.
        - "gender" (str) : Athlete's gender.
        - "date" (datetime) : Date of the test.
        - "team" (str) : Team name.
        - "height" (float) : Height in cm.
        - "weight" (float) : Weight in kg.
        - "sport" (str) : Sport practiced by the athlete.
    """

    # Create the database session
    session = next(get_db())
            
    try:

        # Data import from Excel
        if pd.isnull(df_lactate.iloc[4, 8]):
            raise ValueError("La date du test est manquante.")
        date_test = pd.to_datetime(df_lactate.iloc[4, 8], errors='coerce')

        # Retrieve and normalize basic information
        last_name_value = normalize_string(df_lactate.iloc[3, 2], "last_name")
        first_name_value = normalize_string(df_lactate.iloc[4, 2], "first_name")
        date_of_birth_value = pd.to_datetime(
            check_and_save("date de naissance", df_lactate, 3, 8), errors='coerce'
        ).date()
        gender_value = check_and_save("genre", df_lactate, 7, 2)

        # Search for the athlete in the database
        athlete = session.query(Athlete).filter(
            Athlete.first_name == first_name_value,
            Athlete.last_name == last_name_value,
            Athlete.date_of_birth == date_of_birth_value,
            Athlete.gender == gender_value
        ).first()

        # Determine gender for proper agreement
        e_gender = "" if gender_value in ("Homme", "Autre") else "e"
        le_gender = "" if gender_value in ("Homme", "Autre") else "le"
        
        # If the athlete does not exist, prepare the data for creation
        if athlete is None:
            mode = "new"
            height_value = check_and_save("taille", df_lactate, 5, 2)
            weight_value = check_and_save("poids", df_lactate, 6, 2)
            sport_value = check_and_save("sport", df_lactate, 5, 8).lower().strip()
            team_value = check_and_save("team", df_lactate, 6, 8).lower().strip()

            # Prevent if a similar athlete already exists
            all_athletes = get_all(session, Athlete)
            text = ""
            for athlete in all_athletes :
                if (
                    string_similarity(athlete.last_name, last_name_value) > 0.7
                    and string_similarity(athlete.first_name, first_name_value) > 0.7
                ):
                    text = (
                        f"Un{e_gender} nouvel{le_gender} athlète va être créé{e_gender} : "
                        f"{first_name_value.capitalize()} {last_name_value.capitalize()} "
                        f"né{e_gender} le {date.strftime(date_of_birth_value, '%d/%m/%Y')}.\n\n"
                        f"Un{e_gender} athlète similaire du nom de {athlete.first_name.capitalize()} "
                        f"{athlete.last_name.capitalize()} né{e_gender} le "
                        f"{date.strftime(athlete.date_of_birth, '%d/%m/%Y')} a été trouvé dans la base de données.\n\n"
                        f"Êtes-vous sûr d'avoir correctement orthographié son nom, son prénom, sa date de naissance, "
                        f"et d'avoir correctement spécifié son genre ?"
                    )
                    break

            if not text:
                text = (
                    f"Un{e_gender} nouvel{le_gender} athlète va être créé{e_gender} : {first_name_value.capitalize()} {last_name_value.capitalize()} "
                    f"né{e_gender} le {date.strftime(date_of_birth_value, '%d/%m/%Y')}.\n\n"
                    f"Êtes-vous sûr d'avoir correctement orthographié toutes ses informations personnelles ?"
                )

        else :
            # If the athlete exists, prepare the data for update
            mode = "update"
            height_value = df_lactate.iloc[5,2]
            weight_value = df_lactate.iloc[6, 2]
            sport_value = df_lactate.iloc[5, 8]
            team_value = df_lactate.iloc[6, 8]

            text = (
                "Les informations (poids, taille, team et/ou sport) de l'athlète suivant sont susceptibles d'être modifiées :\n\n"
                f"{first_name_value.capitalize()} {last_name_value.capitalize()} ({gender_value}) né{e_gender} le {date.strftime(date_of_birth_value, '%d/%m/%Y')}.\n\n"
                f"Êtes-vous sûr de vouloir continuer ?"
            )
        
        return {
            'text': text,
            'mode': mode,
            'first_name': first_name_value,
            'last_name': last_name_value,
            'date_of_birth': date_of_birth_value,
            'gender': gender_value,
            'date': date_test,
            'team': team_value,
            'height': height_value,
            'weight': weight_value,
            'sport': sport_value
        }
    
    except Exception as e:
        logging.error(f"Error in load_athlete_profile: {e}\n{traceback.format_exc()}")
        raise
    
def save_athlete_profile(athlete_data):
    """
    Saves an athlete profile in the database.

    This function creates a new athlete and associates them with a team.
    If the team does not exist in the database, it is created.

    Parameters
    ----------
    athlete_data : dict
        A dictionary containing the athlete's information to be saved.
        The expected keys are:
        - "first_name" (str) : Athlete's first name.
        - "last_name" (str) : Athlete's last name.
        - "height" (int or float) : Height in cm.
        - "weight" (int or float) : Weight in kg.
        - "date_of_birth" (str or datetime) : Date of birth (ISO format or datetime object).
        - "gender" (str) : Athlete's gender.
        - "team" (str) : Team name.
        - "sport" (str) : Sport practiced.

    Returns
    -------
    int
        Athlete id.
    """
    try:

        # Create the database session
        session = next(get_db())

        # Check for required fields
        required_fields = ["first_name", "last_name", "height", "weight", "date_of_birth", "gender", "team", "sport"]
        for field in required_fields:
            if field not in athlete_data or pd.isnull(athlete_data[field]):
                raise ValueError(f"Le champ '{field}' est manquant ou invalide.")
            
        # Handle the team: create or retrieve
        team_name = athlete_data["team"].strip().lower()
        team = session.query(Team).get(team_name) or create(session, Team, {"name": team_name})

        # Create the athlete
        athlete = create(session, Athlete, {
            "first_name": athlete_data["first_name"].strip(),
            "last_name": athlete_data["last_name"].strip(),
            "height": int(athlete_data["height"]),
            "weight": float(athlete_data["weight"]),
            "date_of_birth": pd.to_datetime(athlete_data["date_of_birth"]).date(),
            "gender": athlete_data["gender"].strip(),
            "team_name": team.name,
            "sport": athlete_data["sport"].strip().lower()
        })
        
        return athlete.id
        
    except Exception as e:
        logging.error(f"Error in save_athlete_profile: {e}\n{traceback.format_exc()}")
        raise
    
def update_athlete_profile(athlete_data):
    """
    Updates an athlete's profile in the database.

    This function checks if the provided information differs from the existing data 
    in the database. If differences are detected, the corresponding attributes are updated.

    Parameters
    ----------
    athlete_data : dict
        A dictionary containing the athlete's information to be updated. 
        The expected keys are: 
        - "first_name" (str) : Athlete's first name.
        - "last_name" (str) : Athlete's last name.
        - "date_of_birth" (str or datetime) : Date of birth (ISO format or datetime object).
        - "gender" (str) : Athlete's gender.
        - "height" (int or float, optional) : Height in cm.
        - "weight" (int or float, optional) : Weight in kg.
        - "team" (str, optional) : Team name.
        - "sport" (str, optional) : Sport practiced.

    Returns
    -------
    int
        Athlete id.
    """
    try:

        # Create the database session
        session = next(get_db())

        # Search for the athlete
        athlete = session.query(Athlete).filter(
            Athlete.first_name == athlete_data["first_name"].strip(),
            Athlete.last_name == athlete_data["last_name"].strip(),
            Athlete.date_of_birth == pd.to_datetime(athlete_data["date_of_birth"]).date(),
            Athlete.gender == athlete_data["gender"].strip()
        ).first()

        if not athlete:
            raise ValueError("Aucun athlète correspondant trouvé dans la base de données.")
        
        # List of attributes to check and update
        updatable_fields = {
            "height": lambda x: int(x),
            "weight": lambda x: int(x),
            "sport": lambda x: x.lower().strip()
        }

        # Update weight, height, and sport
        for field, transform in updatable_fields.items():
            if field in athlete_data and not pd.isnull(athlete_data[field]):
                new_value = transform(athlete_data[field])
                if getattr(athlete, field) != new_value:
                    setattr(athlete, field, new_value)

        # Handle team
        if "team" in athlete_data and not pd.isnull(athlete_data["team"]):
            team_name = athlete_data["team"].lower().strip()
            team = session.query(Team).get(team_name) or create(session, Team, {"name": team_name})
            if athlete.team_name != team.name:
                athlete.team_name = team.name

        # Save the changes
        session.commit()
        session.refresh(athlete)

        return athlete.id
    
    except Exception as e:
        session.rollback()
        logging.error(f"Error in update_athlete_profile: {e}\n{traceback.format_exc()}")
        raise

def string_similarity(str1, str2):
    """
    Calculates the similarity between two strings using 
    the Levenshtein distance.

    The Levenshtein distance measures the minimum number of operations 
    (insertions, deletions, substitutions) required to transform 
    one string into another. This function returns a similarity coefficient 
    between 0 (no similarity) and 1 (identical).

    Parameters:
    ----------
    str1 : str
        The first string to compare.
    str2 : str
        The second string to compare.

    Returns:
    -------
    float
        Similarity coefficient between 0 and 1.
    """
    try:
        if not str1 or not str2:
            raise ValueError("Les chaînes ne peuvent pas être vides.")

        distance = Levenshtein.distance(str1, str2)
        max_length = max(len(str1), len(str2))
        similarity_ratio = 1 - (distance / max_length)

        return similarity_ratio

    except Exception as e:
        logging.error(f"Error in string_similarity: {e}\n{traceback.format_exc()}")
        raise