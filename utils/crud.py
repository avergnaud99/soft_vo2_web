'''
This module creates the Create, Read, Update, and Delete functions for the Database.
'''

### Imports ###

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import Type, Any, List
from utils.models import *

### Methods ###

def create(db: Session, model: Type[Base], obj_in: dict) -> Any: # type: ignore
    """
    Creates a new object in the database.
    
    Parameters
    ----------
    db : Session
        SQLAlchemy session.
    model : Type[Base]
        The model on which to perform the CRUD operation.
    obj_in : dict
        The data to insert into the object.
    
    Returns
    -------
    Any
        The created object.

    Raises
    ------
    Exception
        If an error occurs while creating the object.
    """

    try:

        # Create the object
        db_obj = model(**obj_in)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)

        return db_obj
    
    except SQLAlchemyError as e:

        # Rollback in case of error
        db.rollback()  

        raise Exception(f"Error while creating the object: {str(e)}")

def get_all(db: Session, model: Type[Base]) -> List[Any]: # type: ignore
    """
    Retrieves all objects of the specified model.
    
    Parameters
    ----------
    db : Session
        SQLAlchemy session.
    model : Type[Base]
        The model from which to retrieve the objects.
    
    Returns
    -------
    List[Any]
        List of retrieved objects.

    Raises
    ------
    Exception
        If an error occurs while retrieving the objects.
    """

    try:
        return db.query(model).all()
    
    except SQLAlchemyError as e:
        raise Exception(f"Error while retrieving objects: {str(e)}")

def get_by_id(db: Session, model: Type[Base], object_id: int) -> Any: # type: ignore
    """
    Retrieves an object by its ID.
    
    Parameters
    ----------
    db : Session
        SQLAlchemy session.
    model : Type[Base]
        The model on which to perform the search.
    object_id : int
        The ID of the object to retrieve.
    
    Returns
    -------
    Any
        The object corresponding to the ID or None if not found.

    Raises
    ------
    Exception
        - If an error occurs while retrieving the object.
        - If the object with the given ID does not exist.
    """

    try:

        db_obj = db.query(model).filter(model.id == object_id).first()

        if db_obj is None:
            raise Exception(f"Object with ID {object_id} not found.")
        
        return db_obj
    
    except SQLAlchemyError as e:
        raise Exception(f"Error while retrieving the object: {str(e)}")

def update(db: Session, model: Type[Base], object_id: int, obj_in: dict) -> Any: # type: ignore
    """
    Updates an existing object in the database.
    
    Parameters
    ----------
    db : Session
        SQLAlchemy session.
    model : Type[Base]
        The model on which to perform the update.
    object_id : int
        The ID of the object to update.
    obj_in : dict
        The new data to update in the object.
    
    Returns
    -------
    Any
        The updated object.

    Raises
    ------
    Exception
        - If the object with the given ID does not exist.
        - If an error occurs while updating the object.
    """

    try:

        # Retrieve the object
        db_obj = db.query(model).filter(model.id == object_id).first()

        if db_obj is None:
            raise Exception(f"Object with ID {object_id} not found for update.")
        
        # Update the object
        for key, value in obj_in.items():
            setattr(db_obj, key, value)
        
        db.commit()
        db.refresh(db_obj)

        return db_obj
    
    except SQLAlchemyError as e:
        db.rollback()
        raise Exception(f"Error while updating the object: {str(e)}")

def delete(db: Session, model: Type[Base], object_id: int) -> bool: # type: ignore
    """
    Deletes an object from the database by its ID.
    
    Parameters
    ----------
    db : Session
        SQLAlchemy session.
    model : Type[Base]
        The model on which to perform the deletion.
    object_id : int
        The ID of the object to delete.
    
    Returns
    -------
    bool
        True if the deletion was successful, otherwise False.

    Raises
    ------
    Exception
        - If the object with the given ID does not exist.
        - If an error occurs while deleting the object.
    """

    try:

        # Retrieve the object
        db_obj = db.query(model).filter(model.id == object_id).first()

        if db_obj is None:
            raise Exception(f"Object with ID {object_id} not found for deletion.")
        
        # Delete the object
        db.delete(db_obj)
        db.commit()

        return True
    
    except SQLAlchemyError as e:
        db.rollback()
        raise Exception(f"Error while deleting the object: {str(e)}")

