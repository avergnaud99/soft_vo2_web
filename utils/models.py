'''
This module contains the model classes for the SQLAlchemy database

Migration terminal command lines
` alembic revision --autogenerate -m "Name of the migration" `
` alembic upgrade head `
'''

### Imports ###

from sqlalchemy import Column, Integer, String, ForeignKey, Date, JSON, Float, CheckConstraint, BLOB
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

### Main ###

Base = declarative_base()

### Classes ###

class Test(Base):
    """ A class to represent the Test table of the database """

    # Name of the table
    __tablename__ = 'test'

    # Test most important attributes
    id = Column(Integer, primary_key = True)
    athlete_id = Column(Integer, ForeignKey('athlete.id'), nullable = False)
    date = Column(Date, nullable = False)
    weight = Column(Float)
    source_excel = Column(JSON, nullable = False)
    source_vo2 = Column(JSON, nullable = False)

    # Relationships
    athlete = relationship("Athlete", back_populates = "tests")

    # Test dataframes and remarks
    computed_dataframe = Column(JSON)
    plateau_dataframe = Column(JSON)
    remarks = Column(String)

    # PDF & plots
    pdf_report = Column(BLOB)
    graphics = Column(BLOB)
    protocol = Column(BLOB)
    plateau_study = Column(BLOB)

    # Plateau analysis
    time_under_trend_plateau = Column(Integer)
    time_above_trend_plateau = Column(Integer)
    total_time = Column(Integer)
    area_under_trend_plateau = Column(Integer)
    area_above_trend_plateau = Column(Integer)
    total_area_plateau = Column(Integer)
    start_speed_plateau = Column(Float)
    shape_plateau = Column(String)

    # First threshold values
    slope_s1 = Column(Float)
    speed_s1 = Column(Float)
    vo2_s1 = Column(Float)
    vo2_kg_s1 = Column(Float)
    vo2_ratio_s1 = Column(Float)
    ve_s1 = Column(Float)
    hr_s1 = Column(Integer)
    hr_ratio_s1 = Column(Float)
    watt_s1 = Column(Float)
    watt_kg_s1 = Column(Float)
    de_s1 = Column(Float)
    glu_s1 = Column(Float)
    lip_s1 = Column(Float)
    lactate_s1 = Column(Float)

    # Second threshold values
    slope_s2 = Column(Float)
    speed_s2 = Column(Float)
    vo2_s2 = Column(Float)
    vo2_kg_s2 = Column(Float)
    vo2_ratio_s2 = Column(Float)
    ve_s2 = Column(Float)
    hr_s2 = Column(Integer)
    hr_ratio_s2 = Column(Float)
    watt_s2 = Column(Float)
    watt_kg_s2 = Column(Float)
    de_s2 = Column(Float)
    glu_s2 = Column(Float)
    lip_s2 = Column(Float)
    lactate_s2 = Column(Float)

    # Maximal values
    tps_max = Column(Integer)
    slope_max = Column(Float)
    speed_max = Column(Float)
    vo2_max = Column(Float)
    vo2_kg_max = Column(Float)
    vo2_ratio_max = Column(Float)
    ve_max = Column(Float)
    hr_max = Column(Integer)
    hr_ratio_max = Column(Float)
    watt_max = Column(Float)
    watt_kg_max = Column(Float)
    de_max = Column(Float)
    glu_max = Column(Float)
    lip_max = Column(Float)
    lactate_max = Column(Float)

class Athlete(Base):
    """ A class to represent the Athlete table of the database """

    # Name of the table
    __tablename__ = 'athlete'

    # Attributes
    id = Column(Integer, primary_key = True)
    first_name : str = Column(String, nullable = False)
    last_name : str = Column(String, nullable = False)
    date_of_birth = Column(Date, nullable = False)
    gender = Column(String, nullable = False)
    team_name = Column(String, ForeignKey('team.name'))
    sport = Column(String)
    height = Column(Integer)
    weight = Column(Float, nullable = False)

    # Relationships
    team = relationship("Team", back_populates = "athletes")
    tests = relationship("Test", back_populates = "athlete")

    # Special arguments for the gender
    __table_args__ = (
        CheckConstraint(gender.in_(['Homme', 'Femme', 'Autre']), name = 'valid_gender'),
    )

class Team(Base):
    """ A class to represent the Team table of the database """

    # Name of the table
    __tablename__ = 'team'

    # Attributes
    name = Column(String, primary_key = True)

    # Relationships
    athletes = relationship("Athlete", back_populates = "team")
