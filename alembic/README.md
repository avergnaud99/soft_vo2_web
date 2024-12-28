# Database Schema

## Overview

This README provides an overview of the database schema used in the project. It describes the tables, columns, and relationships.

## Tables

### alembic_version

- Description: TODO
- Columns:
  - `version_num`: VARCHAR(32)

### athlete

- Description: TODO
- Columns:
  - `id`: INTEGER
  - `first_name`: VARCHAR
  - `last_name`: VARCHAR
  - `date_of_birth`: DATE
  - `gender`: VARCHAR
  - `team_name`: VARCHAR
  - `sport`: VARCHAR
  - `height`: INTEGER
  - `weight`: FLOAT
- Relationships:
  - `['team_name']` references `team (['name'])`

### team

- Description: TODO
- Columns:
  - `name`: VARCHAR

### test

- Description: TODO
- Columns:
  - `id`: INTEGER
  - `athlete_id`: INTEGER
  - `date`: DATE
  - `source_excel`: JSON
  - `source_vo2`: JSON
  - `computed_dataframe`: JSON
  - `plateau_dataframe`: JSON
  - `pdf_report`: BLOB
  - `vo2_s1`: FLOAT
  - `vo2_kg_s1`: FLOAT
  - `vo2_ratio_s1`: FLOAT
  - `ve_s1`: FLOAT
  - `hr_s1`: INTEGER
  - `hr_ratio_s1`: FLOAT
  - `glu_s1`: FLOAT
  - `lip_s1`: FLOAT
  - `de_s1`: FLOAT
  - `lactate_s1`: FLOAT
  - `vo2_s2`: FLOAT
  - `vo2_kg_s2`: FLOAT
  - `vo2_ratio_s2`: FLOAT
  - `ve_s2`: FLOAT
  - `hr_s2`: INTEGER
  - `hr_ratio_s2`: FLOAT
  - `glu_s2`: FLOAT
  - `lip_s2`: FLOAT
  - `de_s2`: FLOAT
  - `lactate_s2`: FLOAT
  - `vo2_max`: FLOAT
  - `vo2_kg_max`: FLOAT
  - `vo2_ratio_max`: FLOAT
  - `ve_max`: FLOAT
  - `hr_max`: INTEGER
  - `hr_ratio_max`: FLOAT
  - `glu_max`: FLOAT
  - `lip_max`: FLOAT
  - `de_max`: FLOAT
  - `lactate_max`: FLOAT
  - `remarks`: VARCHAR
  - `graphics`: BLOB
  - `protocol`: BLOB
  - `plateau_study`: BLOB
  - `slope_s1`: FLOAT
  - `speed_s1`: FLOAT
  - `watt_s1`: FLOAT
  - `watt_kg_s1`: FLOAT
  - `slope_s2`: FLOAT
  - `speed_s2`: FLOAT
  - `watt_s2`: FLOAT
  - `watt_kg_s2`: FLOAT
  - `tps_max`: INTEGER
  - `slope_max`: FLOAT
  - `speed_max`: FLOAT
  - `watt_max`: FLOAT
  - `watt_kg_max`: FLOAT
  - `time_under_trend_plateau`: INTEGER
  - `time_above_trend_plateau`: INTEGER
  - `total_time`: INTEGER
  - `area_under_trend_plateau`: INTEGER
  - `area_above_trend_plateau`: INTEGER
  - `total_area_plateau`: INTEGER
  - `start_speed_plateau`: FLOAT
  - `weight`: FLOAT
- Relationships:
  - `['athlete_id']` references `athlete (['id'])`

