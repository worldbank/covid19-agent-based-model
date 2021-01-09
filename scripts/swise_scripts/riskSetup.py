#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import geopandas as gpd
import os.path

from covid19_abm.dir_manager import get_data_dir


#
# SETUP
os.chdir("/Users/sophie/Documents/Github/covid19-agent-based-model/data")
cwd = os.getcwd()
print(cwd)

# define the relevant filenames

district_geo_filename = os.path.join('raw','shapefiles','new_districts/ZWE_adm2.shp')
district_risk_filename = os.path.join('raw','risk','district_hw_w_risk.csv')
district_severe_risk_filename = os.path.join('raw','risk','severe_disease_risk_district.csv')

output_filename = 'preprocessed/risk/hw_and_severe_disease_risk.csv'

# identify the columns of interest in the given datasets

hw_cols = ['NAME_2', 'mean_hw_risk_pop_weighted', 'mean_w_risk_pop_weighted']
severe_cols = ['severe_covid_risk', 'severe_covid_risk_improved_1', 'severe_covid_risk_improved_2']

# read in the data

print("Beginning to read in data...")
print("From file " + district_geo_filename + "...")
new_district = gpd.read_file(district_geo_filename)

print("From file " + district_risk_filename + "...")
hw_risk = pd.read_csv(district_risk_filename)

print("From file " + district_severe_risk_filename + "...")
severe_disease_risk = pd.read_csv(district_severe_risk_filename)

print("Finished reading in risk data!")

# merge files based on district names

print("Processing risk data...")
hw_risk = hw_risk.merge(new_district[['NAME_2', 'ID_2']], on='NAME_2').set_index('ID_2')
severe_disease_risk = severe_disease_risk.merge(new_district[['NAME_2', 'ID_2']], on='NAME_2').set_index('ID_2')

# combine datasets

hw_severe_disease_risk = pd.concat([hw_risk[hw_cols], severe_disease_risk[severe_cols]], axis=1)

# export the final dataset

hw_severe_disease_risk.to_csv(output_filename)
print("Successfully exported risk data to " + output_filename)