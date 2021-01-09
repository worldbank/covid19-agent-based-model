#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import numpy as np
from collections import Counter
import os.path
import sys
from pathlib import Path
from covid19_abm.dir_manager import get_data_dir

#sys.exit()
# This is how to set the directory - 
os.chdir("/Users/sophie/Documents/Github/covid19-agent-based-model/data")
cwd = os.getcwd()
print(cwd)

# define the relevant filenames

census_filename =os.path.join('raw', 'census', 'census_dummy_0.001_pct.dta')
#'ABM_Simulated_Pop_WardDistributed_UpdatedMay30_school_complete_060520.dta'
district_filename = os.path.join('raw','district_relation.csv')
output_filename = os.path.join('preprocessed', 'census', 'dummy_100pct.pickle')

# set up mappings between the input data and the values used by the census builder

age_map = {
    'less than 1 year': 0,
    '1 year': 1,
    '2 years': 2,
    '98': 98,
    'not reported/missing': None,
}

econ_stat_map = {
    'Not working, inactive, not': 'Not working, inactive, not in universe',
}

#
# BEGIN READING IN DATA
#

full_individual_df = pd.read_stata(census_filename)

print("Successfully read in file...")

# map the input data to the mappings defined above
full_individual_df['age'] = full_individual_df['age'].map(lambda x: age_map.get(x.strip(), x.strip()))
full_individual_df['economic_status'] = full_individual_df['economic_status'].str.strip().map(lambda x: econ_stat_map.get(x, x))

# read the full set of available economic statuses and print this out for the user to see
print("\n===Count of Individual Economic Statuses===")
l = full_individual_df['economic_status'].unique()
print(full_individual_df['economic_status'].value_counts()[l])

#
# DEAL WITH MISSING AGES
#

individual_df = full_individual_df

# extra the columns on which age will be predicted
age_cols = ['geo1_zw2012', 'urban', 'persons', 'sex', 'marst', 'citizen', 'race', 'disabled', 'economic_status']
# missing_val = 'not reported/missing'
X = pd.get_dummies(individual_df[age_cols], drop_first=True)

# extract the training set - the set of individuals who already have ages
X_train = X[individual_df['age'].notnull()]

# set up the regressor and train it on the sample data
gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=1029)
# rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=1029, n_jobs=50)
frac = 0.05
X_tr = X_train.sample(frac=frac, random_state=1029)
gb.fit(X_tr, individual_df.loc[X_tr.index, 'age'])

# now pull out the individuals with null ages and replace their ages with regressed integer values
missing_age = X[individual_df['age'].isnull()]
individual_df.loc[missing_age.index, 'age'] = gb.predict(missing_age).astype(int)

print("\nAll missing ages regressed and predicted successfully...")

#
# SET UP LOCATION-BASED IDs
#

relevant_cols = [
    'person_id', 'age', 'sex',
    'household_id', 'district_id',
    'economic_status', 'economic_activity_location_id',
    # 'household', 'district'
]

old_new_districts = pd.read_csv(district_filename, index_col=0)
old_new_districts.index = old_new_districts['ADMIN_NAME'].str.lower()

individual_df['serial_expanded'] = individual_df['serial']

individual_df['household_id'] = individual_df['serial_expanded'].map(lambda x: f'h_{x}')
individual_df['old_district_id'] = individual_df['geo2_zw2012'].map(lambda x: f'd_{old_new_districts["DIST2012"][x]}')                                          
individual_df['new_district_id'] = individual_df['geo2_zw2012'].map(lambda x: f'd_{old_new_districts["NEW_DIST_ID_2"][x]}')

expanded_individual_df = individual_df.copy()

expanded_individual_df['school_goers'] = 1 * (expanded_individual_df['school_goers'] != 0)


#
# MINING SECTION
#

relevant_cols = [
    'person_id', 'age', 'sex',
    'household_id', 'district_id',
    'economic_status', 'economic_activity_location_id',
    # 'school_id_district', 
    'school_goers',
    'manufacturing_workers',
    # 'mining_district_id'
    # 'household', 'district'
]

mining_df = expanded_individual_df.copy()

p = mining_df['economic_status']

mining_df.loc[p == 'Disabled and not working', 'economic_activity_location_id'] = mining_df.loc[p == 'Disabled and not working', 'household_id']
mining_df.loc[p != 'Disabled and not working', 'economic_activity_location_id'] = mining_df.loc[p != 'Disabled and not working', 'new_district_id']

try:
    mining_df.drop('person_id', axis=1, inplace=True)
except KeyError:
    pass

mining_df['person_id'] = mining_df.index
mining_df['age'] = mining_df['age'].astype(int)
mining_df['economic_status'] = mining_df['economic_status'].str.strip()
mining_df.rename(columns={'new_district_id': 'district_id'}, inplace=True)

print(mining_df[relevant_cols].head())

#
# WRITING OUT TO THE PICKLE FILES
#

print("Writing out to file...")

mining_df[relevant_cols].to_pickle(output_filename)

# want to only export a subset? Uncomment these!
#mining_df[relevant_cols][mining_df.serial_expanded % 100 < 5].to_pickle( get_data_dir(output_filename) + '_5pct.pickle')
#mining_df[relevant_cols][mining_df.serial_expanded % 100 < 10].to_pickle( get_data_dir(output_filename) + '_10pct.pickle')

print("FINISHED!")

#mining_df[relevant_cols][
#    mining_df.serial_expanded.str.endswith('_01')
#].to_pickle( get_data_dir(output_5pct_filename) + '_5pct.pickle')

#if __name__ == '__main__':
#    main()
