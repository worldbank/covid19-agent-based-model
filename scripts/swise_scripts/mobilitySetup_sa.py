#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime
import os.path 

from covid19_abm.dir_manager import get_data_dir

#
### SETUP
os.chdir("/Users/sophie/Documents/Github/covid19-agent-based-model/data/preprocessed/mobility")
cwd = os.getcwd()
print(cwd)


# define mobility intput/output filenames

mobility_filename = os.path.join('month_home_vs_day_location_per_day-20200424.csv')
#we are not using this file above anymore, this needs to be changes for different set of indicators
lockdownMobility_filename = os.path.join('week_home_vs_day_location_febapr.csv')

# outputs

mobility_output = os.path.join('mobility_output','daily_region_transition_probability-new-district.csv')
weekday_mobility_filename = os.path.join('mobility_output','weekday_mobility_duration_count_df-new-district.pickle')

prelockdown_output = os.path.join('mobility_output','daily_region_transition_probability-new-district-pre-lockdown.csv')

mobility_postlockdown_output = os.path.join('mobility_output','daily_region_transition_probability-new-district-post-lockdown.csv')

###
### DEFINE METHODS
### define methods that will be used multiple times for different datasets
###

#
# READ IN THE MOBILITY MATRIX
#
def readInMobility(filename):

	print("Reading in file: " + filename)

	# read in the raw mobility filename
	mobility_df = pd.read_csv(filename)

	print("Processing file...")

	# drop any empty lines
	mobility_df = mobility_df.dropna(axis=0)

	# exclude any individuals with home regions great than 100
	# @Sarah I think this was something to do with not wanting to allow people to be in their home region more than 100% of the time. Basically it was something to do with the way that the indicator was calculated that it showed up like this. 
	mobility_df = mobility_df[mobility_df['home_region'] < 100]

	# convert region names from integers to 'd_[regionNumber]' formats
	mobility_df['region'] = mobility_df['region'].astype(int).astype(str).map(lambda x: f'd_{x}')
	mobility_df['home_region'] = mobility_df['home_region'].astype(int).astype(str).map(lambda x: f'd_{x}')
	
	# in some inherited datasets, dates were rendered differently: harmonise them
	if 'date2' in mobility_df.columns:
		print("converting from date2 to datetimes...")
		mobility_df['day'] = pd.to_datetime(mobility_df['date2'])

	# acquire sorted sets of all of the regions, home regions, and days represented in the dataset
	region = sorted(mobility_df['region'].unique())
	home_region = sorted(mobility_df['home_region'].unique())
	days = sorted(mobility_df['day'].unique())

	# create the matrix to hold flow information
	expanded_idx = [(
	    day,
	    h_reg,
	    reg) for day in days for h_reg in home_region for reg in region]

	# index the dataframe based on the location within the matrix
	# at the same time, fill any empty cells to reflect the absence of data for such a journey
	mobility_df = mobility_df.set_index(
	    ['day', 'home_region', 'region']).reindex(expanded_idx).fillna(0)

	# recalculate the index
	mobility_df = mobility_df.reset_index()

	#mobility_df[mobility_df['home_region'].isnull()]

	#######
	print("Entries with null home_region values: ")
	print(mobility_df[mobility_df['home_region'].isnull()])
	
	# set up a few extra attributes
	mobility_df['date'] = pd.to_datetime(mobility_df['day'])
	mobility_df['weekday'] = mobility_df.date.dt.weekday

	return mobility_df

#
# CALCULATE TRANSITION RATE
#
def calculateTransitions(mobility_data, outputfilename, 
	mindate=datetime.datetime.fromisoformat('1970-01-01 00:00:00+00:00'), 
	maxdate=datetime.datetime.now(datetime.timezone.utc)):	

	## summarise movement by date and home region, per destination region

	print(maxdate)
	print(mindate)

	mobility_df = mobility_data[ (mobility_data['date'] >= mindate) & (mobility_data['date'] <= maxdate)]

	print(mobility_df.head())

	# number of recorded people from each home region, on any given date
	daily_population_subscribers = mobility_df.groupby(['date', 'home_region'])['count'].sum()

	# number of recorded people who move between home_region and region, on any given date
	day_src_dst_location_count = mobility_df.groupby(['date', 'home_region', 'region'])['count'].sum()

	# normalise the likelihood of moving from a home region to any given region
	daily_region_transition_probability = day_src_dst_location_count.divide(
	    daily_population_subscribers
	).groupby(level=['date', 'home_region']).cumsum()

	# add the date/home_region combo as its own column
	daily_region_transition_probability = daily_region_transition_probability.reset_index()
	daily_region_transition_probability['weekday'] = daily_region_transition_probability['date'].dt.weekday

	# rename columns
	daily_region_transition_probability.rename(columns={'count': 'prob'}, inplace=True)

	# take the average for each date/home region combination
	daily_region_transition_probability = daily_region_transition_probability.groupby(['weekday', 'home_region', 'region'], sort=True)['prob'].mean()

	# pivot on region
	daily_region_transition_probability = daily_region_transition_probability.unstack('region')

	# export to CSV
	daily_region_transition_probability.to_csv(outputfilename)
	print("Successfully output daily region transition probability as CSV to " + outputfilename)


# presently the file outputs to here only. this doesn't matter because we are going to change the input files anyway

# CALCULATE DURATION
#
def calculateDurationData(mobility_data, outputfilename):

	print("Beginning duration calculation...")

	mobility_df = mobility_data.copy()

	mobility_df.rename(columns={'mean_duration': 'avg_duration', 'stdev_duration': 'stddev_duration'}, inplace=True)

	#######

	regions = sorted(mobility_df['home_region'].unique())
	full_idx = [(dow, src, dst) for dow in range(7) for src in regions for dst in regions]

	weekday_mobility_duration_df = (
	    mobility_df.groupby(
	        ['weekday', 'home_region', 'region']
	    )[['avg_duration', 'stddev_duration']].mean()
	)
	weekday_mobility_duration_df[['avg_duration', 'stddev_duration']] = weekday_mobility_duration_df[['avg_duration', 'stddev_duration']] / (60 * 60)

	weekday_mobility_duration_df = weekday_mobility_duration_df.reindex(full_idx)
	weekday_mobility_duration_df['avg_duration'] = weekday_mobility_duration_df['avg_duration'].fillna(24)
	weekday_mobility_duration_df['stddev_duration'] = weekday_mobility_duration_df['stddev_duration'].fillna(weekday_mobility_duration_df['stddev_duration'].mean())

	#######
	print("Exporting mobility duration data...")
	weekday_mobility_duration_df.to_pickle(outputfilename)
	print("Successfully output duration data as .pickle to " + outputfilename)

### BEGIN CALLING FUNCTIONS HERE

myMobilityDF = readInMobility(mobility_filename)
calculateTransitions(myMobilityDF, mobility_output)
calculateDurationData(myMobilityDF, weekday_mobility_filename)

# pre-lockdown
startOfLockdown = datetime.datetime(2020, 3, 20, tzinfo=datetime.timezone.utc)
mobility2020_DF = readInMobility(mobility_filename)#lockdownMobility_filename)
calculateTransitions(mobility2020_DF, prelockdown_output, maxdate=startOfLockdown)

# during lockdown
start_lockdown_date = datetime.datetime(2020, 3, 1, tzinfo=datetime.timezone.utc)
end_lockdown_date = datetime.datetime(2020, 5, 1, tzinfo=datetime.timezone.utc)
calculateTransitions(mobility2020_DF, mobility_postlockdown_output, mindate=start_lockdown_date, maxdate=end_lockdown_date)
