import os
import sys
import pylab as plt

import mesa
from mesa import Agent, Model
from mesa.time import RandomActivation, BaseScheduler
from mesa.datacollection import DataCollector
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
# import seaborn as sns
from collections import Counter
from covid19_abm.params import log_to_file
import json


class EpidemicScheduler(BaseScheduler):
    '''
    This is the scheduler class that manages the step method for different agents in the model.
    '''
    def __init__(self, model):
        super().__init__(model)
        self.real_time = self.model.start_datetime
        self.step_timedelta = self.model.step_timedelta

        self.weekday_move_hours = set([self.model.params.WEEKDAY_START_DAY_HOUR, self.model.params.WEEKDAY_END_DAY_HOUR])
        self.other_day_move_hours = set([self.model.params.OTHER_DAY_START_DAY_HOUR, self.model.params.OTHER_DAY_END_DAY_HOUR])
        self.day_location_map = {}

    def step(self):
        current_time = self.real_time

        self.model.log_model_output()

        self.update_agent_states()

        ################################## START: Movement logic  ##################################
        # Logic to make people return to their houses at the end of the day
        if self.real_time.hour in [self.model.params.WEEKDAY_END_DAY_HOUR, self.model.params.OTHER_DAY_END_DAY_HOUR]:
            self.go_home()

        # Logic to implement movement
        if self.real_time.hour in [self.model.params.WEEKDAY_START_DAY_HOUR, self.model.params.OTHER_DAY_START_DAY_HOUR]:
            self.go_out()

        ################################## END: Movement logic  ##################################


        ################################## START: Epidemic transmission logic  ##################################
        # Get set of newly contagious individuals
        new_contagious_ids = np.where(
            (self.model.date_start_contagious < current_time) &
            (self.model.epidemic_state < self.model.STATE_CONTAGIOUS)
        )[0]
        self.model.epidemic_state[new_contagious_ids] = self.model.STATE_CONTAGIOUS

        contagious_ids = np.where(self.model.epidemic_state == self.model.STATE_CONTAGIOUS)[0]
        np.random.shuffle(contagious_ids)

        contagious_current_location_ids = self.model.current_location_ids[contagious_ids]
        contagious_economic_status_ids = self.model.economic_status_ids[contagious_ids]
        unique_contagious_current_location_ids = np.unique(contagious_current_location_ids)
        unique_contagious_economic_status_ids = np.unique(contagious_economic_status_ids)

        location_person_ids = self.model.get_location_person_ids(self.model.current_location_ids, None)
        contagious_location_person_ids = self.model.get_location_person_ids(contagious_current_location_ids, contagious_ids)

        unique_economic_status_ids = set(self.model.params.ECON_STAT_ID_TO_NAME)
        num_econ_stats = len(unique_economic_status_ids)
        econ_stats_tile = np.arange(num_econ_stats).reshape(num_econ_stats, 1, 1)
        econ_stat_cids_cache = {}

        for loc in unique_contagious_current_location_ids:
            loc_cids = contagious_location_person_ids[loc]
            potential_neighbors = location_person_ids[loc]

            if loc not in self.model.params.HOUSEHOLD_ID_TO_NAME:

                econ_neighbors_map = {
                    es: list(
                        self.model.economic_status_ids_person_ids[es].intersection(potential_neighbors)
                    ) for es in unique_contagious_economic_status_ids}

                for econ_stat_id in unique_contagious_economic_status_ids:
                    if econ_stat_id not in econ_stat_cids_cache:
                        econ_stat_cids_cache[econ_stat_id] = set(contagious_ids[(contagious_economic_status_ids == econ_stat_id)])

                    cont_person_ids = np.array(list(econ_stat_cids_cache[econ_stat_id].intersection(loc_cids)), dtype=int)
                    num_contagious = cont_person_ids.size

                    if num_contagious == 0:
                        continue

                    econ_stat = self.model.params.ECON_STAT_ID_TO_NAME[econ_stat_id]
                    num_interactions = self.model.params.ECONOMIC_STATUS_INTERACTION_SIZE_MAP[econ_stat]

                    neighbors_matrix = np.full(shape=(num_contagious, num_interactions), fill_value=-1)
                    neighbors_offset = np.zeros(num_contagious, dtype=int)

                    interaction_economic_status_ids = np.random.choice(
                        num_econ_stats,
                        (num_contagious, num_interactions),
                        replace=True,
                        p=self.model.params.ECONOMIC_STATUS_INTERACTION_MATRIX_VALUES[econ_stat_id, :]
                    )

                    num_neighbors_by_econ_stat = (
                        interaction_economic_status_ids.reshape(1, -1, num_interactions) ==
                        econ_stats_tile).sum(axis=2).T

                    if num_neighbors_by_econ_stat.size == 0:
                        continue

                    for es in unique_economic_status_ids:
                        econ_neighbors = econ_neighbors_map.get(es, [])
                        num_econ_neighbors = len(econ_neighbors)

                        if num_econ_neighbors == 0:
                            continue

                        np.random.shuffle(econ_neighbors)

                        # Pad to handle edge case
                        ext = econ_neighbors[:num_interactions]
                        if len(ext) < num_interactions:
                            ext = econ_neighbors * num_interactions
                        econ_neighbors.extend(ext)
                        size = num_neighbors_by_econ_stat[:, es]

                        # Get random indices for slicing and use num_neighbors_by_econ_stat as offset
                        inds_start = np.random.randint(0, num_econ_neighbors, num_contagious)
                        inds_end = inds_start + size

                        for cix, (st, en, n_start, start_offset) in enumerate(zip(inds_start, inds_end, neighbors_offset, size)):
                            neighbors_matrix[cix, n_start:n_start + start_offset] = econ_neighbors[st:en]

                        neighbors_offset += size

                    infection_rates = self.get_infection_rates(cont_person_ids, location='outside')
                    neighbors_infect_probs = np.random.random(size=neighbors_matrix.shape)

                    neighbors_to_infect = neighbors_matrix[neighbors_infect_probs < infection_rates]

                    # Remove unfilled elements
                    neighbors_to_infect = neighbors_to_infect[neighbors_to_infect > -1]

                    if neighbors_to_infect.size > 0:
                        neighbors_to_infect = neighbors_to_infect[
                            (self.model.epidemic_state[neighbors_to_infect] < self.model.STATE_INFECTED)
                        ]

                        self.model.set_epidemic_status(neighbors_to_infect)
            else:
                house_contagious_ids = loc_cids
                if house_contagious_ids.size == 0:
                    continue

                neighbors = potential_neighbors

                neighbors = np.tile(neighbors.reshape(1, -1), (len(house_contagious_ids), 1))

                if neighbors.size > 0:
                    infection_rates = self.get_infection_rates(house_contagious_ids, location='house')
                    neighbors_infect_probs = np.random.random(size=neighbors.shape)

                    neighbors_to_infect = neighbors[neighbors_infect_probs < infection_rates]

                    neighbors_to_infect = neighbors_to_infect[
                        (self.model.epidemic_state[neighbors_to_infect] < self.model.STATE_INFECTED)
                    ]

                    self.model.set_epidemic_status(neighbors_to_infect)

        self.steps += 1
        self.time += 1
        self.real_time += self.step_timedelta

    def get_infection_rates(self, contagious_ids, location):
        infection_rates = self.model.infection_rate[contagious_ids].reshape(contagious_ids.size, 1)

        if location == 'house':
            # Adjust here since the average contact in household is ~3x less than the contacts outside.
            # Average household size is 4 while age structured contact rate is ~11.
            infection_rates *= 3

        if self.model.params.SCENARIO.startswith('HANDWASHING_RISK'):
            # Since we assume that all people are in the same location, the district information should be the same.
            infection_rates *= self.model.hw_risk[self.model.current_ward_ids[contagious_ids[0]]]

        return infection_rates

    def get_active_ids(self, return_type=None):
        current_time = self.real_time

        # Consider all people that haven't been hospitalized or those that have recovered to be active for moving.
        active_ids = np.where(
            # Everyone that haven't been hospitalized (healthy and infected)
            (self.model.clinical_state < self.model.CLINICAL_HOSPITALIZED) |
            # Everyone that recovered
            (self.model.epidemic_state == self.model.STATE_RECOVERED)
        )[0]

        active_ids = active_ids[np.where(
            # Consider only movement for valid locations
            self.model.current_location_ids[active_ids] >= 0
        )]

        # Assume that all symptomatic can still move with some probability unless they start
        if self.model.params.MILD_SYMPTOM_MOVEMENT_PROBABILITY < 1:
            symptomatic_not_recovered = active_ids[np.where(
                (self.model.date_start_symptomatic[active_ids] < current_time) &
                (self.model.epidemic_state[active_ids] < self.model.STATE_RECOVERED)
            )]
            self.model.mild_symptom_movement_probability[symptomatic_not_recovered] = (
                self.model.params.MILD_SYMPTOM_MOVEMENT_PROBABILITY
            )

        if return_type is not None:
            active_ids = return_type(active_ids)

        return active_ids

    def update_agent_states(self):
        current_time = self.real_time

        # Update dead agents
        dead_ids = np.where(
            (self.model.date_died < current_time) &
            (self.model.epidemic_state < self.model.STATE_DEAD)
        )[0]
        self.model.epidemic_state[dead_ids] = self.model.STATE_DEAD
        self.model.current_location_ids[dead_ids] = self.model.DEAD_LOCATION_ID
        # Set the clinical state to dead.
        self.model.clinical_state[dead_ids] = self.model.CLINICAL_RELEASED_OR_DEAD

        # Update recovered agents
        recovered_ids = np.where(
            (self.model.date_recovered < current_time) &
            (self.model.epidemic_state < self.model.STATE_RECOVERED)
        )[0]
        self.model.epidemic_state[recovered_ids] = self.model.STATE_RECOVERED
        # If person is hospitalized set place to house when recovered
        hospitalized_ids = np.where(self.model.date_start_hospitalized < current_time)[0]
        just_recovered_hospitalized_ids = list(set(recovered_ids).intersection(hospitalized_ids))
        self.model.current_location_ids[just_recovered_hospitalized_ids] = (
            self.model.household_ids[just_recovered_hospitalized_ids]
        )
        # Set the clinical state of all recovered regardless whether they were hospitalized as released.
        self.model.clinical_state[recovered_ids] = self.model.CLINICAL_RELEASED_OR_DEAD
        # Let recovered persons be fully mobile
        self.model.mild_symptom_movement_probability[recovered_ids] = 1

        # Update critical agents
        critical_ids = np.where(
            (self.model.date_start_critical < current_time) &
            (self.model.clinical_state < self.model.CLINICAL_CRITICAL)
        )[0]
        self.model.clinical_state[critical_ids] = self.model.CLINICAL_CRITICAL
        # Update hospitalized agents
        hospitalized_ids = np.where(
            (self.model.date_start_hospitalized < current_time) &
            (self.model.clinical_state < self.model.CLINICAL_HOSPITALIZED)
        )[0]
        self.model.clinical_state[hospitalized_ids] = self.model.CLINICAL_HOSPITALIZED
        # Move location to hospital
        # TODO: implement an efficient way to move to a hospital based on current ward
        self.model.current_location_ids[hospitalized_ids] = (
            -self.model.current_ward_ids[hospitalized_ids]
        )

    def go_home(self):
        current_time = self.real_time

        active_ids = self.get_active_ids(return_type=set)
        # Go home logic for people in their home wards
        in_home_ward_person_ids = np.where(
            self.model.return_ward_at == datetime.max
        )[0]
        in_home_ward_person_ids = list(
            active_ids.intersection(in_home_ward_person_ids))
        self.model.current_location_ids[in_home_ward_person_ids] = (
            self.model.household_ids[in_home_ward_person_ids]
        )

        # For people in other wards and are due to return home
        other_ward_person_ids = np.where(
            self.model.return_ward_at < current_time
        )[0]
        other_ward_person_ids = list(
            active_ids.intersection(other_ward_person_ids))
        self.model.current_location_ids[other_ward_person_ids] = (
            self.model.household_ids[other_ward_person_ids]
        )
        # Reset return_ward_at status
        self.model.return_ward_at[other_ward_person_ids] = datetime.max
        # Reset current_ward_ids status
        self.model.current_ward_ids[other_ward_person_ids] = (
            self.model.ward_ids[other_ward_person_ids]
        )

    def go_out(self):
        current_time = self.real_time

        active_ids = self.get_active_ids()

        if current_time.weekday() <= 4:  # Weekday 0-4 = Mon-Fri
            econ_move_prob = self.model.economic_status_weekday_movement_probability[active_ids]
        else:
            econ_move_prob = self.model.economic_status_other_day_movement_probability[active_ids]

        # Allow people that have mild symptoms to go out of the house with some probability.
        # This results to intermittent economic activity of a person (absences).
        symptom_move_prob = self.model.mild_symptom_movement_probability[active_ids]
        rand = np.random.random(size=len(active_ids))
        mover_ids = active_ids[np.less(rand, econ_move_prob * symptom_move_prob)]

        ################################## START: Persons returning from other wards ##################################
        ward_in_mover_ids = self.return_to_home_ward(mover_ids)
        ################################## END: Persons returning from other wards ##################################


        ################################## START: Persons moving to other wards  ##################################
        actual_ward_out_mover_ids = self.move_to_other_ward(mover_ids)
        ################################## END: Persons moving to other wards  ##################################


        ################################## START: Normal persons moving in their home wards  ##################################
        normal_mover_ids = np.array(
            list(set(mover_ids)
            .difference(ward_in_mover_ids)
            .difference(actual_ward_out_mover_ids)),
            dtype=int
        )
        self.model.current_location_ids[normal_mover_ids] = (
            self.model.economic_activity_location_ids[normal_mover_ids]
        )
        ################################## END: Normal persons moving in their home wards  ##################################

    def return_to_home_ward(self, mover_ids):
        current_time = self.real_time

        ward_in_mover_ids = mover_ids[self.model.return_ward_at[mover_ids] < current_time]
        self.model.current_location_ids[ward_in_mover_ids] = (
            # Since this is the start of day, place returning people to places outside their households because
            # they're determined to be moving outside as prescribed in the filters used to generate `mover_ids`.
            # TODO: Must be modified to accommodate for weekends if work/school specific locations are used in
            # `economic_activity_location_ids`
            self.model.economic_activity_location_ids[ward_in_mover_ids]
        )
        # Reset return_ward_at status
        self.model.return_ward_at[ward_in_mover_ids] = datetime.max
        # Reset current_ward_ids status
        self.model.current_ward_ids[ward_in_mover_ids] = (
            self.model.ward_ids[ward_in_mover_ids]
        )

        return ward_in_mover_ids

    def move_to_other_ward(self, mover_ids):
        current_time = self.real_time

        # Check for inter-ward movement
        ward_out_mover_ids = mover_ids[np.where(
            (self.model.ward_mover[mover_ids] == self.model.WARD_MOVER_TRUE) &
            # Consider only people that are in their home wards
            (self.model.return_ward_at[mover_ids] == datetime.max)
        )]

        ward_probs = self.model.params.DAILY_WARD_TRANSITION_PROBABILITY.loc[current_time.weekday()]
        target_ward_ids = np.less(
            np.random.random((len(ward_out_mover_ids), 1)),
            # NOTE: We can use .iloc because we explicitly defined that ward_ids are sorted increasingly.
            ward_probs.iloc[self.model.ward_ids[ward_out_mover_ids]]
        ).values.argmax(axis=1)

        if self.model.lockdown:
            #### NOTE: SCENARIO SPECIFIC: Execute check for blocking places with outbound
            # Get all target wards that are supposed to be on lockdown
            lockdown_target_ward_index = np.in1d(target_ward_ids, self.model.params.LOCKDOWN_WARDS_IDS)
            # Take the probability that a person is not allowed to move to other wards
            lockdown_not_allowed_target_ward_index = self.model.params.LOCKDOWN_ALLOWED_PROBABILITY < np.random.random(len(target_ward_ids))
            # If a person is not allowed to move and target location is on lockdown
            lockdown_ward_index = lockdown_target_ward_index & lockdown_not_allowed_target_ward_index

            # If identified to be restricted, set target ward to home ward
            target_ward_ids[lockdown_ward_index] = self.model.ward_ids[ward_out_mover_ids][lockdown_ward_index]
            #### NOTE: SCENARIO SPECIFIC: Execute check for blocking places with outbound

        if self.model.blocked:
            #### NOTE: SCENARIO SPECIFIC: Execute check for blocking places with outbound
            # Get all target wards that are supposed to be on lockdown
            blocked_target_ward_index = np.in1d(target_ward_ids, self.model.params.BLOCK_WARDS_IDS)
            # Take the probability that a person is not allowed to move to other wards
            blocked_not_allowed_target_ward_index = self.model.params.BLOCK_ALLOWED_PROBABILITY < np.random.random(len(target_ward_ids))
            # If a person is not allowed to move and target location is on lockdown
            blocked_ward_index = blocked_target_ward_index & blocked_not_allowed_target_ward_index

            # If identified to be restricted, set target ward to home ward
            target_ward_ids[blocked_ward_index] = self.model.ward_ids[ward_out_mover_ids][blocked_ward_index]
            #### NOTE: SCENARIO SPECIFIC: Execute check for blocking places with outbound


        other_ward_index = np.not_equal(target_ward_ids, self.model.ward_ids[ward_out_mover_ids])

        actual_ward_out_mover_ids = ward_out_mover_ids[other_ward_index]

        src_ward_ids = self.model.ward_ids[actual_ward_out_mover_ids]
        dst_ward_ids = target_ward_ids[other_ward_index]
        dow = current_time.weekday()

        stay_idx = [(
            dow,
            self.model.params.WARD_ID_TO_NAME.get(src),
            self.model.params.WARD_ID_TO_NAME.get(dst)
            ) for src, dst in zip(src_ward_ids, dst_ward_ids)]

        od_stay_matrix = self.model.params.WARD_WEEKDAY_OD_STAY_COUNT_MATRIX.loc[stay_idx]

        return_ward_params = self.model.params.get_gamma_shape_scale(
            od_stay_matrix['avg_duration'],
            od_stay_matrix['stddev_duration'])

        return_ward_at_times = current_time + np.array([
            timedelta(hours=h) for h in np.random.gamma(*return_ward_params)
        ])

        self.model.return_ward_at[actual_ward_out_mover_ids] = return_ward_at_times

        self.model.current_ward_ids[actual_ward_out_mover_ids] = target_ward_ids[other_ward_index]
        # We don't assign movers to households. They will be actively interacting with people in different wards.
        # This also suggests that at night, people from other wards will be interacting with each other and not with locals in the ward.
        self.model.current_location_ids[actual_ward_out_mover_ids] = target_ward_ids[other_ward_index]

        return actual_ward_out_mover_ids


class Country(Model):
    def __init__(self, params, model_log_file=None, individual_log_file=None):
        '''
        params: class or dict containing the global parameters for the model.
        start_datetime: datetime object corresponding to the start date of the simulation.
        step_timedelta: timedelta corresponding to the timestep in the simulation.
        '''
        self.individual_log_file = individual_log_file
        self.model_log_file = model_log_file
        self.params = params
        self.start_datetime = params.start_datetime
        self.step_timedelta = params.step_timedelta

        # Agent Vectors
        self.person_ids = None
        self.ward_ids = None
        self.household_ids = None

        self.age = None
        self.sex = None

        self.economic_status_ids = None
        self.economic_activity_location_ids = None

        # Set initial location as the household
        # Current location can be [school, household, outside]  # Since we don't have specific location data yet.
        self.current_location_ids = None
        self.current_ward_ids = None

        self.left_ward_at = None
        self.return_ward_at = None

        self.economic_status_weekday_movement_probability = None
        self.economic_status_other_day_movement_probability = None
        self.mild_symptom_movement_probability = None

        # Epidemic Vectors
        ## Epidemic status
        self.epidemic_state = None  # default to 0 -> susceptible, 1 -> infected, 2 -> contagious, 3 -> recovered, 4 -> dead
        self.infection_rate = None
        self.infected_symptomatic_status = None  # default to -1 -> uninfected, 0 -> asymptomatic, 1 -> symptomatic
        self.clinical_state = None  # default to 0 -> uninfected/not hospitalized, 1 -> hospitalized, 2 -> critical

        self.date_infected = None  # default to np.inf
        self.date_start_contagious = None  # default to np.inf
        self.date_start_symptomatic = None  # default to np.inf
        self.date_recovered = None  # default to np.inf

        self.infected_at_ward_ids = None
        self.infected_at_location_ids = None

        ## Clinical care
        self.date_start_hospitalized = None  # default to np.inf
        self.date_end_hospitalized = None  # default to np.inf
        self.date_start_critical = None  # default to np.inf
        self.date_end_critical = None  # default to np.inf
        ## Note, a person in icu can recover and need to get a hospital bed for recovery.

        ## Fatality status
        self.date_died = None  # default to np.inf


        # if (self.economic_status in self.model.params.WARD_MOVING_ECONOMIC_STATUS) and (self.age >= self.model.params.WARD_MOVEMENT_ALLOWED_AGE):
        self.ward_mover = None  # 0 -> not allowed to move between wards, 1 -> allowed to move between wards

        # Trackers
        self.infected_count = 0
        self.asymptomatic_count = 0
        self.symptomatic_count = 0
        self.hospitalized_count = 0
        self.critical_count = 0
        self.died_count = 0
        self.recovered_count = 0

        self.scheduler = EpidemicScheduler(self)

        self.lockdown = params.lockdown
        self.blocked = params.blocked

    def initialize_epidemic_vectors(self, size):
        self.STATE_SUSCEPTIBLE = 0
        self.STATE_INFECTED = 1
        self.STATE_CONTAGIOUS = 2
        self.STATE_RECOVERED = 3
        self.STATE_DEAD = 4

        self.DEAD_LOCATION_ID = -1

        self.CLINICAL_NOT_HOSPITALIZED = 0
        self.CLINICAL_HOSPITALIZED = 1
        self.CLINICAL_CRITICAL = 2
        self.CLINICAL_RELEASED_OR_DEAD = 3

        self.SYMPTOM_NONE = -1
        self.SYMPTOM_ASYMPTOMATIC = 0
        self.SYMPTOM_SYMPTOMATIC = 1

        ## Epidemic status
        self.epidemic_state = np.zeros(shape=size)  # default to 0 -> susceptible, 1 -> infected, 2 -> contagious, 3 -> recovered, 4 -> dead
        self.infection_rate = np.zeros(shape=size)
        self.infected_symptomatic_status = np.full(shape=size, fill_value=self.SYMPTOM_NONE)  # default to -1 -> uninfected, 0 -> asymptomatic, 1 -> symptomatic
        self.clinical_state = np.full(shape=size, fill_value=self.CLINICAL_NOT_HOSPITALIZED)  # default to 0 -> uninfected/not hospitalized, 1 -> hospitalized, 2 -> critical

        self.date_infected = np.full(shape=size, fill_value=datetime.max)  # default to np.inf
        self.date_start_contagious = np.full(shape=size, fill_value=datetime.max)  # default to np.inf
        self.date_start_symptomatic = np.full(shape=size, fill_value=datetime.max)  # default to np.inf
        self.date_recovered = np.full(shape=size, fill_value=datetime.max)  # default to np.inf

        self.infected_at_ward_ids = np.full(shape=size, fill_value=-1)
        self.infected_at_location_ids = np.full(shape=size, fill_value=-1)

        self.severe_disease_risk = np.ones(shape=size)

    def initialize_clinical_vectors(self, size):
        ## Clinical care
        self.date_start_hospitalized = np.full(shape=size, fill_value=datetime.max)  # default to np.inf
        self.date_end_hospitalized = np.full(shape=size, fill_value=datetime.max)  # default to np.inf
        self.date_start_critical = np.full(shape=size, fill_value=datetime.max)  # default to np.inf
        self.date_end_critical = np.full(shape=size, fill_value=datetime.max)  # default to np.inf
        ## Note, a person in icu can recover and need to get a hospital bed for recovery.

        ## Fatality status
        self.date_died = np.full(shape=size, fill_value=datetime.max)  # default to np.inf

    def initialize_agent_vectors(self, df):
        # Agent Vectors
        size = df.shape[0]

        # Define household id relative to ward id so that we can unify location.
        self.params.HOUSEHOLD_ID_TO_NAME = dict(enumerate(
            sorted(df['household_id'].unique()),
            max(self.params.WARD_ID_TO_NAME) + 1)
        )
        self.params.HOUSEHOLD_NAME_TO_ID = {j: i for i, j in self.params.HOUSEHOLD_ID_TO_NAME.items()}

        self.params.SEX_ID_TO_NAME = dict(enumerate(sorted(df['sex'].unique())))
        self.params.SEX_NAME_TO_ID = {j: i for i, j in self.params.SEX_ID_TO_NAME.items()}

        self.params.LOCATION_ID_TO_NAME = dict(self.params.WARD_ID_TO_NAME)
        self.params.LOCATION_ID_TO_NAME.update(self.params.HOUSEHOLD_ID_TO_NAME)
        self.params.LOCATION_NAME_TO_ID = {j: i for i, j in self.params.LOCATION_ID_TO_NAME.items()}

        self.person_ids = np.array(range(size), dtype=int)

        self.ward_ids = np.array(df['ward_id'].map(self.params.WARD_NAME_TO_ID))
        self.household_ids = np.array(df['household_id'].map(self.params.HOUSEHOLD_NAME_TO_ID))

        self.age = np.array(df['age'])
        self.sex = np.array(df['sex'].map(self.params.SEX_NAME_TO_ID))

        self.economic_status_ids = np.array(df['economic_status'].map(self.params.ECON_STAT_NAME_TO_ID))
        self.economic_activity_location_ids = np.array(df['economic_activity_location_id'].map(
            self.params.LOCATION_NAME_TO_ID))

        self.economic_status_ids_person_ids = {i: set(np.where(self.economic_status_ids == i)[0]) for i in self.params.ECON_STAT_ID_TO_NAME}

        self.economic_status_weekday_movement_probability = np.array(df['economic_status'].map(self.params.ECONOMIC_STATUS_WEEKDAY_MOVEMENT_PROBABILITY))
        self.economic_status_other_day_movement_probability = np.array(df['economic_status'].map(self.params.ECONOMIC_STATUS_OTHER_DAY_MOVEMENT_PROBABILITY))
        self.mild_symptom_movement_probability = np.ones(shape=size)

        # Set initial location as the household
        # Current location can be [school, household, outside]  # Since we don't have specific location data yet.
        self.current_location_ids = np.array(self.household_ids)
        self.current_ward_ids = np.array(self.ward_ids)

        self.left_ward_at = np.full(shape=size, fill_value=datetime.max)
        self.return_ward_at = np.full(shape=size, fill_value=datetime.max)

        ward_moving_economic_status_ids = [self.params.ECON_STAT_NAME_TO_ID[es] for es in self.params.WARD_MOVING_ECONOMIC_STATUS]

        self.WARD_MOVER_FALSE = 0
        self.WARD_MOVER_TRUE = 1
        self.ward_mover = self.WARD_MOVER_TRUE * (
            np.in1d(self.economic_status_ids, ward_moving_economic_status_ids) &
            (self.age >= self.params.WARD_MOVEMENT_ALLOWED_AGE)
        )

    def scenario_data_preprocessing(self, df):
        pass

    def scenario_data_postprocessing(self, df):
        pass

    def load_agents(self, filename, size=None, infect_num=None):
        if filename.endswith('.csv'):
            df = pd.read_csv(filename)
        elif filename.endswith('.pickle'):
            df = pd.read_pickle(filename)
        else:
            raise ValueError('Invalid file type!')

        if size is not None:
            df = df.head(size)
        else:
            size = df.shape[0]

        self.scenario_data_preprocessing(df)
        self.initialize_agent_vectors(df)
        self.scenario_data_postprocessing(df)

        self.initialize_epidemic_vectors(size)
        self.initialize_clinical_vectors(size)

        if infect_num is not None:
            candidate_ids = self.person_ids[
                np.in1d(self.current_ward_ids, self.params.SEED_INFECT_WARD_IDS) &
                ((self.age > self.params.SEED_INFECT_AGE_MIN) & (self.age < self.params.SEED_INFECT_AGE_MAX))
            ]

            neighbors_to_infect = []

            for did, pr in self.params.DISTRICT_ID_INFECTED_PROB.items():
                cands = list(set(candidate_ids).intersection(np.where(self.current_ward_ids == did)[0]))

                ic = int(pr * infect_num) + 1
                neighbors_to_infect.append(np.random.choice(cands, size=ic, replace=False))

            neighbors_to_infect = np.concatenate(neighbors_to_infect)

            self.set_epidemic_status(neighbors_to_infect)

    def step(self):
        self.scheduler.step()

    def set_epidemic_status(self, neighbors_to_infect):
        num_neighbors_to_infect = len(neighbors_to_infect)

        if num_neighbors_to_infect == 0:
            return

        current_time = self.scheduler.real_time

        self.infected_at_ward_ids[neighbors_to_infect] = self.current_ward_ids[neighbors_to_infect]
        self.infected_at_location_ids[neighbors_to_infect] = self.current_location_ids[neighbors_to_infect]

        # set_epidemic_status
        self.epidemic_state[neighbors_to_infect] = self.STATE_INFECTED
        self.date_infected[neighbors_to_infect] = current_time

        self.infected_symptomatic_status[neighbors_to_infect] = np.where(
            np.random.random(num_neighbors_to_infect) < self.params.SYMPTOMATIC_RATE,
            self.SYMPTOM_SYMPTOMATIC,
            self.SYMPTOM_ASYMPTOMATIC
        )

        symptomatic_neighbor_ids = neighbors_to_infect[
            self.infected_symptomatic_status[neighbors_to_infect] == self.SYMPTOM_SYMPTOMATIC
        ]

        if len(symptomatic_neighbor_ids) > 0:
            self.infection_rate[symptomatic_neighbor_ids] = (
                self.params.AGE_SYMPTOMATIC_INFECTION_RATE_VALUES[self.age[symptomatic_neighbor_ids]]
            )

            symptomatic_times = np.array([
                timedelta(days=d) for d in np.random.gamma(
                    self.params.INCUBATION_PERIOD_SHAPE,
                    self.params.INCUBATION_PERIOD_SCALE,
                    len(symptomatic_neighbor_ids)
                )
            ])

            self.date_start_symptomatic[symptomatic_neighbor_ids] = (
                current_time + symptomatic_times
            )

            self.date_start_contagious[symptomatic_neighbor_ids] = (
                self.date_start_symptomatic[symptomatic_neighbor_ids] - timedelta(hours=12)
            )

            # set_clinical_need_status
            hospitalization_probability = np.array(list(
                map(
                    self.params.AGE_HOSPITALIZATION_PROBABILITY.get,
                    self.age[symptomatic_neighbor_ids]
                )
            ))

            hospitalization_probability = (
                hospitalization_probability * self.severe_disease_risk[
                    self.current_ward_ids[symptomatic_neighbor_ids]
                ]
            )

            hospitalized_ids = symptomatic_neighbor_ids[np.less(
                np.random.random(len(hospitalization_probability)),
                hospitalization_probability)
            ]

            critical_care_probability = np.array(list(
                map(
                    self.params.AGE_CRITICAL_CARE_PROBABILITY.get,
                    self.age[hospitalized_ids]
                )
            ))
            critical_ids = hospitalized_ids[np.less(
                np.random.random(len(critical_care_probability)),
                critical_care_probability)
            ]

            hospitalized_ids = np.array(
                list(set(hospitalized_ids).difference(critical_ids)),
                dtype=int
            )

            critical_fatality_ids = critical_ids[
                np.random.random(len(critical_ids)) < self.params.CRITICAL_FATALITY_RATE
            ]

            hospitalization_fatality_probability = np.array(list(
                map(
                    self.params.AGE_HOSPITALIZATION_FATALITY_PROBABILITY.get,
                    self.age[hospitalized_ids]
                )
            ))

            hospitalized_fatality_ids = hospitalized_ids[np.less(
                np.random.random(len(hospitalized_ids)),
                hospitalization_fatality_probability)
            ]

            hospitalized_ids = np.array(
                list(set(hospitalized_ids).union(critical_ids)),
                dtype=int
            )

            if len(hospitalized_ids) > 0:
                # Set hospital care period
                self.date_start_hospitalized[hospitalized_ids] = (
                    self.date_start_symptomatic[hospitalized_ids] +
                    self.params.SYMPTOM_TO_HOSPITALIZATION_PERIOD
                )
                self.date_end_hospitalized[hospitalized_ids] = (
                    self.date_start_hospitalized[hospitalized_ids] +
                    self.params.HOSPITALIZATION_PERIOD
                )

            if len(critical_ids) > 0:
                # Set critical care period
                self.date_start_critical[critical_ids] = (
                    self.date_end_hospitalized[critical_ids]
                )
                self.date_end_critical[critical_ids] = (
                    self.date_start_critical[critical_ids] +
                    self.params.CRITICAL_PERIOD
                )

            # Get dead persons
            dead_ids = np.array(
                list(set(hospitalized_fatality_ids).union(critical_fatality_ids)),
                dtype=int
            )

            if len(dead_ids) > 0:
                self.date_died[dead_ids] = (
                    self.date_start_symptomatic[dead_ids] +
                    self.params.SYMPTOMATIC_TO_DEATH_PERIOD
                )

            critical_and_dead_ids = np.array(
                list(set(dead_ids).intersection(critical_ids)),
                dtype=int
            )
            critical_not_dead_ids = np.array(
                list(set(critical_ids).difference(critical_and_dead_ids)),
                dtype=int
            )
            hospitalized_not_critical_and_dead = np.array(
                list(set(hospitalized_ids).difference(critical_ids).difference(dead_ids)),
                dtype=int
            )
            symptomatic_not_hospitalized_not_dead = np.array(
                list(set(symptomatic_neighbor_ids).difference(hospitalized_ids).difference(dead_ids)),
                dtype=int
            )

            if len(critical_and_dead_ids) > 0:
                self.date_died[critical_and_dead_ids] = np.maximum(
                    self.date_died[critical_and_dead_ids],
                    self.date_end_critical[critical_and_dead_ids]
                )  # Do this in case we use a distribution for the other clinical periods.

            if len(critical_not_dead_ids) > 0:
                # Recovery for critical symptomatic persons
                self.date_recovered[critical_not_dead_ids] = (
                    self.date_end_critical[critical_not_dead_ids]
                )

            if len(hospitalized_not_critical_and_dead) > 0:
                # Recovery for hospitalized symptomatic persons
                self.date_recovered[hospitalized_not_critical_and_dead] = (
                    self.date_end_hospitalized[hospitalized_not_critical_and_dead]
                )

            if len(symptomatic_not_hospitalized_not_dead) > 0:
                # Recovery for not hospitalized and not dead persons

                symptomatic_recovery_times = np.array([
                    timedelta(days=d) for d in np.random.gamma(
                        self.params.SYMPTOMATIC_CONTAGIOUS_PERIOD_SHAPE,
                        self.params.SYMPTOMATIC_CONTAGIOUS_PERIOD_SCALE,
                        len(symptomatic_not_hospitalized_not_dead)
                    )
                ])

                self.date_recovered[symptomatic_not_hospitalized_not_dead] = (
                    self.date_start_symptomatic[symptomatic_not_hospitalized_not_dead] +
                    symptomatic_recovery_times
                )

        # Update status for asymptomatic persons
        asymptomatic_neighbor_ids = np.array(
            list(set(neighbors_to_infect).difference(symptomatic_neighbor_ids)),
            dtype=int
        )

        if len(asymptomatic_neighbor_ids) > 0:
            self.infection_rate[asymptomatic_neighbor_ids] = (
                self.params.AGE_ASYMPTOMATIC_INFECTION_RATE_VALUES[self.age[asymptomatic_neighbor_ids]]
            )

            self.date_start_contagious[asymptomatic_neighbor_ids] = (
                current_time + np.array([
                    timedelta(days=d) for d in np.random.exponential(
                        self.params.ASYMPTOMATIC_TO_CONTAGIOUS_PERIOD_MEAN,
                        size=len(asymptomatic_neighbor_ids)
                    )
                ])
            )

            asymptomatic_recovery_times = np.array([
                timedelta(days=d) for d in np.random.gamma(
                    self.params.ASYMPTOMATIC_CONTAGIOUS_PERIOD_SHAPE,
                    self.params.ASYMPTOMATIC_CONTAGIOUS_PERIOD_SCALE,
                    len(asymptomatic_neighbor_ids)
                )
            ])

            self.date_recovered[asymptomatic_neighbor_ids] = (
                self.date_start_contagious[asymptomatic_neighbor_ids] +
                asymptomatic_recovery_times
            )

    def get_location_person_ids(self, location_ids, person_ids):
        '''
            Set person_ids to None if transforming the full person_ids current_location map.
        '''
        unique_location_ids = np.unique(location_ids)

        # This line returns the indices corresponding to the sorted values of location_ids.
        # This means that contiguous elements belong to the same location. We just need to find the boundaries of separation.
        sorted_args = np.argsort(location_ids)
        sorted_location_ids = location_ids[sorted_args]

        if person_ids is not None:
            # We assume here that sorted_args corresponds to the person_ids so we must cast the sorted_args to the person_ids if person_ids is passed.
            sorted_args = person_ids[sorted_args]

        # This is how we identify the boundaries of separation.
        # We pad the sorted_location_ids with -np.inf and np.inf then take the diff.
        # All non-zero values corresponds to boundaries of separation.
        d = np.diff(np.concatenate([[-np.inf], sorted_location_ids, [np.inf]]))
        inds = np.where(d != 0)[0]

        location_person_ids = {uli: sorted_args[inds[ix]: inds[ix + 1]] for ix, uli in enumerate(unique_location_ids)}

        return location_person_ids

    def log_model_output(self):
        if self.scheduler.real_time.hour == 8:
            current_time = self.scheduler.real_time
            if self.model_log_file is not None:
                data = dict(
                    date=self.scheduler.real_time.isoformat(),

                    current_contagious_count=int((self.date_start_contagious < current_time).sum()),
                    infected_count=int((self.date_infected < current_time).sum()),

                    current_exposed_cases=int((self.epidemic_state == self.STATE_INFECTED).sum()),
                    current_contagious_cases=int((self.epidemic_state == self.STATE_CONTAGIOUS).sum()),
                    current_hospitalized_cases=int((self.clinical_state == self.CLINICAL_HOSPITALIZED).sum()),
                    current_critical_cases=int((self.clinical_state == self.CLINICAL_CRITICAL).sum()),

                    asymptomatic_count=int((self.infected_symptomatic_status == self.SYMPTOM_ASYMPTOMATIC).sum()),
                    symptomatic_count=int((self.infected_symptomatic_status == self.SYMPTOM_SYMPTOMATIC).sum()),

                    hospitalized_count=int((self.date_start_hospitalized < current_time).sum()),
                    critical_count=int((self.date_start_critical < current_time).sum()),
                    died_count=int((self.date_died < current_time).sum()),
                    recovered_count=int((self.date_recovered < current_time).sum()),
                )
                info = json.dumps(data)
                log_to_file(self.model_log_file, info, as_log=True, verbose=False)
