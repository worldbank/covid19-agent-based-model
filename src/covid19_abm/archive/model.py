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
from params import log_to_file
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

    def step(self):
        current_time = self.real_time

        self.model.log_model_output()

        recovered_persons = set()
        for person_id in self.model.not_recovered_or_dead:
            person = self.model.persons[person_id]

            if person.date_died and (current_time > person.date_died):
                if not person.is_dead:
                    person.is_dead = True
                    self.model.died_count += 1
                    person.move_dead()
                    self.model.dead_person_ids.add(person_id)
                continue

            if person.date_recovered and (current_time > person.date_recovered):
                if not person.is_recovered:
                    person.is_recovered = True
                    self.model.recovered_count += 1
                    recovered_persons.add(person_id)
                    if person.is_hospitalized:
                        person.move_hospitalized_recovered()
                continue

            if person.date_start_hospitalized:
                if person.date_start_critical and (current_time > person.date_start_critical):
                    if not person.is_critical:
                        person.is_critical = True
                        self.model.critical_count += 1
                    continue
                elif (current_time > person.date_start_hospitalized):
                    if not person.is_hospitalized:
                        person.is_hospitalized = True
                        self.model.hospitalized_count += 1
                        person.move_to_hospital()
                    continue
                else:
                    # This is for people who will later require hospitalization.
                    if (self.real_time.hour == self.model.params.WEEKDAY_START_DAY_HOUR) or (self.real_time.hour == self.model.params.WEEKDAY_END_DAY_HOUR):
                        person.move(person_status='mild_symptoms')
            else:
                if (self.real_time.hour == self.model.params.WEEKDAY_START_DAY_HOUR) or (self.real_time.hour == self.model.params.WEEKDAY_END_DAY_HOUR):
                    if person.is_symptomatic:
                        # This is for people who are sick but will not be hospitalized.
                        person.move(person_status='mild_symptoms')
                    else:
                        person.move(person_status='asymptomatic')

        self.model.not_recovered_or_dead = self.model.not_recovered_or_dead.difference(recovered_persons).difference(self.model.dead_person_ids)
        self.model.all_living_person_ids = self.model.all_living_person_ids.difference(self.model.dead_person_ids)
        update_moves = self.model.all_living_person_ids.difference(self.model.not_recovered_or_dead)

        if (self.real_time.hour == self.model.params.WEEKDAY_START_DAY_HOUR) or (self.real_time.hour == self.model.params.WEEKDAY_END_DAY_HOUR):
            for person_id in update_moves:
                person = self.model.persons[person_id]
                person.move(person_status='healthy')

        # Shuffle ids
        # contagious_person_ids = np.random.choice(list(self.model.contagious_persons), size=len(self.model.contagious_persons), replace=False)
        contagious_person_ids = list(self.model.contagious_persons)
        np.random.shuffle(contagious_person_ids)
        for contagious_person_id in contagious_person_ids:
            self.model.contagious_persons[contagious_person_id].step()

        self.steps += 1
        self.time += 1
        self.real_time += self.step_timedelta

    def add_person(self, person):
        self.model.persons[person.person_id] = person
        self.model.location_persons_map.setdefault(person.current_location_id, set()).add(person.person_id)
        self.model.all_living_person_ids.add(person.person_id)

        if person.ward_mover:
            self.model.ward_movement_persons_map.setdefault(person.current_ward_id, set()).add(person.person_id)

        self.model.add_remove_ward_economic_status_person(person, action='add')
        self.model.household_persons.setdefault(person.household_id, set()).add(person.person_id)


class Person(Agent):
    '''
    This class represents the people agents.
    '''
    def __init__(self, person_params, model):
        self.model = model

        self.person_id = person_params['person_id']  #
        self.ward_id = person_params['ward_id']  #
        self.household_id = person_params['household_id']  #
        # self.grid_id = person_params['grid_id']  # This is the grid location of the household if available
        # self.household_type = person_params['household_type']  # This is the household structure, e.g., family of 4 with 2 children, etc.

        self.age = person_params['age']  #
        self.sex = person_params['sex']  #

        self.economic_status = person_params['economic_status']  # Enum of `student`, `employed`, `unemployed`, `looking for job`, `stay-at-home`, `sick/disabled`, `retired`
        self.economic_activity_location_id = person_params['economic_activity_location_id']  # Location id of school, workplace, household, or -1 (this means can randomly go to community places)

        self.left_ward_at = None
        self.return_ward_at = None

        # Epidemic status
        self.is_infected = None
        self.is_symptomatic = None
        self.is_recovered = None

        self.date_infected = None
        self.date_start_contagious = None
        self.date_start_symptomatic = None
        self.date_recovered = None

        # Clinical care
        self.is_hospitalized = None
        self.is_critical = None
        self.date_start_hospitalized = None
        self.date_end_hospitalized = None
        self.date_start_critical = None
        self.date_end_critical = None
        # Note, a person in icu can recover and need to get a hospital bed for recovery.

        # Fatality status
        self.is_dead = None
        self.date_died = None

        self.infected_at_ward_id = None
        self.infected_at_location_id = None

        # Set initial location as the household
        # Current location can be [school, household, outside]  # Since we don't have specific location data yet.
        # Let us encode outside as the negative of ward_id to inform `location_persons_map` that people there are still in the same ward.
        self.current_location_id = self.household_id
        self.current_ward_id = self.ward_id

        self.ward_mover = False

        if (self.economic_status in self.model.params.WARD_MOVING_ECONOMIC_STATUS) and (self.age >= self.model.params.WARD_MOVEMENT_ALLOWED_AGE):
            # Only economically active individuals and individuals at least 18 years old are allowed to move between wards.
            self.ward_mover = True

        # Note that teachers can be in a school, which means that location id for workplace should be similar to school id

    def move(self, person_status):
        current_time = self.model.scheduler.real_time

        if (person_status == 'healthy') or (person_status == 'asymptomatic'):
            # Healthy, asymptomatic, and recovered persons move normally.
            if current_time.weekday() <= 4:  # Weekday 0-4 = Mon-Fri
                if current_time.hour == self.model.params.WEEKDAY_START_DAY_HOUR:
                    self.weekday_move(action="leave_home")  # Go to school or work
                elif current_time.hour == self.model.params.WEEKDAY_END_DAY_HOUR:
                    self.weekday_move(action="go_home")
            else:
                if current_time.hour == self.model.params.OTHER_DAY_START_DAY_HOUR:
                    self.other_day_move(action="leave_home")
                elif current_time.hour == self.model.params.OTHER_DAY_END_DAY_HOUR:
                    self.other_day_move(action="go_home")
        elif person_status == 'mild_symptoms':
            # Mild symptoms
            # Opportunity for policy scenario. Making sure that symptomatic (not hospitalized) must not be allowed to move!
            if current_time.weekday() <= 4:  # Weekday 0-4 = Mon-Fri
                if current_time.hour == self.model.params.WEEKDAY_START_DAY_HOUR:
                    self.weekday_mild_symptom_move(action="leave_home")
                elif current_time.hour == self.model.params.WEEKDAY_END_DAY_HOUR:
                    self.weekday_mild_symptom_move(action="go_home")
            else:
                if current_time.hour == self.model.params.OTHER_DAY_START_DAY_HOUR:
                    self.other_day_mild_symptom_move(action="leave_home")
                elif current_time.hour == self.model.params.OTHER_DAY_END_DAY_HOUR:
                    self.other_day_mild_symptom_move(action="go_home")
        else:
            raise ValueError(f'Person status `{person_status}` not valid!')

    def weekday_move(self, action):
        params = self.model.params

        if action == 'go_home':
            if self.return_ward_at is None:
                self.update_location(self.household_id)
            else:
                self.check_return_ward(location_id=self.household_id)
        elif action == 'leave_home':
            if np.random.random() < params.ECONOMIC_STATUS_WEEKDAY_MOVEMENT_PROBABILITY[self.economic_status]:
                self.move_to_ward()
                if self.return_ward_at is None:
                    self.update_location(self.economic_activity_location_id)
        else:
            raise ValueError(f"Action `{action}` not valid!")

    def other_day_move(self, action):
        params = self.model.params
        if action == 'go_home':
            if self.return_ward_at is None:
                self.update_location(self.household_id)
            else:
                self.check_return_ward(location_id=self.household_id)
        elif action == 'leave_home':
            if np.random.random() < params.ECONOMIC_STATUS_OTHER_DAY_MOVEMENT_PROBABILITY[self.economic_status]:
                self.move_to_ward()
                if self.return_ward_at is None:
                    self.update_location(self.ward_id)
        else:
            raise ValueError(f"Action `{action}` not valid!")

    def weekday_mild_symptom_move(self, action):
        params = self.model.params

        # Movement for stay-at-home or unemployed individuals.
        if action == 'go_home':
            if self.return_ward_at is None:
                self.update_location(self.household_id)
            else:
                self.check_return_ward(location_id=self.household_id)
        elif action == 'leave_home':
            if np.random.random() < params.ECONOMIC_STATUS_WEEKDAY_MOVEMENT_PROBABILITY[self.economic_status]:
                if np.random.random() < params.MILD_SYMPTOM_MOVEMENT_PROBABILITY:
                    # Allow people that have mild symptoms to go out of the house with some probability.
                    # This results to intermittent economic activity of a person (absences).
                    self.move_to_ward()
                    if self.return_ward_at is None:
                        self.update_location(self.economic_activity_location_id)
        else:
            raise ValueError(f"Action `{action}` not valid!")

    def other_day_mild_symptom_move(self, action):
        params = self.model.params
        if action == 'go_home':
            if self.return_ward_at is None:
                self.update_location(self.household_id)
            else:
                self.check_return_ward(location_id=self.household_id)
        elif action == 'leave_home':
            if np.random.random() < params.ECONOMIC_STATUS_OTHER_DAY_MOVEMENT_PROBABILITY[self.economic_status]:
                if np.random.random() < params.MILD_SYMPTOM_MOVEMENT_PROBABILITY:
                    self.move_to_ward()
                    if self.return_ward_at is None:
                        self.update_location(self.current_ward_id)
        else:
            raise ValueError(f"Action `{action}` not valid!")

    def other_ward_move(self):
        # For now, only economically active persons will be allowed to move to other wards.
        pass

    def move_to_hospital(self):
        if self.model.scheduler.real_time < self.date_recovered:
            if self.current_location_id not in self.model.hospital_ids:
                hospital_id = self.model.get_hospital(self)
                self.update_location(hospital_id)

    def move_dead(self):
        self.update_location(None)

    def move_hospitalized_recovered(self):
        if self.ward_id != self.current_ward_id:
            self.update_ward_details(self.ward_id, direction='return', location_id=self.household_id)
        else:
            self.update_location(self.household_id)

    def check_return_ward(self, location_id=None):
        if self.ward_id != self.current_ward_id:
            # The person is already in a different ward.
            if self.model.scheduler.real_time > self.return_ward_at:
                self.update_ward_details(self.ward_id, direction='return', location_id=location_id)
            return True
        return False

    def move_to_ward(self):
        if not self.ward_mover:
            # Make sure that only those who are allowed to move between wards are processed.
            return

        current_time = self.model.scheduler.real_time
        params = self.model.params

        if self.check_return_ward():
            return

        ward_probs = params.DAILY_WARD_TRANSITION_PROBABILITY[(current_time.weekday(), self.ward_id)]
        ward_move_ind = (np.random.random() < ward_probs).argmax()
        ward_id = params.WARD_ID_TO_NAME[ward_move_ind]

        if ward_id == self.ward_id:
            return

        self.update_ward_details(ward_id, direction='leave')

    def update_location(self, new_location_id):
        if new_location_id != self.current_location_id:
            self.model.location_persons_map.setdefault(self.current_location_id, set()).remove(self.person_id)
            self.current_location_id = new_location_id
            self.model.location_persons_map.setdefault(self.current_location_id, set()).add(self.person_id)

    def update_ward_details(self, ward_id, direction, location_id=None):

        self.model.ward_movement_persons_map[self.current_ward_id].remove(self.person_id)
        self.model.add_remove_ward_economic_status_person(self, action='remove')

        if direction == 'leave':
            current_time = self.model.scheduler.real_time

            stay_period = self.model.params.get_ward_movement_stay_period(
                current_time.weekday(),
                self.current_ward_id, ward_id)

            self.left_ward_at = current_time
            self.return_ward_at = current_time + stay_period
        elif direction == 'return':
            self.left_ward_at = None
            self.return_ward_at = None
        else:
            raise ValueError(f'Direction `{direction}` not valid!')

        self.current_ward_id = ward_id

        self.model.ward_movement_persons_map.setdefault(self.current_ward_id, set()).add(self.person_id)
        self.model.add_remove_ward_economic_status_person(self, action='add')

        self.update_location(location_id or ward_id)

    def step(self):
        current_time = self.model.scheduler.real_time

        if self.is_infected:
            if current_time >= (self.date_recovered or self.date_died):
                self.model.add_remove_person_from_contagious(self)

            elif current_time >= self.date_start_contagious:
                self.infect()

            # self.infect()

    def infect(self):
        current_time = self.model.scheduler.real_time
        params = self.model.params

        neighbors = self.model.get_neighbors(self)
        infection_rate = params.SYMPTOMATIC_INFECTION_RATE if self.is_symptomatic else params.ASYMPTOMATIC_INFECTION_RATE

        for n in neighbors:
            if np.random.random() < infection_rate:
                if not n.is_infected:
                    n.set_epidemic_status(current_time)
                    self.model.add_remove_person_from_contagious(n)

    def set_epidemic_status(self, current_time):
        self.model.infected_count += 1
        self.model.ward_infected_persons.setdefault(self.current_ward_id, set()).add(self)

        self.is_infected = True
        self.is_symptomatic = False
        self.is_hospitalized = False
        self.is_critical = False
        self.is_recovered = False
        self.is_dead = False

        self.infected_at_ward_id = self.current_ward_id
        self.infected_at_location_id = self.current_location_id

        self.date_infected = current_time

        self.is_symptomatic = np.random.random() < self.model.params.SYMPTOMATIC_RATE
        self.date_start_symptomatic = (self.get_incubation_period() + current_time) if self.is_symptomatic else None
        self.date_start_contagious = self.get_start_contagious_period(current_time)

        self.set_clinical_need_status(current_time)

    def get_incubation_period(self):
        incubation_period = timedelta(days=np.random.gamma(self.model.params.INCUBATION_PERIOD_SHAPE, self.model.params.INCUBATION_PERIOD_SCALE))
        return incubation_period

    def get_start_contagious_period(self, current_time):
        if self.is_symptomatic:
            # Start 12 hours prior to onset of symptom
            start_contagious_period = self.date_start_symptomatic - timedelta(hours=12)
        else:
            start_contagious_period = current_time + self.model.params.ASYMPTOMATIC_TO_CONTAGIOUS_PERIOD

        return start_contagious_period

    def set_clinical_need_status(self, current_time):
        params = self.model.params
        hospitalization_probability = params.AGE_HOSPITALIZATION_PROBABILITY[self.age]
        critical_care_probability = params.AGE_CRITICAL_CARE_PROBABILITY[self.age]
        fatality_probability = params.AGE_FATALITY_PROBABILITY[self.age]
        self.model.not_recovered_or_dead.add(self.person_id)

        if self.is_symptomatic:
            self.model.symptomatic_count += 1
            self.date_recovered = self.date_start_symptomatic + params.SYMPTOM_TO_RECOVERY_PERIOD
            rh, rc, rd = np.random.random(3)

            if rc < critical_care_probability:
                self.date_start_hospitalized = params.SYMPTOM_TO_HOSPITALIZATION_PERIOD + self.date_start_symptomatic
                self.date_end_hospitalized = self.date_start_hospitalized + params.HOSPITALIZATION_PERIOD
                self.date_start_critical = self.date_end_hospitalized
                self.date_end_critical = self.date_start_critical + params.CRITICAL_PERIOD
            elif rh < hospitalization_probability:
                self.date_start_hospitalized = params.SYMPTOM_TO_HOSPITALIZATION_PERIOD + self.date_start_symptomatic
                self.date_end_hospitalized = self.date_start_hospitalized + params.HOSPITALIZATION_PERIOD

            # Some people may die even if they haven't been hospitalized. Is this a good assumption?
            if rd < fatality_probability:
                # self.is_dead = True
                self.date_died = self.date_start_symptomatic + params.SYMPTOMATIC_TO_DEATH_PERIOD

                if self.date_end_critical is not None:
                    self.date_died = max(self.date_died, self.date_end_critical)  # Do this in case we use a distribution for the other clinical periods.
            else:
                # self.is_recovered = True
                self.date_recovered = self.date_end_critical or self.date_end_hospitalized or self.date_recovered
            # Define recovery or fatality date for symptomatic here
        else:
            self.model.asymptomatic_count += 1
            # Define recovery for asymptomatic here
            # self.is_recovered = True
            # NOTICE!!! CHECK IF THE REFERENCE DATA IS CORRECT (date_start_contagious or date_infected?)
            self.date_recovered = self.date_start_contagious + params.ASYMPTOMATIC_TO_RECOVERY_PERIOD

        self.report_infection_and_clinical_status()

    def report_infection_and_clinical_status(self):
        if self.model.individual_log_file is not None:
            d = dict(self.__dict__)
            d.pop('model')

            info = (
                json.dumps(pd.Series(d).astype('str').to_dict())
                .replace('"True"', 'true')
                .replace('"False"', 'false')
                .replace('"None"', 'null')
            )

            log_to_file(self.model.individual_log_file, info, as_log=False, verbose=False)


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
        self.exposed_persons = {}
        self.contagious_persons = {}
        self.ward_movement_persons_map = {}
        self.school_ids = set()
        self.location_persons_map = {}  # location_id -> set([agent_ids])
        self.persons = {}
        self.hospital_ids = set()
        self.ward_economic_status_persons = {}
        self.household_persons = {}
        self.ward_infected_persons = {}
        self.not_recovered_or_dead = set()
        self.dead_person_ids = set()
        self.all_living_person_ids = set()

        # Trackers
        self.infected_count = 0
        self.asymptomatic_count = 0
        self.symptomatic_count = 0
        self.hospitalized_count = 0
        self.critical_count = 0
        self.died_count = 0
        self.recovered_count = 0

        for ward_id in self.params.WARD_IDS:
            for econ_stat in self.params.ECON_STAT_NAME_TO_ID:
                if ward_id not in self.ward_economic_status_persons:
                    self.ward_economic_status_persons[ward_id] = {}

                self.ward_economic_status_persons[ward_id][econ_stat] = set()

        self.scheduler = EpidemicScheduler(self)

    def load_agents(self, filename, size=None, infect_num=None):
        if filename.endswith('.csv'):
            df = pd.read_csv(filename)
        elif filename.endswith('.pickle'):
            df = pd.read_pickle(filename)
        else:
            raise ValueError('Invalid file type!')

        if size is not None:
            df = df.head(size)

        for _, row in df.iterrows():
            person = Person(person_params=row, model=self)
            self.scheduler.add_person(person)

        if infect_num is not None:
            person_ids = list(self.persons)
            np.random.shuffle(person_ids)
            person_ids = person_ids[:infect_num]
            # for n in np.random.choice(list(self.persons), size=infect_num, replace=False):

            for n in person_ids:
                n = self.persons[n]
                n.set_epidemic_status(self.scheduler.real_time)
                self.add_remove_person_from_contagious(n)

    def step(self):
        self.scheduler.step()

    def get_neighbors(self, person):
        '''
        Things to consider.
        People living in the same ward.

        People travelling to a different ward than their household.
        '''

        if person.current_location_id and (person.current_location_id.startswith('w_') or person.current_location_id.startswith('s_')):
            # The person is outside the household.
            # Use economic status-based interaction to get the individuals that are likely the person will interact with.
            num_interactions = self.params.ECONOMIC_STATUS_INTERACTION_SIZE_MAP[person.economic_status]
            econ_stats_prob = np.random.random(size=num_interactions)

            econ_stats_id = self.params.ECON_STAT_NAME_TO_ID[person.economic_status]
            int_mat = self.params.ECONOMIC_STATUS_INTERACTION_MATRIX_CUMSUM_VALUES[econ_stats_id, :]

            interaction_economic_status = [self.params.ECON_STAT_ID_TO_NAME[(p < int_mat).argmax()] for p in econ_stats_prob]
            interaction_economic_status = Counter(interaction_economic_status)

            neighbors = []

            for es, count in interaction_economic_status.items():
                candidates = list(self.ward_economic_status_persons[person.current_ward_id][es])
                size = min(count, len(candidates))
                np.random.shuffle(candidates)

                vc = 0
                for n in candidates:
                    if person.person_id == n:
                        continue

                    n = self.persons[n]
                    if n.current_location_id == person.current_location_id:
                        neighbors.append(n)
                        vc += 1

                        if vc == size:
                            break
        else:
            # Cast to set so that we don't pass the reference and deplete the household!
            neighbors = set(self.household_persons[person.household_id])

            if person.person_id in neighbors:
                neighbors.remove(person.person_id)

            neighbors = [self.persons[n] for n in neighbors if self.persons[n].current_location_id == person.current_location_id]

        return neighbors

    def add_remove_person_from_contagious(self, person):
        if person.person_id in self.contagious_persons:
            self.contagious_persons.pop(person.person_id)
        else:
            # Make sure to only add people to contagious list before they recover or die.
            if (person.date_recovered or person.date_died) > self.scheduler.real_time:
                self.contagious_persons[person.person_id] = person

    def add_remove_ward_economic_status_person(self, person, action):
        # if person.current_ward_id not in self.ward_economic_status_persons:
        #     self.ward_economic_status_persons[person.current_ward_id] = {}

        # if person.economic_status not in self.ward_economic_status_persons[person.current_ward_id]:
        #     self.ward_economic_status_persons[person.current_ward_id][person.economic_status] = set()

        if action == 'add':
            # if person.person_id not in self.ward_economic_status_persons[person.current_ward_id][person.economic_status]:
            # No need to check since this is a `set` anyway.
            self.ward_economic_status_persons[person.current_ward_id][person.economic_status].add(person.person_id)
        elif action == 'remove':
            # if person.person_id in self.ward_economic_status_persons[person.current_ward_id][person.economic_status]:
            try:
                self.ward_economic_status_persons[person.current_ward_id][person.economic_status].remove(person.person_id)
            except:
                # The person is not in the ward, but we save an `if` check.
                pass
        else:
            raise ValueError(f'Action `{action}` not valid!')

    def get_hospital(self, person):
        # Get hospital in ward.
        return np.random.choice(self.params.WARD_HOSPITALS[person.current_ward_id])

    def log_model_output(self):
        if self.scheduler.real_time.hour == 8:
            if self.model_log_file is not None:
                # Trackers
                data = dict(
                    date=self.scheduler.real_time.isoformat(),
                    current_contagious_count=len(self.contagious_persons),
                    infected_count=self.infected_count,
                    asymptomatic_count=self.asymptomatic_count,
                    symptomatic_count=self.symptomatic_count,
                    hospitalized_count=self.hospitalized_count,
                    critical_count=self.critical_count,
                    died_count=self.died_count,
                    recovered_count=self.recovered_count,
                )
                info = json.dumps(data)
                log_to_file(self.model_log_file, info, as_log=True, verbose=False)
