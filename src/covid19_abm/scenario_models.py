import os
import sys
from datetime import datetime, timedelta
import numpy as np

from covid19_abm.base_model import Country


class UnmitigatedScenario(Country):
    def __init__(self, params, model_log_file=None, individual_log_file=None):
        super().__init__(
            params, model_log_file=model_log_file,
            individual_log_file=individual_log_file)

    def scenario_data_preprocessing(self, df):
        pass

    def scenario_data_postprocessing(self, df):
        assert(self.params.SCENARIO == 'UNMITIGATED')


class HandWashingRiskScenario(Country):
    def __init__(self, params, model_log_file=None, individual_log_file=None):
        super().__init__(
            params, model_log_file=model_log_file,
            individual_log_file=individual_log_file)

        self.params.scenario_handwashing_risk()

    def scenario_data_preprocessing(self, df):
        pass

    def scenario_data_postprocessing(self, df):
        assert(self.params.SCENARIO == 'HANDWASHING_RISK')
        self.hw_risk = np.array(list(map(
            self.params.DISTRICT_HW_RISK.get,
            self.params.WARD_IDS  # NOTE: there is an assumed order here that WARD_IDS are properly sorted.
        )))

        self.severe_disease_risk = np.array(list(map(
            self.params.DISTRICT_SEVERE_DISEASE_RISK.get,
            self.params.WARD_IDS  # NOTE: there is an assumed order here that WARD_IDS are properly sorted.
        )))


class HandWashingRiskImproved1Scenario(Country):
    def __init__(self, params, model_log_file=None, individual_log_file=None):
        super().__init__(
            params, model_log_file=model_log_file,
            individual_log_file=individual_log_file)

        self.params.scenario_improved_handwashing_risk_1()

    def scenario_data_preprocessing(self, df):
        pass

    def scenario_data_postprocessing(self, df):
        assert(self.params.SCENARIO == 'HANDWASHING_RISK_1')
        self.hw_risk = np.array(list(map(
            self.params.DISTRICT_HW_RISK.get,
            self.params.WARD_IDS  # NOTE: there is an assumed order here that WARD_IDS are properly sorted.
        )))

        self.severe_disease_risk = np.array(list(map(
            self.params.DISTRICT_SEVERE_DISEASE_RISK.get,
            self.params.WARD_IDS  # NOTE: there is an assumed order here that WARD_IDS are properly sorted.
        )))


class HandWashingRiskImproved2Scenario(Country):
    def __init__(self, params, model_log_file=None, individual_log_file=None):
        super().__init__(
            params, model_log_file=model_log_file,
            individual_log_file=individual_log_file)

        self.params.scenario_improved_handwashing_risk_2()

    def scenario_data_preprocessing(self, df):
        pass

    def scenario_data_postprocessing(self, df):
        assert(self.params.SCENARIO == 'HANDWASHING_RISK_2')
        self.hw_risk = np.array(list(map(
            self.params.DISTRICT_HW_RISK.get,
            self.params.WARD_IDS  # NOTE: there is an assumed order here that WARD_IDS are properly sorted.
        )))

        self.severe_disease_risk = np.array(list(map(
            self.params.DISTRICT_SEVERE_DISEASE_RISK.get,
            self.params.WARD_IDS  # NOTE: there is an assumed order here that WARD_IDS are properly sorted.
        )))


class InteractionSensitivityScenario(Country):
    def __init__(self, params, model_log_file=None, individual_log_file=None):
        super().__init__(
            params, model_log_file=model_log_file,
            individual_log_file=individual_log_file)

        self.params.scenario_test_interaction_matrix_sensitivity()

    def scenario_data_preprocessing(self, df):
        pass

    def scenario_data_postprocessing(self, df):
        assert(self.params.SCENARIO == 'INTERACTION_MATRIX_SENSITIVITY')


class IsolateSymptomaticScenario(Country):
    def __init__(self, params, model_log_file=None, individual_log_file=None):
        super().__init__(
            params, model_log_file=model_log_file,
            individual_log_file=individual_log_file)

        self.params.scenario_isolate_symptomatic_population()

    def scenario_data_preprocessing(self, df):
        pass

    def scenario_data_postprocessing(self, df):
        #### NOTE: SCENARIO SPECIFIC: Execute check for symptomatic isolation
        assert(self.params.SCENARIO == 'ISOLATE_SYMPTOMATIC')
        assert(self.params.MILD_SYMPTOM_MOVEMENT_PROBABILITY < 1)


class IsolateVulnerableScenario(Country):
    def __init__(self, params, model_log_file=None, individual_log_file=None):
        super().__init__(
            params, model_log_file=model_log_file,
            individual_log_file=individual_log_file)

        self.params.scenario_isolate_vulnerable_groups_in_house()

    def scenario_data_preprocessing(self, df):
        pass

    def scenario_data_postprocessing(self, df):
        ward_moving_economic_status_ids = [self.params.ECON_STAT_NAME_TO_ID[es] for es in self.params.WARD_MOVING_ECONOMIC_STATUS]

        #### NOTE: SCENARIO SPECIFIC: Execute scenario for isolating vulnerable individuals
        assert(self.params.SCENARIO == 'ISOLATE_VULNERABLE_HOUSE')
        self.ward_mover = self.WARD_MOVER_TRUE * (
            np.in1d(self.economic_status_ids, ward_moving_economic_status_ids) &
            (self.age >= self.params.WARD_MOVEMENT_ALLOWED_AGE) &
            (self.age < self.params.VULNERABLE_AGE)
        )
        self.economic_activity_location_ids[self.age >= self.params.VULNERABLE_AGE] = (
            self.current_location_ids[self.age >= self.params.VULNERABLE_AGE]
        )


class BlockGreatestMobilityScenario(Country):
    def __init__(self, params, model_log_file=None, individual_log_file=None):
        super().__init__(
            params, model_log_file=model_log_file,
            individual_log_file=individual_log_file)

        self.params.scenario_block_new_district_greatest_movement()

    def scenario_data_preprocessing(self, df):
        pass

    def scenario_data_postprocessing(self, df):
        #### NOTE: SCENARIO SPECIFIC: Execute check for blocking places with outbound
        assert(self.params.SCENARIO == 'BLOCK_GREATEST_NEW_DIST')
        blocked_ids = self.person_ids[np.in1d(self.ward_ids, self.params.BLOCK_WARDS_IDS)]
        ward_blocked_movers = blocked_ids[self.ward_mover[blocked_ids] == self.WARD_MOVER_TRUE]
        ward_blocked_allowed_movers = ward_blocked_movers[np.random.random(len(ward_blocked_movers)) < self.params.BLOCK_ALLOWED_PROBABILITY]

        self.ward_mover[ward_blocked_movers] = self.WARD_MOVER_FALSE
        self.ward_mover[ward_blocked_allowed_movers] = self.WARD_MOVER_TRUE


class LockdownGreatestMobilityScenario(Country):
    def __init__(self, params, model_log_file=None, individual_log_file=None):
        super().__init__(
            params, model_log_file=model_log_file,
            individual_log_file=individual_log_file)

        self.params.scenario_lockdown_new_district_greatest_movement()

    def scenario_data_preprocessing(self, df):
        pass

    def scenario_data_postprocessing(self, df):
        #### NOTE: SCENARIO SPECIFIC: Execute check for blocking places with outbound
        assert(self.params.SCENARIO == 'LOCKDOWN_GREATEST_NEW_DIST')
        lockdown_ids = self.person_ids[np.in1d(self.ward_ids, self.params.LOCKDOWN_WARDS_IDS)]
        ward_lockdown_movers = lockdown_ids[self.ward_mover[lockdown_ids] == self.WARD_MOVER_TRUE]
        ward_lockdown_allowed_movers = ward_lockdown_movers[np.random.random(len(ward_lockdown_movers)) < self.params.LOCKDOWN_ALLOWED_PROBABILITY]

        self.ward_mover[ward_lockdown_movers] = self.WARD_MOVER_FALSE
        self.ward_mover[ward_lockdown_allowed_movers] = self.WARD_MOVER_TRUE

        decreased_mobility_rate = np.array([self.params.LOCKDOWN_DECREASED_MOBILITY_RATE[i] for i in self.ward_ids[lockdown_ids]])

        self.economic_status_weekday_movement_probability[lockdown_ids] *= decreased_mobility_rate
        self.economic_status_other_day_movement_probability[lockdown_ids] *= decreased_mobility_rate


class ContinuedLockdownScenario(Country):
    def __init__(self, params, model_log_file=None, individual_log_file=None):
        super().__init__(
            params, model_log_file=model_log_file,
            individual_log_file=individual_log_file)

        self.params.scenario_continued_all_lockdown()

    def scenario_data_preprocessing(self, df):
        pass

    def scenario_data_postprocessing(self, df):
        #### NOTE: SCENARIO SPECIFIC: Execute check for blocking places with outbound
        assert(self.params.SCENARIO == 'CONTINUED_ALL_LOCKDOWN')
        lockdown_ids = self.person_ids[np.in1d(self.ward_ids, self.params.LOCKDOWN_WARDS_IDS)]
        ward_lockdown_movers = lockdown_ids[self.ward_mover[lockdown_ids] == self.WARD_MOVER_TRUE]
        ward_lockdown_allowed_movers = ward_lockdown_movers[np.random.random(len(ward_lockdown_movers)) < self.params.LOCKDOWN_ALLOWED_PROBABILITY]

        self.ward_mover[ward_lockdown_movers] = self.WARD_MOVER_FALSE
        self.ward_mover[ward_lockdown_allowed_movers] = self.WARD_MOVER_TRUE

        decreased_mobility_rate = np.array([self.params.LOCKDOWN_DECREASED_MOBILITY_RATE[i] for i in self.ward_ids[lockdown_ids]])

        self.economic_status_weekday_movement_probability[lockdown_ids] *= decreased_mobility_rate
        self.economic_status_other_day_movement_probability[lockdown_ids] *= decreased_mobility_rate


class EasedLockdownScenario(Country):
    def __init__(self, params, model_log_file=None, individual_log_file=None):
        super().__init__(
            params, model_log_file=model_log_file,
            individual_log_file=individual_log_file)

        self.params.scenario_eased_all_lockdown()

    def scenario_data_preprocessing(self, df):
        pass

    def scenario_data_postprocessing(self, df):
        #### NOTE: SCENARIO SPECIFIC: Execute check for blocking places with outbound
        assert(self.params.SCENARIO == 'EASED_ALL_LOCKDOWN')
        lockdown_ids = self.person_ids[np.in1d(self.ward_ids, self.params.LOCKDOWN_WARDS_IDS)]
        ward_lockdown_movers = lockdown_ids[self.ward_mover[lockdown_ids] == self.WARD_MOVER_TRUE]
        ward_lockdown_allowed_movers = ward_lockdown_movers[np.random.random(len(ward_lockdown_movers)) < self.params.LOCKDOWN_ALLOWED_PROBABILITY]

        self.ward_mover[ward_lockdown_movers] = self.WARD_MOVER_FALSE
        self.ward_mover[ward_lockdown_allowed_movers] = self.WARD_MOVER_TRUE

        decreased_mobility_rate = np.array([self.params.LOCKDOWN_DECREASED_MOBILITY_RATE[i] for i in self.ward_ids[lockdown_ids]])

        self.economic_status_weekday_movement_probability[lockdown_ids] *= decreased_mobility_rate
        self.economic_status_other_day_movement_probability[lockdown_ids] *= decreased_mobility_rate


class OpenMiningScenario(Country):
    def __init__(self, params, model_log_file=None, individual_log_file=None):
        super().__init__(
            params, model_log_file=model_log_file,
            individual_log_file=individual_log_file)

        self.params.scenario_continued_all_lockdown_open_mining()

    def scenario_data_preprocessing(self, df):
        #### NOTE: SCENARIO SPECIFIC: Continued lockdown with mining open
        df.loc[df['mining_district_id'] != '', 'economic_activity_location_id'] = df.loc[df['mining_district_id'] != '', 'mining_district_id']
        df.loc[df['mining_district_id'] != '', 'household_id'] = df.loc[df['mining_district_id'] != '', 'mining_district_id']
        #### NOTE: SCENARIO SPECIFIC: Continued lockdown with mining open

    def scenario_data_postprocessing(self, df):
        #### NOTE: SCENARIO SPECIFIC: Continued lockdown with mining open
        assert(self.params.SCENARIO == 'CONTINUED_ALL_LOCKDOWN_MINING')
        mining_ids = df[~df['household_id'].str.startswith('h_')]['person_id'].values

        lockdown_ids = self.person_ids[np.in1d(self.ward_ids, self.params.LOCKDOWN_WARDS_IDS)]
        lockdown_ids = np.array(list(set(lockdown_ids).difference(mining_ids)))

        ward_lockdown_movers = lockdown_ids[self.ward_mover[lockdown_ids] == self.WARD_MOVER_TRUE]
        ward_lockdown_allowed_movers = ward_lockdown_movers[np.random.random(len(ward_lockdown_movers)) < self.params.LOCKDOWN_ALLOWED_PROBABILITY]

        self.ward_mover[ward_lockdown_movers] = self.WARD_MOVER_FALSE
        self.ward_mover[ward_lockdown_allowed_movers] = self.WARD_MOVER_TRUE

        decreased_mobility_rate = np.array([self.params.LOCKDOWN_DECREASED_MOBILITY_RATE[i] for i in self.ward_ids[lockdown_ids]])

        self.economic_status_weekday_movement_probability[lockdown_ids] *= decreased_mobility_rate
        self.economic_status_other_day_movement_probability[lockdown_ids] *= decreased_mobility_rate

        # Don't allow miners to move between wards
        self.ward_mover[mining_ids] = self.WARD_MOVER_FALSE


class OpenSchoolsScenario(Country):
    def __init__(self, params, model_log_file=None, individual_log_file=None):
        super().__init__(
            params, model_log_file=model_log_file,
            individual_log_file=individual_log_file)

        self.params.scenario_continued_all_lockdown_open_schools()

    def scenario_data_preprocessing(self, df):
        pass

    def scenario_data_postprocessing(self, df):
        #### NOTE: SCENARIO SPECIFIC: Continued lockdown with mining open
        assert(self.params.SCENARIO == 'CONTINUED_ALL_LOCKDOWN_SCHOOLS')
        if 'school_id_district' in df:
            school_educ_ids = df[df['school_id_district'] != '']['person_id'].values
        elif 'school_goers' in df:
            school_educ_ids = df[df['school_goers'] == 1]['person_id'].values
        else:
            raise ValueError('Column `school_id_district` or `school_goers` not found!')

        lockdown_ids = self.person_ids[np.in1d(self.ward_ids, self.params.LOCKDOWN_WARDS_IDS)]
        lockdown_ids = np.array(list(set(lockdown_ids).difference(school_educ_ids)))

        ward_lockdown_movers = lockdown_ids[self.ward_mover[lockdown_ids] == self.WARD_MOVER_TRUE]
        ward_lockdown_allowed_movers = ward_lockdown_movers[np.random.random(len(ward_lockdown_movers)) < self.params.LOCKDOWN_ALLOWED_PROBABILITY]

        self.ward_mover[ward_lockdown_movers] = self.WARD_MOVER_FALSE
        self.ward_mover[ward_lockdown_allowed_movers] = self.WARD_MOVER_TRUE

        decreased_mobility_rate = np.array([self.params.LOCKDOWN_DECREASED_MOBILITY_RATE[i] for i in self.ward_ids[lockdown_ids]])

        self.economic_status_weekday_movement_probability[lockdown_ids] *= decreased_mobility_rate
        self.economic_status_other_day_movement_probability[lockdown_ids] *= decreased_mobility_rate

        # Don't allow students/teachers to move between wards
        self.ward_mover[school_educ_ids] = self.WARD_MOVER_FALSE
        # During weekends apply mobility restrictions to students/teachers?
        other_decreased_mobility_rate = np.array([self.params.LOCKDOWN_DECREASED_MOBILITY_RATE[i] for i in self.ward_ids[school_educ_ids]])
        self.economic_status_other_day_movement_probability[school_educ_ids] *= other_decreased_mobility_rate


class EasedOpenSchoolsScenario(Country):
    def __init__(self, params, model_log_file=None, individual_log_file=None):
        super().__init__(
            params, model_log_file=model_log_file,
            individual_log_file=individual_log_file)

        self.params.scenario_eased_all_lockdown_open_schools()

    def scenario_data_preprocessing(self, df):
        pass

    def scenario_data_postprocessing(self, df):
        #### NOTE: SCENARIO SPECIFIC: Continued lockdown with mining open
        assert(self.params.SCENARIO == 'EASED_ALL_LOCKDOWN_SCHOOLS')
        if 'school_id_district' in df:
            school_educ_ids = df[df['school_id_district'] != '']['person_id'].values
        elif 'school_goers' in df:
            school_educ_ids = df[df['school_goers'] == 1]['person_id'].values
        else:
            raise ValueError('Column `school_id_district` or `school_goers` not found!')

        lockdown_ids = self.person_ids[np.in1d(self.ward_ids, self.params.LOCKDOWN_WARDS_IDS)]
        lockdown_ids = np.array(list(set(lockdown_ids).difference(school_educ_ids)))

        ward_lockdown_movers = lockdown_ids[self.ward_mover[lockdown_ids] == self.WARD_MOVER_TRUE]
        ward_lockdown_allowed_movers = ward_lockdown_movers[np.random.random(len(ward_lockdown_movers)) < self.params.LOCKDOWN_ALLOWED_PROBABILITY]

        self.ward_mover[ward_lockdown_movers] = self.WARD_MOVER_FALSE
        self.ward_mover[ward_lockdown_allowed_movers] = self.WARD_MOVER_TRUE

        decreased_mobility_rate = np.array([self.params.LOCKDOWN_DECREASED_MOBILITY_RATE[i] for i in self.ward_ids[lockdown_ids]])

        self.economic_status_weekday_movement_probability[lockdown_ids] *= decreased_mobility_rate
        self.economic_status_other_day_movement_probability[lockdown_ids] *= decreased_mobility_rate

        # Don't allow students/teachers to move between wards
        self.ward_mover[school_educ_ids] = self.WARD_MOVER_FALSE
        # During weekends apply mobility restrictions to students/teachers?
        other_decreased_mobility_rate = np.array([self.params.LOCKDOWN_DECREASED_MOBILITY_RATE[i] for i in self.ward_ids[school_educ_ids]])
        self.economic_status_other_day_movement_probability[school_educ_ids] *= other_decreased_mobility_rate


class OpenSchoolsSeedKidsScenario(Country):
    def __init__(self, params, model_log_file=None, individual_log_file=None):
        super().__init__(
            params, model_log_file=model_log_file,
            individual_log_file=individual_log_file)

        self.params.scenario_continued_all_lockdown_open_schools_seed_kids()

    def scenario_data_preprocessing(self, df):
        pass

    def scenario_data_postprocessing(self, df):
        #### NOTE: SCENARIO SPECIFIC: Continued lockdown with mining open
        assert(self.params.SCENARIO == 'CONTINUED_ALL_LOCKDOWN_SCHOOLS_SEED_KIDS')
        school_educ_ids = df[df['school_id_district'] != '']['person_id'].values

        lockdown_ids = self.person_ids[np.in1d(self.ward_ids, self.params.LOCKDOWN_WARDS_IDS)]
        lockdown_ids = np.array(list(set(lockdown_ids).difference(school_educ_ids)))

        ward_lockdown_movers = lockdown_ids[self.ward_mover[lockdown_ids] == self.WARD_MOVER_TRUE]
        ward_lockdown_allowed_movers = ward_lockdown_movers[np.random.random(len(ward_lockdown_movers)) < self.params.LOCKDOWN_ALLOWED_PROBABILITY]

        self.ward_mover[ward_lockdown_movers] = self.WARD_MOVER_FALSE
        self.ward_mover[ward_lockdown_allowed_movers] = self.WARD_MOVER_TRUE

        decreased_mobility_rate = np.array([self.params.LOCKDOWN_DECREASED_MOBILITY_RATE[i] for i in self.ward_ids[lockdown_ids]])

        self.economic_status_weekday_movement_probability[lockdown_ids] *= decreased_mobility_rate
        self.economic_status_other_day_movement_probability[lockdown_ids] *= decreased_mobility_rate

        # Don't allow students/teachers to move between wards
        self.ward_mover[school_educ_ids] = self.WARD_MOVER_FALSE
        # During weekends apply mobility restrictions to students/teachers?
        other_decreased_mobility_rate = np.array([self.params.LOCKDOWN_DECREASED_MOBILITY_RATE[i] for i in self.ward_ids[school_educ_ids]])
        self.economic_status_other_day_movement_probability[school_educ_ids] *= other_decreased_mobility_rate


class OpenManufacturingScenario(Country):
    def __init__(self, params, model_log_file=None, individual_log_file=None):
        super().__init__(
            params, model_log_file=model_log_file,
            individual_log_file=individual_log_file)

        self.params.scenario_continued_all_lockdown_open_manufacturing()

    def scenario_data_preprocessing(self, df):
        pass

    def scenario_data_postprocessing(self, df):
        assert(self.params.SCENARIO == 'CONTINUED_ALL_LOCKDOWN_MANUFACTURING')

        manufacturing_ids = df[df['manufacturing_workers'].notnull()]['person_id'].values

        lockdown_ids = self.person_ids[np.in1d(self.ward_ids, self.params.LOCKDOWN_WARDS_IDS)]
        lockdown_ids = np.array(list(set(lockdown_ids).difference(manufacturing_ids)))

        ward_lockdown_movers = lockdown_ids[self.ward_mover[lockdown_ids] == self.WARD_MOVER_TRUE]
        ward_lockdown_allowed_movers = ward_lockdown_movers[np.random.random(len(ward_lockdown_movers)) < self.params.LOCKDOWN_ALLOWED_PROBABILITY]

        self.ward_mover[ward_lockdown_movers] = self.WARD_MOVER_FALSE
        self.ward_mover[ward_lockdown_allowed_movers] = self.WARD_MOVER_TRUE

        decreased_mobility_rate = np.array([self.params.LOCKDOWN_DECREASED_MOBILITY_RATE[i] for i in self.ward_ids[lockdown_ids]])

        self.economic_status_weekday_movement_probability[lockdown_ids] *= decreased_mobility_rate
        self.economic_status_other_day_movement_probability[lockdown_ids] *= decreased_mobility_rate

        # Don't allow manufacturers to move between wards
        self.ward_mover[manufacturing_ids] = self.WARD_MOVER_FALSE
        # During weekends apply mobility restrictions to manufacturing workers?
        other_decreased_mobility_rate = np.array([self.params.LOCKDOWN_DECREASED_MOBILITY_RATE[i] for i in self.ward_ids[manufacturing_ids]])
        self.economic_status_other_day_movement_probability[manufacturing_ids] *= other_decreased_mobility_rate


class OpenManufacturingAndSchoolsScenario(Country):
    def __init__(self, params, model_log_file=None, individual_log_file=None):
        super().__init__(
            params, model_log_file=model_log_file,
            individual_log_file=individual_log_file)

        self.params.scenario_continued_all_lockdown_open_manufacturing_schools()

    def scenario_data_preprocessing(self, df):
        pass

    def scenario_data_postprocessing(self, df):
        assert(self.params.SCENARIO == 'CONTINUED_ALL_LOCKDOWN_MANUFACTURING_SCHOOLS')

        manufacturing_ids = df[df['manufacturing_workers'].notnull()]['person_id'].values
        school_educ_ids = df[df['school_id_district'] != '']['person_id'].values

        manufacturing_and_school_ids = np.array(list(set(manufacturing_ids).union(school_educ_ids)))

        lockdown_ids = self.person_ids[np.in1d(self.ward_ids, self.params.LOCKDOWN_WARDS_IDS)]
        lockdown_ids = np.array(list(set(lockdown_ids).difference(manufacturing_and_school_ids)))

        ward_lockdown_movers = lockdown_ids[self.ward_mover[lockdown_ids] == self.WARD_MOVER_TRUE]
        ward_lockdown_allowed_movers = ward_lockdown_movers[np.random.random(len(ward_lockdown_movers)) < self.params.LOCKDOWN_ALLOWED_PROBABILITY]

        self.ward_mover[ward_lockdown_movers] = self.WARD_MOVER_FALSE
        self.ward_mover[ward_lockdown_allowed_movers] = self.WARD_MOVER_TRUE

        decreased_mobility_rate = np.array([self.params.LOCKDOWN_DECREASED_MOBILITY_RATE[i] for i in self.ward_ids[lockdown_ids]])

        self.economic_status_weekday_movement_probability[lockdown_ids] *= decreased_mobility_rate
        self.economic_status_other_day_movement_probability[lockdown_ids] *= decreased_mobility_rate

        # Don't allow manufacturers and schools to move between wards
        self.ward_mover[manufacturing_and_school_ids] = self.WARD_MOVER_FALSE
        # During weekends apply mobility restrictions to manufacturing workers and schools?
        other_decreased_mobility_rate = np.array([self.params.LOCKDOWN_DECREASED_MOBILITY_RATE[i] for i in self.ward_ids[manufacturing_and_school_ids]])
        self.economic_status_other_day_movement_probability[manufacturing_and_school_ids] *= other_decreased_mobility_rate


def run_scenario(scenarioClass, scenario_name, sim_fname, R0, sample_size, seed_num, start_date=None, timestep=timedelta(hours=4), scaled_mobility=False):
    import pickle
    import sys
    from covid19_abm.params import ParamsConfig
    from covid19_abm.dir_manager import get_data_dir

    # scenario_name = sys.argv[0]
    # sim_fname = sys.argv[1].zfill(2)
    # R0 = 1.9
    # sample_size = 10
    # seed_num = 90  # 6 -> 66 for 10% data sample

    sim_fname = sim_fname.zfill(2)

    if scaled_mobility:
        sim_fname = f'scaled_{scenario_name}_{sim_fname}_R{R0}_samp{sample_size}_seed{seed_num}'
        stay_duration_file = 'weekday_mobility_duration_count_df-new-district-scaled.pickle'
        transition_probability_file = 'daily_region_transition_probability-new-district-scaled.csv'
    else:
        sim_fname = f'{scenario_name}_{sim_fname}_R{R0}_samp{sample_size}_seed{seed_num}'
        stay_duration_file = 'weekday_mobility_duration_count_df-new-district.pickle'
        transition_probability_file = 'daily_region_transition_probability-new-district-pre-lockdown.csv'

    now = datetime.now().isoformat()

    model_log_file = get_data_dir('logs',  f'model_log_file_{sim_fname}.{now}.log')
    individual_log_file = get_data_dir('logs', f'individual_log_file_{sim_fname}.{now}.log')

    params = ParamsConfig(
        district='new', data_sample_size=sample_size, R0=R0,
        interaction_matrix_file=get_data_dir('raw', 'final_close_interaction_matrix.xlsx'),
        stay_duration_file=get_data_dir('preprocessed', 'mobility', stay_duration_file),
        transition_probability_file=get_data_dir('preprocessed', 'mobility', transition_probability_file),
        timestep=timestep
    )
    params.set_new_district_seed(seed_infected=seed_num)

    model = scenarioClass(params, model_log_file=model_log_file, individual_log_file=individual_log_file)

    if start_date is not None:
        params.SIMULATION_START_DATE = start_date
        model.scheduler.real_time = params.SIMULATION_START_DATE

    model.load_agents(params.data_file_name, size=None, infect_num=params.SEED_INFECT_NUM)
    # end_date = model.scheduler.real_time + timedelta(days=30 * 24, hours=4)
    end_date = datetime(2020, 12, 1)

    while model.scheduler.real_time <= end_date:
        model.step()

        if ((model.epidemic_state >= model.STATE_INFECTED) & (model.epidemic_state < model.STATE_RECOVERED)).sum() == 0:
            break

    model_dump_file = get_data_dir('logs', f'model_dump_file_{sim_fname}.{now}.pickle')

    with open(model_dump_file, 'wb') as fl:
        pickle.dump(model, fl)
