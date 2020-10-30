from joblib import Parallel, delayed
from datetime import timedelta, datetime
import sys

import covid19_abm.scenario_models as sm


scenarios = {
    'LockdownGreatestMobilityScenario': sm.LockdownGreatestMobilityScenario,
    'BlockGreatestMobilityScenario': sm.BlockGreatestMobilityScenario,
    'UnmitigatedScenario': sm.UnmitigatedScenario,
    'ContinuedLockdownScenario': sm.ContinuedLockdownScenario,
    'OpenSchoolsScenario': sm.OpenSchoolsScenario,
    'OpenSchoolsSeedKidsScenario': sm.OpenSchoolsSeedKidsScenario,
    'OpenManufacturingAndSchoolsScenario': sm.OpenManufacturingAndSchoolsScenario,
    'HandWashingRiskScenario': sm.HandWashingRiskScenario,
    'HandWashingRiskImproved1Scenario': sm.HandWashingRiskImproved1Scenario,
    'HandWashingRiskImproved2Scenario': sm.HandWashingRiskImproved2Scenario,
    'InteractionSensitivityScenario': sm.InteractionSensitivityScenario,
    'OpenMiningScenario': sm.OpenMiningScenario,
    'OpenManufacturingScenario': sm.OpenManufacturingScenario,
    'EasedLockdownScenario': sm.EasedLockdownScenario,
    'EasedOpenSchoolsScenario': sm.EasedOpenSchoolsScenario,
    'Phase1GovernmentOpenSchoolsScenario': sm.Phase1GovernmentOpenSchoolsScenario,
    'DynamicPhase1GovernmentOpenSchoolsScenario': sm.DynamicPhase1GovernmentOpenSchoolsScenario,
    'AcceleratedGovernmentOpenSchoolsScenario': sm.AcceleratedGovernmentOpenSchoolsScenario,
}

# R0 = 1.9
# sample_size = 10
# seed_num = 90
timestep = timedelta(hours=4)
start_date = datetime(2020, 9, 1)


if __name__ == '__main__':
    scenario_name = sys.argv[1]
    sim_fname = sys.argv[2]
    R0 = float(sys.argv[3])
    sample_size = int(sys.argv[4])
    seed_num = int(sys.argv[5])

    scenarioClass = scenarios[scenario_name]

    sm.run_scenario(scenarioClass, scenario_name, sim_fname, R0, sample_size, seed_num, start_date=start_date, timestep=timestep)
