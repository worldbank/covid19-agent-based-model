import subprocess as sub
import os

num_simulations = 5

python_path = '/opt/anaconda3/envs/covid19_abm/bin/python'

scenarios = [
    # 'HandWashingRiskScenario',
    # 'HandWashingRiskImproved1Scenario',
    # 'HandWashingRiskImproved2Scenario',
    'UnmitigatedScenario',
    # 'LockdownGreatestMobilityScenario',
    # 'BlockGreatestMobilityScenario',
    'ContinuedLockdownScenario',
    # 'InteractionSensitivityScenario',
    'OpenSchoolsScenario',
    # 'OpenManufacturingAndSchoolsScenario',
    # 'OpenMiningScenario',
    # 'OpenManufacturingScenario',
    'EasedLockdownScenario',
    'EasedOpenSchoolsScenario',
]

run_env = os.environ.copy()
run_env['PATH'] = '/opt/anaconda3/envs/covid19_abm/bin/:' + run_env['PATH']
run_env['OMP_NUM_THREADS'] = '1'

run_scenarios_script = 'run_scenarios.py'

process_list = []
max_process_list = 30  # os.cpu_count() - 4

configs = []

sample_size = 100
seed_num = 100
R0_set = [1.9, 1.2, 1.4]

for R0 in R0_set:
    for sim_num in range(num_simulations):
        for scenario in scenarios:
            configs.append((scenario, str(sim_num), str(R0), str(sample_size), str(seed_num)))

while True:
    # First time in loop, populate list.
    if len(process_list) == 0:
        for _ in range(min(len(configs), max_process_list)):
            scenario, sim_num, R0, sample_size, seed_num = configs.pop(0)

            p = sub.Popen([python_path, run_scenarios_script, scenario, str(sim_num), str(R0), str(sample_size), str(seed_num)], env=run_env)
            print(f'Running {scenario} {R0} {str(sim_num).zfill(2)}...')
            process_list.append(p)

    active_procs = len(process_list)
    completed_procs = []

    for pix in range(active_procs):
        p = process_list[pix]
        if p.poll() is not None:
            completed_procs.append(p)

    for p in completed_procs:
        pix = process_list.index(p)
        process_list.pop(pix)

    if len(process_list) == max_process_list:
        continue

    if len(configs) > 0:
        scenario, sim_num, R0, sample_size, seed_num = configs.pop(0)

        p = sub.Popen([python_path, run_scenarios_script, scenario, str(sim_num), str(R0), str(sample_size), str(seed_num)], env=run_env)
        print(f'Running {scenario} {R0} {str(sim_num).zfill(2)}...')
        process_list.append(p)

    if len(process_list) == 0:
        break

# for scenario in scenarios:
#     for i in range(num_simulations):
#         p = sub.Popen([python_path, run_scenarios_script, scenario, str(i)], env=run_env)
#         print(f'Running {scenario} {str(i).zfill(2)}...')

# p.communicate()
