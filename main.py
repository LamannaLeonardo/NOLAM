import datetime
import json
import os
import random
import shutil
import sys
from shutil import copyfile
from NOLAM.Learner import *
import pandas as pd

from Util.metrics import eval_model

if __name__ == "__main__":

    random.seed(Configuration.RANDOM_SEED)
    np.random.seed(Configuration.RANDOM_SEED)

    TRACES_DOMAIN_DIR = 'Analysis/Input traces/NOLAM'

    domains = sorted(os.listdir(TRACES_DOMAIN_DIR))

    os.makedirs("Analysis/Results/NOLAM/noisy_states", exist_ok=True)

    run = len([d for d in os.listdir("Analysis/Results/NOLAM/noisy_states") if 'run' in d])

    results = pd.DataFrame()

    for domain in domains:

        results_dir = os.path.join("Analysis/Results", "NOLAM", 'noisy_states', f"run{run}", domain)

        all_noise_rates = sorted([float(o) for o in os.listdir(os.path.join(TRACES_DOMAIN_DIR, domain, 'noisy_states'))])

        for noise_rate in all_noise_rates:

            os.makedirs('PDDL', exist_ok=True)

            traces = sorted(os.listdir(f'{TRACES_DOMAIN_DIR}/{domain}/noisy_states/{str(noise_rate)}'),
                            key=lambda x: int(x.split("_")[0]))
            trace_names = [os.path.join(TRACES_DOMAIN_DIR, domain, 'noisy_states', str(noise_rate), t) for t in traces]

            # Initialize empty input action model
            copyfile(os.path.join("Analysis/Benchmarks/", domain + ".pddl"),
                     "PDDL/domain_empty.pddl")
            old_stdout = sys.stdout
            sys.stdout = None
            gt_action_model = ActionModel(input_file="PDDL/domain_empty.pddl")
            gt_action_model.empty()
            gt_action_model.write('PDDL/domain_empty.pddl')
            sys.stdout = old_stdout

            print(f"\nProcessing input traces of domain {domain} with noise rate {noise_rate}")

            path_logs = os.path.join(results_dir, str(noise_rate))

            try:
                os.makedirs(path_logs, exist_ok=True)
            except OSError:
                print("Creation of the directory %s is failed" % path_logs)

            log_file_path = "{}/log".format(path_logs)
            log_file = open(log_file_path, "w")

            print("Running NOLAM...")

            old_stdout = sys.stdout

            if not Configuration.DEBUGGING:
                print(f'Standard output redirected to {log_file_path}')
                sys.stdout = log_file

            start = datetime.datetime.now()

            l = Learner()

            domain_learned = l.learn(trace_names, noise_rate)

            end = datetime.datetime.now()
            cpu_time = (end - start).microseconds * 1e-6

            # Store a copy of the learned domain
            domain_learned.write(f"{results_dir}/{noise_rate}/model.pddl")

            eval = {'Domain': domain} | eval_model(f"{results_dir}/{noise_rate}/model.pddl", f'Analysis/Benchmarks/{domain}.pddl')
            eval['Noise rate'] = noise_rate
            eval['CPU time'] = round(cpu_time, 2)

            with open(os.path.join(path_logs, "op_stats.json"), "w") as outfile:
                outfile.write(json.dumps(l.op_stats, indent=4))

            if not Configuration.DEBUGGING:
                sys.stdout = log_file

            print('\n\n')
            print(f'Number of processed traces: {len(traces)}')
            print(f'Total CPU time: {round(cpu_time, 2)}')

            log_file.close()

            sys.stdout = old_stdout

            print("End of NOLAM resolution.")

            # Clean PDDL files
            shutil.rmtree("PDDL")

            results = pd.concat([results, pd.DataFrame([eval])], ignore_index=True)


    writer = pd.ExcelWriter(os.path.join("Analysis/Results", "NOLAM", 'noisy_states', f"run{run}", 'nolam_results.xlsx'))
    results.to_excel(writer, index=False, float_format="%0.2f")
    writer.close()
