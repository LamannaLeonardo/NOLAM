import os
import re

import numpy as np

from NOLAM.Action import Action
from NOLAM.Observation import Observation
from NOLAM.Trace import Trace


def parse_trace_planminer(input_trace):
    with open(input_trace, 'r') as f:
        data = [el.strip() for el in f.read().split("\n") if el.strip() != '']

        data = [r for r in data if r.strip() not in ['', 'New plan!!!']]

        trace_observations = []
        trace_actions = []
        all_traces = []
        adding_state = True
        adding_action = False
        for r in data:
            if r.startswith('##Tasks'):
                adding_action = True
                adding_state = False

                if len(trace_observations) > 0:
                    all_traces.append(Trace(f"{input_trace}_{len(all_traces)}",
                                            trace_observations, trace_actions))
                    trace_observations = []
                    trace_actions = []

            elif r.startswith('##States'):
                adding_action = False
                adding_state = True

            elif adding_action:
                a = r.split(':')[1].strip()[1:-1].lower()
                a = re.sub(r'\s+-\s*\S+', '', a)
                a = f"({a})"

                a_name = a.strip()[1:-1].split()[0]

                a = a.strip()[1:-1]

                if len(a.split()) > 1:
                    objects = a.split()[1:]
                else:
                    objects = []
                action = Action(a_name, objects, None, None, None)
                trace_actions.append(action)
            elif adding_state:
                r = r.split(':')[1].strip().lower()

                neg_literals = [e.strip()[1:-1].replace('not', '', 1).strip()
                                for e in re.findall("\(not[^)]*\)\)", r)
                                if not len(e.replace('(and', '').replace(')', '').strip()) == 0]
                pos_literals = [e.strip() for e in re.findall("\([^()]*\)", r)
                                if e not in neg_literals and not len(e.replace('(and', '').replace(')', '').strip()) == 0]

                neg_literals = ['(' + re.sub(r'\s+-\s*\S+', '', n.strip()[1:-1]) + ')' for n in neg_literals]
                pos_literals = ['(' + re.sub(r'\s+-\s*\S+', '', n.strip()[1:-1]) + ')' for n in pos_literals]

                pos_literals = [
                    f"{l.strip()[1:-1].split()[0]}({f','.join([o for o in l.strip()[1:-1].split()[1:] if o != ''])})"
                    for l in pos_literals]
                neg_literals = [
                    f"not_{l.strip()[1:-1].split()[0]}({f','.join([o for o in l.strip()[1:-1].split()[1:] if o != ''])})"
                    for l in neg_literals]

                trace_observations.append(Observation(pos_literals + neg_literals))
            else:
                print(f'An error occurred while parsing input trace {input_trace}')
                exit()

    return all_traces


def parse_trace(input_trace):
    with open(input_trace, 'r') as f:
        data = [el.strip() for el in f.read().split("\n") if el.strip() != '']

        data = [r for r in data if r.strip().startswith('(:state') or r.startswith('(:action')]

        states = []
        actions = []
        adding_state = True
        adding_action = False
        for r in data:
            if adding_state:
                if r.startswith('(:state'):
                    states.append(r.replace('(:state', '').strip()[:-1].strip())
                    adding_action = True
                    adding_state = False
                elif r.startswith('(:action'):
                    states.append('')
                    actions.append(r.replace('(:action', '').strip()[:-1].strip())
                else:
                    print(f'Error when parsing input trace {input_trace}')
                    exit()

            elif adding_action:
                if r.startswith('(:action'):
                    actions.append(r.replace('(:action', '').strip()[:-1].strip())
                    adding_action = False
                    adding_state = True
                elif r.startswith('(:state'):
                    actions.append(None)
                    states.append(r.replace('(:state', '').strip()[:-1].strip())
                else:
                    print(f'Error when parsing input trace {input_trace}')
                    exit()

        trace_observations = []
        trace_actions = []

        for s in states:
            neg_literals = [e.strip()[1:-1].replace('not', '', 1).strip() for e in re.findall("\(not[^)]*\)\)", s)
                            if not len(e.replace('(and', '').replace(')', '').strip()) == 0]
            pos_literals = [e.strip() for e in re.findall("\([^()]*\)", s)
                            if e not in neg_literals and not len(e.replace('(and', '').replace(')', '').strip()) == 0]
            pos_literals = [
                f"{l.strip()[1:-1].split()[0]}({f','.join([o for o in l.strip()[1:-1].split()[1:] if o != ''])})"
                for l in pos_literals]
            neg_literals = [
                f"not_{l.strip()[1:-1].split()[0]}({f','.join([o for o in l.strip()[1:-1].split()[1:] if o != ''])})"
                for l in neg_literals]

            trace_observations.append(Observation(pos_literals + neg_literals))

        for a in actions:

            if a is None:
                trace_actions.append(None)
            else:
                a_name = a.strip()[1:-1].split()[0]
                a = a.strip()[1:-1]

                if len(a.split()) > 1:
                    objects = a.split()[1:]
                else:
                    objects = []

                action = Action(a_name, objects, None, None, None)
                trace_actions.append(action)

    return Trace(input_trace, trace_observations, trace_actions)


if __name__ == "__main__":

    NOLAM_TRACES_DIR = 'Analysis/Input traces/NOLAM/'
    PLANMINER_TRACES_DIR = 'Analysis/Input traces/PlanMiner/'

    for domain in os.listdir(NOLAM_TRACES_DIR):
        print(f'Processing domain {domain} traces...')
        for noise_rate in os.listdir(f'{NOLAM_TRACES_DIR}/{domain}/noisy_states'):

            # Parse NOLAM traces for a given domain and noise rate
            nolam_traces = [parse_trace(f'{NOLAM_TRACES_DIR}/{domain}/noisy_states/{noise_rate}/{nolam_trace}')
                            for nolam_trace in os.listdir(f'{NOLAM_TRACES_DIR}{domain}/noisy_states/{noise_rate}')]

            # Parse PlanMiner traces for a given domain and noise rate
            planminer_traces = parse_trace_planminer(f'{PLANMINER_TRACES_DIR}/{domain}/'
                                                     f'noisy_states/{noise_rate}/traces.pts')

            # Look for discrepancies in trace actions and observations
            for pm_t in planminer_traces:
                match = np.any([no_t == pm_t for no_t in nolam_traces])
                if not match:
                    print(f"Found no matching between planminer trace {pm_t.name} and NOLAM traces "
                          f"with noise rate {noise_rate}.")
