import itertools
import logging
import math
import re

import numpy as np

from nolam.algorithm import Configuration
from nolam.algorithm.Action import Action
from nolam.algorithm.ActionModel import ActionModel
from nolam.algorithm.Observation import Observation
from nolam.algorithm.Trace import Trace


class Learner:

    def __init__(self):
        self.op_stats = None

    def count_traces(self, trace_names, action_model):
        traces = [self.parse_trace(t, action_model) for t in trace_names]
        for trace in traces:

            # Check negative literals exist
            neg_literals_count = sum([len(v) for o in trace.observations
                                      for k, v in o.negative_literals.items()])
            if neg_literals_count == 0:
                logging.warning(f"There are no negative literals in trace {trace.name}. "
                                f"NOLAM assumes trace observations to explicitly specify "
                                f"both positive and negative literals.")


            for i in range(len(trace.observations) - 1):

                prev_observation = trace.observations[i]
                prev_action = trace.actions[i]
                next_observation = trace.observations[i + 1]

                op_name = prev_action.operator_name

                # Update transitions counting
                pos_pos = {p for p in prev_action.eff_pos
                           if p in next_observation.positive_literals[p.split('(')[0]]
                           and p in prev_observation.positive_literals[p.split('(')[0]]}
                pos_neg = {p for p in prev_action.eff_pos
                           if f"not_{p}" in next_observation.negative_literals[p.split('(')[0]]
                           and p in prev_observation.positive_literals[p.split('(')[0]]}
                neg_pos = {p for p in prev_action.eff_pos
                           if p in next_observation.positive_literals[p.split('(')[0]]
                           and f"not_{p}" in prev_observation.negative_literals[p.split('(')[0]]}
                neg_neg = {p for p in prev_action.eff_pos
                           if f"not_{p}" in next_observation.negative_literals[p.split('(')[0]]
                           and f"not_{p}" in prev_observation.negative_literals[p.split('(')[0]]}

                for ground_atom in pos_pos:
                    # for lifted_atom in self.lift_ground_atoms(prev_action, pos_pos):
                    ambiguous_bind = len(self.lift_ground_atoms(prev_action, [ground_atom])) > 1
                    if not ambiguous_bind:
                        for lifted_atom in self.lift_ground_atoms(prev_action, [ground_atom]):
                                if lifted_atom in self.op_stats[op_name]:
                                    self.op_stats[op_name][lifted_atom]['pos-pos'] += 1
                                else:
                                    print(f'[Debug] Warning: atom {lifted_atom} is not in the set of all possible positive '
                                          f'effects of operator {op_name}. You may want to check this is correct.')

                for ground_atom in pos_neg:
                    # for lifted_atom in self.lift_ground_atoms(prev_action, pos_neg):
                    ambiguous_bind = len(self.lift_ground_atoms(prev_action, [ground_atom])) > 1
                    if not ambiguous_bind:
                        for lifted_atom in self.lift_ground_atoms(prev_action, [ground_atom]):
                            if not ambiguous_bind:
                                if lifted_atom in self.op_stats[op_name]:
                                    self.op_stats[op_name][lifted_atom]['pos-neg'] += 1
                                else:
                                    print(f'[Debug] Warning: atom {lifted_atom} is not in the set of all possible positive '
                                          f'effects of operator {op_name}. You may want to check this is correct.')

                for ground_atom in neg_neg:
                    # for lifted_atom in self.lift_ground_atoms(prev_action, neg_neg):
                    ambiguous_bind = len(self.lift_ground_atoms(prev_action, [ground_atom])) > 1
                    if not ambiguous_bind:
                        for lifted_atom in self.lift_ground_atoms(prev_action, [ground_atom]):
                            if not ambiguous_bind:
                                if lifted_atom in self.op_stats[op_name]:
                                    self.op_stats[op_name][lifted_atom]['neg-neg'] += 1
                                else:
                                    print(f'[Debug] Warning: atom {lifted_atom} is not in the set of all possible positive '
                                          f'effects of operator {op_name}. You may want to check this is correct.')

                for ground_atom in neg_pos:
                    # for lifted_atom in self.lift_ground_atoms(prev_action, neg_pos):
                    ambiguous_bind = len(self.lift_ground_atoms(prev_action, [ground_atom])) > 1
                    if not ambiguous_bind:
                        for lifted_atom in self.lift_ground_atoms(prev_action, [ground_atom]):
                            if lifted_atom in self.op_stats[op_name]:
                                self.op_stats[op_name][lifted_atom]['neg-pos'] += 1
                            else:
                                print(f'[Debug] Warning: atom {lifted_atom} is not in the set of all possible positive '
                                      f'effects of operator {op_name}. You may want to check this is correct.')

        return self.op_stats

    def parse_trace(self, input_trace, action_model):

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
                pos_literals = [f"{l.strip()[1:-1].split()[0]}({f','.join([o for o in l.strip()[1:-1].split()[1:] if o != ''])})"
                                for l in pos_literals]
                neg_literals = [f"not_{l.strip()[1:-1].split()[0]}({f','.join([o for o in l.strip()[1:-1].split()[1:] if o != ''])})"
                                for l in neg_literals]

                trace_observations.append(Observation(pos_literals + neg_literals))

            for a in actions:

                if a is None:
                    trace_actions.append(None)
                else:
                    a_name = a.strip()[1:-1].split()[0]
                    a = a.strip()[1:-1]

                    operator = next((o for o in action_model.operators if o.operator_name == a_name), None)

                    if len(a.split()) > 1:
                        objects = a.split()[1:]
                        params_bind = {f'?param_{i + 1}': obj for i, obj in enumerate(objects)}
                    else:
                        objects = []

                    action_precs_pos = {self.ground_lifted_atom(params_bind, p) for p in operator.precs_pos}
                    action_precs_neg = {self.ground_lifted_atom(params_bind, p) for p in operator.precs_neg}
                    action_eff_pos_cert = {self.ground_lifted_atom(params_bind, p) for p in operator.eff_pos}
                    action_eff_neg_cert = {self.ground_lifted_atom(params_bind, p) for p in operator.eff_neg}
                    action = Action(a_name, objects, action_precs_pos, action_precs_neg, action_eff_pos_cert, action_eff_neg_cert)

                    ground_model_actions = [str(a) for a in action_model.ground_actions[action.operator_name]]
                    if str(action) not in ground_model_actions:
                        trace_actions.append(action)
                        action_model.ground_actions[action.operator_name].append(action)
                        action_model.ground_action_labels.add(str(action))
                    else:
                        action_idx = ground_model_actions.index(str(action))
                        trace_actions.append(action_model.ground_actions[action.operator_name][action_idx])

        return Trace(input_trace, trace_observations, trace_actions)

    def ground_lifted_atom(self, action_params_bind, lifted_atom):
        lifted_atom_split = lifted_atom.split('(')
        lifted_atom_params = [p for p in lifted_atom_split[1][:-1].split(',') if p != '']
        return f"{lifted_atom_split[0]}({','.join([action_params_bind[p] for p in lifted_atom_params])})"

    def lift_ground_atoms(self, ground_action, ground_atoms):
        lifted_precs = []
        action_params = ground_action.parameters
        action_params_bind = {p: [f'?param_{i + 1}' for i in range(len(action_params)) if action_params[i] == p]
                              for p in action_params}

        for atom in ground_atoms:
            prec_objects = [o for o in atom.split('(')[1][:-1].split(',') if o.strip() != '']
            try:
                params_bind_combinations = [list(p) for p in itertools.product(*[action_params_bind[obj]
                                                                                 for obj in prec_objects])]
                for tup in params_bind_combinations:
                    lifted_prec = f"{atom.split('(')[0]}({','.join(tup)})"
                    lifted_precs.append(lifted_prec)
            except:
                print(f'Warning: cannot lift ground atom {atom} for action {ground_action}')

        return lifted_precs

    def learn(self, input_file, trace_names, e):

        domain_learned = ActionModel(input_file)
        domain_learned.empty()
        domain_learned.init_prec_eff()
        self.op_stats = {o.operator_name: {p: {'pos-pos': 0, 'pos-neg': 0, 'neg-pos': 0, 'neg-neg': 0}
                                           for p in o.eff_pos}
                         for o in domain_learned.operators}
        # Parse traces by considering action model with all possible precs/effs
        op_stats = self.count_traces(trace_names, domain_learned)

        domain_learned.empty()

        for operator in domain_learned.operators:

            for atom in op_stats[operator.operator_name]:

                Npp = op_stats[operator.operator_name][atom]['pos-pos']
                Nmp = op_stats[operator.operator_name][atom]['neg-pos']
                Npm = op_stats[operator.operator_name][atom]['pos-neg']
                Nmm = op_stats[operator.operator_name][atom]['neg-neg']

                N = Npp + Nmp + Npm + Nmm

                coeff = math.factorial(N) / (math.factorial(Npp) * math.factorial(Nmm)
                                             * math.factorial(Npm) * math.factorial(Nmp))

                hypothesis = np.array([('none', 'none'),
                                       ('none', 'pos'),
                                       ('none', 'neg'),
                                       ('pos', 'none'),
                                       ('neg', 'none'),
                                       ('pos', 'pos'),
                                       ('neg', 'pos'),
                                       ('pos', 'neg'),
                                       ('neg', 'neg')])

                PrNmm = {k: {v: None for v in hypothesis[:, 1]} for k in hypothesis[:, 0]}
                PrNmp = {k: {v: None for v in hypothesis[:, 1]} for k in hypothesis[:, 0]}
                PrNpm = {k: {v: None for v in hypothesis[:, 1]} for k in hypothesis[:, 0]}
                PrNpp = {k: {v: None for v in hypothesis[:, 1]} for k in hypothesis[:, 0]}

                prior = {k: {v: None for v in hypothesis[:, 1]} for k in hypothesis[:, 0]}

                if N > 0:

                    prior_prepos = ((Npp + Npm) / N) * 2 / 3
                    prior_preneg = ((Nmp + Nmm) / N) * 2 / 3
                    prior_prenone = 1 / 3

                    if not Configuration.ALLOW_PREC_NEG:
                        prior_preneg = 0
                        tot = prior_prepos + prior_prenone
                        prior_prepos = prior_prepos / tot
                        prior_prenone = prior_prenone / tot

                    prior_effpos = Nmp / N
                    prior_effneg = Npm / N
                    prior_effnone = (Nmm + Npp) / N

                    prior['none']['none'] = prior_prenone * prior_effnone
                    prior['none']['pos'] = prior_prenone * prior_effpos
                    prior['none']['neg'] = prior_prenone * prior_effneg
                    prior['pos']['none'] = prior_prepos * prior_effnone
                    prior['neg']['none'] = prior_preneg * prior_effnone
                    prior['pos']['neg'] = prior_prepos * prior_effneg
                    prior['neg']['pos'] = prior_preneg * prior_effpos
                    prior['neg']['neg'] = prior_preneg * prior_effneg
                    prior['pos']['pos'] = prior_prepos * prior_effpos

                else:  # operator is never observed in the traces
                    if Configuration.ALLOW_PREC_NEG:
                        prior['none']['none'] = 1 / 9
                        prior['none']['pos'] = 1 / 9
                        prior['none']['neg'] = 1 / 9
                        prior['pos']['none'] = 1 / 9
                        prior['neg']['none'] = 1 / 9
                        prior['pos']['neg'] = 1 / 9
                        prior['neg']['pos'] = 1 / 9
                        prior['neg']['neg'] = 1 / 9
                        prior['pos']['pos'] = 1 / 9
                    else:
                        prior['none']['none'] = 1 / 6
                        prior['none']['pos'] = 1 / 6
                        prior['none']['neg'] = 1 / 6
                        prior['pos']['none'] = 1 / 6
                        prior['neg']['none'] = 0
                        prior['pos']['neg'] = 1 / 6
                        prior['neg']['pos'] = 0
                        prior['neg']['neg'] = 0
                        prior['pos']['pos'] = 1 / 6

                assert 1 - 1e-5 < prior['none']['none'] \
                       + prior['none']['pos'] \
                       + prior['none']['neg'] \
                       + prior['pos']['none'] \
                       + prior['neg']['none'] \
                       + prior['pos']['pos'] \
                       + prior['neg']['pos'] \
                       + prior['pos']['neg'] \
                       + prior['neg']['neg'] < 1 + 1e-5

                posterior = {k: {v: None for v in hypothesis[:, 1]} for k in hypothesis[:, 0]}
                posterior['none']['none'] = 1e-10
                posterior['none']['pos'] = 1e-10
                posterior['none']['neg'] = 1e-10
                posterior['pos']['none'] = 1e-10
                posterior['neg']['none'] = 1e-10
                posterior['pos']['pos'] = 1e-10
                posterior['neg']['pos'] = 1e-10
                posterior['pos']['neg'] = 1e-10
                posterior['neg']['neg'] = 1e-10

                likelihood = {k: {v: None for v in hypothesis[:, 1]} for k in hypothesis[:, 0]}

                if N > 0:
                    k_neg = 1 / 2

                    PrNmm['none']['none'] = k_neg * ((1 - e) ** 2) + (1 - k_neg) * (e ** 2)
                    PrNmp['none']['none'] = k_neg * (1 - e) * e + (1 - k_neg) * e * (1 - e)
                    PrNpm['none']['none'] = k_neg * e * (1 - e) + (1 - k_neg) * (1 - e) * e
                    PrNpp['none']['none'] = k_neg * (e ** 2) + (1 - k_neg) * ((1 - e) ** 2)

                    PrNmm['none']['pos'] = k_neg * (1 - e) * e + (1 - k_neg) * (e ** 2)
                    PrNmp['none']['pos'] = k_neg * ((1 - e) ** 2) + (1 - k_neg) * (1 - e) * e
                    PrNpm['none']['pos'] = (1 - k_neg) * (1 - e) * e + k_neg * (e ** 2)
                    PrNpp['none']['pos'] = (1 - k_neg) * ((1 - e) ** 2) + k_neg * (1 - e) * e

                    PrNmm['none']['neg'] = k_neg * ((1 - e) ** 2) + (1 - k_neg) * (1 - e) * e
                    PrNmp['none']['neg'] = k_neg * (1 - e) * e + (1 - k_neg) * (e ** 2)
                    PrNpm['none']['neg'] = (1 - k_neg) * ((1 - e) ** 2) + k_neg * (1 - e) * e
                    PrNpp['none']['neg'] = (1 - k_neg) * (1 - e) * e + k_neg * (e ** 2)

                    PrNmm['pos']['none'] = e ** 2
                    PrNmp['pos']['none'] = e * (1 - e)
                    PrNpm['pos']['none'] = (1 - e) * e
                    PrNpp['pos']['none'] = (1 - e) ** 2

                    PrNmm['neg']['none'] = (1 - e) ** 2
                    PrNmp['neg']['none'] = (1 - e) * e
                    PrNpm['neg']['none'] = e * (1 - e)
                    PrNpp['neg']['none'] = e ** 2

                    PrNmm['neg']['pos'] = (1 - e) * e
                    PrNmp['neg']['pos'] = (1 - e) ** 2
                    PrNpm['neg']['pos'] = e ** 2
                    PrNpp['neg']['pos'] = e * (1 - e)

                    PrNmm['pos']['neg'] = e * (1 - e)
                    PrNmp['pos']['neg'] = e ** 2
                    PrNpm['pos']['neg'] = (1 - e) ** 2
                    PrNpp['pos']['neg'] = (1 - e) * e

                    PrNmm['neg']['neg'] = (1 - e) ** 2
                    PrNmp['neg']['neg'] = (1 - e) * e
                    PrNpm['neg']['neg'] = e * (1 - e)
                    PrNpp['neg']['neg'] = e ** 2

                    PrNmm['pos']['pos'] = e ** 2
                    PrNmp['pos']['pos'] = e * (1 - e)
                    PrNpm['pos']['pos'] = (1 - e) * e
                    PrNpp['pos']['pos'] = (1 - e) ** 2

                    likelihood['none']['none'] = coeff * PrNmm['none']['none'] ** Nmm \
                                                 * PrNmp['none']['none'] ** Nmp \
                                                 * PrNpm['none']['none'] ** Npm \
                                                 * PrNpp['none']['none'] ** Npp
                    likelihood['none']['pos'] = coeff * PrNmm['none']['pos'] ** Nmm \
                                                * PrNmp['none']['pos'] ** Nmp \
                                                * PrNpm['none']['pos'] ** Npm \
                                                * PrNpp['none']['pos'] ** Npp
                    likelihood['none']['neg'] = coeff * PrNmm['none']['neg'] ** Nmm \
                                                * PrNmp['none']['neg'] ** Nmp \
                                                * PrNpm['none']['neg'] ** Npm \
                                                * PrNpp['none']['neg'] ** Npp
                    likelihood['pos']['none'] = coeff * PrNmm['pos']['none'] ** Nmm \
                                                * PrNmp['pos']['none'] ** Nmp \
                                                * PrNpm['pos']['none'] ** Npm \
                                                * PrNpp['pos']['none'] ** Npp
                    likelihood['neg']['none'] = coeff * PrNmm['neg']['none'] ** Nmm \
                                                * PrNmp['neg']['none'] ** Nmp \
                                                * PrNpm['neg']['none'] ** Npm \
                                                * PrNpp['neg']['none'] ** Npp
                    likelihood['pos']['pos'] = coeff * PrNmm['pos']['pos'] ** Nmm \
                                               * PrNmp['pos']['pos'] ** Nmp \
                                               * PrNpm['pos']['pos'] ** Npm \
                                               * PrNpp['pos']['pos'] ** Npp
                    likelihood['neg']['pos'] = coeff * PrNmm['neg']['pos'] ** Nmm \
                                               * PrNmp['neg']['pos'] ** Nmp \
                                               * PrNpm['neg']['pos'] ** Npm \
                                               * PrNpp['neg']['pos'] ** Npp
                    likelihood['pos']['neg'] = coeff * PrNmm['pos']['neg'] ** Nmm \
                                               * PrNmp['pos']['neg'] ** Nmp \
                                               * PrNpm['pos']['neg'] ** Npm \
                                               * PrNpp['pos']['neg'] ** Npp
                    likelihood['neg']['neg'] = coeff * PrNmm['neg']['neg'] ** Nmm \
                                               * PrNmp['neg']['neg'] ** Nmp \
                                               * PrNpm['neg']['neg'] ** Npm \
                                               * PrNpp['neg']['neg'] ** Npp

                    N_prior = likelihood['none']['none'] * prior['none']['none'] \
                              + likelihood['none']['pos'] * prior['none']['pos'] \
                              + likelihood['none']['neg'] * prior['none']['neg'] \
                              + likelihood['pos']['none'] * prior['pos']['none'] \
                              + likelihood['neg']['none'] * prior['neg']['none'] \
                              + likelihood['pos']['pos'] * prior['pos']['pos'] \
                              + likelihood['neg']['pos'] * prior['neg']['pos'] \
                              + likelihood['pos']['neg'] * prior['pos']['neg'] \
                              + likelihood['neg']['neg'] * prior['neg']['neg']

                    if N_prior != 0:
                        posterior['none']['none'] = likelihood['none']['none'] * prior['none']['none'] / N_prior
                        posterior['none']['pos'] = likelihood['none']['pos'] * prior['none']['pos'] / N_prior
                        posterior['none']['neg'] = likelihood['none']['neg'] * prior['none']['neg'] / N_prior
                        posterior['pos']['none'] = likelihood['pos']['none'] * prior['pos']['none'] / N_prior
                        posterior['neg']['none'] = likelihood['neg']['none'] * prior['neg']['none'] / N_prior
                        posterior['pos']['pos'] = likelihood['pos']['pos'] * prior['pos']['pos'] / N_prior
                        posterior['neg']['pos'] = likelihood['neg']['pos'] * prior['neg']['pos'] / N_prior
                        posterior['pos']['neg'] = likelihood['pos']['neg'] * prior['pos']['neg'] / N_prior
                        posterior['neg']['neg'] = likelihood['neg']['neg'] * prior['neg']['neg'] / N_prior

                assert posterior['none']['none'] \
                       + posterior['none']['pos'] \
                       + posterior['none']['neg'] \
                       + posterior['pos']['none'] \
                       + posterior['neg']['none'] \
                       + posterior['pos']['pos'] \
                       + posterior['neg']['pos'] \
                       + posterior['pos']['neg'] \
                       + posterior['neg']['neg'] < 1 + 1e-5

                if N > 0:

                    b = np.array([posterior[x[0]][x[1]] for x in hypothesis])

                    # MAXIMUM A POSTERIORI
                    if not Configuration.SAMPLING:
                        h = np.random.choice(np.flatnonzero(b == b.max()))
                        # h = max_posterior[0]
                        h = hypothesis[h]
                        if tuple(h) == ('none', 'pos'):
                            operator.eff_pos.add(atom)
                        elif tuple(h) == ('none', 'neg'):
                            operator.eff_neg.add(atom)
                        elif tuple(h) == ('pos', 'none'):
                            operator.precs_pos.add(atom)
                        elif tuple(h) == ('neg', 'none'):
                            operator.precs_neg.add(atom)
                            # pass
                        elif tuple(h) == ('neg', 'pos'):
                            operator.precs_neg.add(atom)
                            operator.eff_pos.add(atom)
                        elif tuple(h) == ('pos', 'neg'):
                            operator.precs_pos.add(atom)
                            operator.eff_neg.add(atom)

                    # SAMPLING
                    else:
                        if 0.99 < sum(b) < 1.01:
                            b = b.round(2)
                            b /= b.sum()  # normalize for numerical errors
                        h = np.random.choice([f"{x[0]},{x[1]}" for x in hypothesis], p=b)
                        h = h.split(',')
                        if tuple(h) == ('none', 'pos'):
                            operator.eff_pos.add(atom)
                        elif tuple(h) == ('none', 'neg'):
                            operator.eff_neg.add(atom)
                        elif tuple(h) == ('pos', 'none'):
                            operator.precs_pos.add(atom)
                        elif tuple(h) == ('neg', 'none'):
                            operator.precs_neg.add(atom)
                            # pass
                        elif tuple(h) == ('neg', 'pos'):
                            operator.precs_neg.add(atom)
                            operator.eff_pos.add(atom)
                        elif tuple(h) == ('pos', 'neg'):
                            operator.precs_pos.add(atom)
                            operator.eff_neg.add(atom)

        return domain_learned
