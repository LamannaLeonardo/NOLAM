# Action Model Learning from Noisy Traces: a Probabilistic Approach
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the official code of the Noisy Offline Learning of Action Models ([NOLAM](https://doi.org/10.1609/icaps.v34i1.31493)) algorithm.


### Installation
```
pip install nolam
```

### Example usage
```
from nolam.algorithm.Learner import Learner
noise_rate = 0.1
model = Learner().learn('path/to/domain.pddl', ['path/to/trace0', 'path/to/trace1'], e=noise_rate)
print(model)
```

## Custom domain learning

The NOLAM algorithm can be run for learning from traces with noisy states with an observation noise varying from 0 to 1. 
For running NOLAM on a custom domain, you need to provide an input domain file `'path/to/domain.pddl'`, a 
list of plan trace files `['path/to/trace0', 'path/to/trace1', etc.]`, and the (possibly estimated) observation noise. 
The input planning domain must contain the predicates, object types, and operator signatures. Note NOLAM does not 
yet exploit input knowledge in terms of preconditions and effects, hence providing such input domain knowledge 
does not currently affect the learning process. NOLAM can learn a planning domain from 
plan traces of different environments (e.g. it is possible to learn a planning domain from small environments 
and exploit the learned domain in large environments). 


## Citations
If you find this repository useful, please consider citing the related paper.
```
@article{lamanna2024action,
  title={Lifted Action Models Learning from Partial Traces},
  author={Lamanna, Leonardo and Serafini, Luciano},
  booktitle={Proceedings of the International Conference on Automated Planning and Scheduling},
  volume={34},
  pages={342--350},
  year={2024}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](/LICENSE) file for details.
