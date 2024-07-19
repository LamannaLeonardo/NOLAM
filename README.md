# Action Model Learning from Noisy Traces: a Probabilistic Approach
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the official code of the Noisy Offline Learning of Action Models ([NOLAM](https://doi.org/10.1609/icaps.v34i1.31493)) algorithm.


## Installation
The following instructions have been tested on macOS Ventura 13.3.1


1. Clone this repository:
```
 git clone https://github.com/LamannaLeonardo/OffLAM.git
```

2. Create a Python 3.9 virtual environment using conda (or [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-a-new-virtual-environment)):
```
 conda create -n nolam python=3.9
```

3. Activate the environment with conda (or [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#activate-a-virtual-environment)):
```
 conda activate nolam
```

4. Install dependencies:
```
pip install numpy pandas matplotlib openpyxl
```

5. Check everything is correctly installed by running `main.py` script.


## Execution

### Running NOLAM
The NOLAM algorithm can be run for learning from traces with noisy states with a noise ratio varying from 0 to 1. 
To run NOLAM on all domains in "Analysis/Input traces/NOLAM/", and all noise ratios in [0, 1], run the `main.py` script.

### Log and results
When you execute NOLAM, a new directory with all logs and results is created in the `Results/NOLAM/noisy_states` folder. For instance, when you run NOLAM for the first time, the logs and results are stored in the folder `Results/NOLAM/noisy_states/run0`. For each considered domain (e.g. blocksworld), a subdirectory is created (e.g. `Results/NOLAM/noisy_states/run0/blocksworld`), which consists of a log file and learned action model for each noise ratio in [0, 1].
In particular, each domain directory contain one subdirectory for every considered noise ratio (e.g. `Analysis/Results/NOLAM/noisy_states/run0/blocksworld/0.2`),
where you can find three files:
1. log: contains some debugging information
2. model.pddl: the learned model
3. op_stats.json: for each operator and potential precondition/effect of the operator, we store the counting of transitions where the ground atom is true/false after/before executing an instantiation of the operator.

Finally, each run directory contains the detailed results over all domains in a pandas dataframe named `nolam_results.xlsx`.


## NOLAM traces
For every domain and noise ratio in [0,1], the set of traces generated for evaluating NOLAM can be found in the directory `Analysis/Input traces/NOLAM/`.  



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
