# CVAPipe

[![Build Status](https://github.com/aics-int/cvapipe/workflows/Build%20Master/badge.svg)](https://github.com/aics-int/cvapipe/actions)

Workflow to manage processing of FOVs and Cells for the Cell Variance Analysis program.

---

## Features
All steps and functionality in this package can be run as single steps or all together
by using the command line.

### Single Steps
In general, all commands for this package will follow the format:
`cvapipe {step} {command}`

* `step` is the name of the step such as "GetDataset"
* `command` is what you want that step to do, such as "run" or "push"

### Whole Pipeline
To run the entire pipeline from start to finish you can simply run:

`cvapipe all run --dataset {path to dataset}`

### Step and Pipeline Commands

* `run`: run the processing for that single step or the entire pipeline
* `pull`: pull down the data required for the step provided (takes your current git
branch into account)
* `push`: push the steps data up (takes your current git branch into account)
* `checkout`: checkout the most recent data for your step and git branch
* `clean`: clean the steps local staging directory

## Installation
**Stable Release:** `pip install cvapipe`<br>
**Development Head:** `pip install git+https://github.com/aics-int/cvapipe.git`

## Development
See [CONTRIBUTING.md](https://github.com/aics-int/cvapipe/blob/master/CONTRIBUTING.md)
for information related to developing the code.

For more details on how this pipeline is constructed please see
[cookiecutter-stepworkflow](https://github.com/AllenCellModeling/cookiecutter-stepworkflow)
and [datastep](https://github.com/AllenCellModeling/datastep).

To add new steps to this pipeline, run `make_new_step` and follow the instructions in
[CONTRIBUTING.md](https://github.com/aics-int/cvapipe/blob/master/CONTRIBUTING.md)

Additionally, for step workflow specific development recommendations please read:
[DEV_RECOMMENDATIONS.md](https://github.com/aics-int/cvapipe/blob/master/DEV_RECOMMENDATIONS.md)

### AICS Developer Instructions
If you do not have the raw pipeline four data to run through the pipeline, run the
following commands to generate the starting dataset:

```bash
pip install -e .[all]
python scripts/create_aics_dataset.py
```

Options for this script are available and can be viewed with:
`python scripts/create_aics_dataset.py --help`

***Free software: Allen Institute Software License***
