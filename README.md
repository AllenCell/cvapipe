# CVAPipe

[![Build Status](https://github.com/AllenCell/cvapipe/workflows/Build%20Master/badge.svg)](https://github.com/AllenCell/cvapipe/actions)

Workflow to manage processing of FOVs and Cells for the Cell Variance Analysis program.

---

## Features
All steps and functionality in this package can be run as single steps or all together
by using the command line.

### Single Steps
In general, all commands for this package will follow the format:
`cvapipe {step} {command}`

* `step` is the name of the step such as "ValidateDataset"
* `command` is what you want that step to do, such as "run" or "push"

**Available Steps**
* `validate_dataset`: `cvapipe validatedataset run --raw_dataset {path_to_dataset}`
will validate that the provided dataset can be processed by the downstream steps.
* `prep_analysis_single_cell_ds`: `cvapipe prepanalysissinglecellds run --dataset /path/to/cell_table.parquet` will prepare the data table for analysis and other downstream steps
* `mito_class`: `cvapipe mitoclass run --dataset /path/to/preprocessed/cell_table.csv` will run 
mitotic classifer and generate the manifest for analysis
* `merge_dataset`: `cvapipe mergedataset run --dataset_with_annotation /path/to/manifest/from/step1 --dataset_from_labkey /path/to/manifest/from/step3` will generate the manifest for CFE

### Whole Pipeline
To run the entire pipeline from start to finish you can simply run:

`cvapipe all run --raw_dataset {path to dataset}`

*Note: The mitotic classifier step was implemented with pytorch-lightning (PLT). PLT support running on slurm in two ways: by submitting a slurm job or with a customized [SlurmCluster API](https://williamfalcon.github.io/test-tube/hpc/SlurmCluster/#slurmcluster-class-api), which is different from the SlurmClaster from Dask. So, the whole pipeline will only run through first 2 steps. The last 2 steps need to run as single steps*

### Step and Pipeline Commands

* `run`: run the processing for that single step or the entire pipeline
* `pull`: pull down the data required for the step provided (takes your current git
branch into account)
* `push`: push the steps data up (takes your current git branch into account)
* `checkout`: checkout the most recent data for your step and git branch
* `clean`: clean the steps local staging directory

## Installation
**Stable Release:** `pip install cvapipe`<br>
**Development Head:** `pip install git+https://github.com/AllenCell/cvapipe.git`

## Development
See [CONTRIBUTING.md](https://github.com/AllenCell/cvapipe/blob/master/CONTRIBUTING.md)
for information related to developing the code.

For more details on how this pipeline is constructed please see
[cookiecutter-stepworkflow](https://github.com/AllenCellModeling/cookiecutter-stepworkflow)
and [datastep](https://github.com/AllenCellModeling/datastep).

To add new steps to this pipeline, run `make_new_step` and follow the instructions in
[CONTRIBUTING.md](https://github.com/AllenCell/cvapipe/blob/master/CONTRIBUTING.md)

Additionally, for step workflow specific development recommendations please read:
[DEV_RECOMMENDATIONS.md](https://github.com/AllenCell/cvapipe/blob/master/DEV_RECOMMENDATIONS.md)

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
