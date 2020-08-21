#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script will run all tasks in a prefect Flow.

When you add steps to you step workflow be sure to add them to the step list
and configure their IO in the `run` function.
"""

import logging
from datetime import datetime
from pathlib import Path

from dask_jobqueue import SLURMCluster
from distributed import LocalCluster
from prefect import Flow
from prefect.engine.executors import DaskExecutor, LocalExecutor

from cvapipe import steps

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class All:
    def __init__(self):
        """
        Set all of your available steps here.
        This is only used for data logging operations, not computation purposes.
        """
        self.step_list = [
            steps.ValidateDataset(),
            steps.PrepAnalysisSingleCellDs(),
            steps.MitoClass(),
            steps.MergeDataset(),
        ]

    def run(
        self,
        distributed: bool = False,
        overwrite: bool = False,
        debug: bool = False,
        **kwargs,
    ):
        """
        Run a flow with your steps.

        Parameters
        ----------
        distributed: bool
            A boolean option to determine if the jobs should be distributed to a SLURM
            cluster when possible.
            Default: False (Do not distribute)
        overwrite: bool
            If this pipeline has already partially or completely run, should it
            overwrite the previous files or not.
            Default: False (Do not overwrite or regenerate files)
        debug: bool
            A debug flag for the developer to use to manipulate how much data runs,
            how it is processed, etc. Additionally, if debug is True, any mapped
            operation will run on threads instead of processes.
            Default: False (Do not debug)

        Notes
        -----
        Documentation on prefect:
        https://docs.prefect.io/core/

        Basic prefect example:
        https://docs.prefect.io/core/
        """
        # Initalize steps
        validate_dataset = steps.ValidateDataset()
        prep_analysis_sc = steps.PrepAnalysisSingleCellDs()
        run_mito_class = steps.MitoClass()
        merge_data_for_cfe = steps.MergeDataset()

        # Choose executor
        if debug:
            exe = LocalExecutor()
            distributed_executor_address = None
            log.info("Debug flagged. Will use threads instead of Dask.")
        else:
            if distributed:
                # Create or get log dir
                # Do not include ms
                log_dir_name = datetime.now().isoformat().split(".")[0]
                log_dir = Path(f".dask_logs/{log_dir_name}").expanduser()
                # Log dir settings
                log_dir.mkdir(parents=True, exist_ok=True)

                # Create cluster
                log.info("Creating SLURMCluster")
                cluster = SLURMCluster(
                    cores=1,
                    memory="60GB",
                    queue="aics_gpu_general",
                    walltime="10:00:00",
                    local_directory=str(log_dir),
                    log_directory=str(log_dir),
                )

                # Spawn workers
                cluster.scale(15)
                log.info("Created SLURMCluster")

                # Use the port from the created connector to set executor address
                distributed_executor_address = cluster.scheduler_address

                # Log dashboard URI
                log.info(f"Dask dashboard available at: {cluster.dashboard_link}")
            else:
                # Create local cluster
                log.info("Creating LocalCluster")
                cluster = LocalCluster()
                log.info("Created LocalCluster")

                # Set distributed_executor_address
                distributed_executor_address = cluster.scheduler_address

                # Log dashboard URI
                log.info(f"Dask dashboard available at: {cluster.dashboard_link}")

            # Use dask cluster
            exe = DaskExecutor(distributed_executor_address)

        # Configure your flow
        with Flow("cvapipe") as flow:
            # Allows us to pass `--raw_dataset {some path}`
            validated_data_path = validate_dataset(**kwargs)

            single_cell_data_path = prep_analysis_sc(
                dataset=validated_data_path,
                distributed_executor_address=distributed_executor_address,
                **kwargs,
            )

            cell_data_with_annotation = run_mito_class(
                dataset=single_cell_data_path,
                distributed_executor_address=distributed_executor_address,
                **kwargs,
            )

            cell_data_cfe = merge_data_for_cfe(
                dataset_with_annotation=cell_data_with_annotation,
                dataset_from_labkey=validated_data_path,
                distributed_executor_address=distributed_executor_address,
                **kwargs,
            )

            #####################################################
            # remove this when new steps are added
            # "cell_data_with_annotation" is the file path to the dataset for analysis
            # "cell_data_cfe" is the file path to the dataset for CFE
            #####################################################
            print(f"data for CFE saved at {cell_data_cfe}")
            print(f"data for analysis saved at {cell_data_with_annotation}")
            #####################################################

        # Run flow and get ending state
        state = flow.run(executor=exe)

        # Get and display any outputs you want to see on your local terminal
        log.info(validate_dataset.get_result(state, flow))

    def pull(self):
        """
        Pull all steps.
        """
        for step in self.step_list:
            step.pull()

    def checkout(self):
        """
        Checkout all steps.
        """
        for step in self.step_list:
            step.checkout()

    def push(self):
        """
        Push all steps.
        """
        for step in self.step_list:
            step.push()

    def clean(self):
        """
        Clean all steps.
        """
        for step in self.step_list:
            step.clean()
