#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from datastep import Step, log_run_params
from image_classifier_3d.proj_tester import ProjectTester

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class MitoClass(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
        dataset: Union[str, Path],
        distributed_executor_address: Optional[str] = None,
        debug: bool = False,
        **kwargs
    ) -> List[Path]:
        """
        Run a pure function.

        Protected Parameters
        --------------------
        distributed_executor_address: Optional[str]
            An optional executor address to pass to some computation engine.
        debug: bool
            A debug flag for the developer to use to manipulate how much data runs,
            how it is processed, etc.
            Default: False (Do not debug)

        Parameters
        ----------
        dataset: Union[str, Path]
            The dataset to use for running the classifier.

            **Required dataset columns:** *["CellId", "crop_raw",
            "crop_seg"]*

        Returns
        -------
        manifest_save_path: Path
            Path to the produced manifest of single cell dataset
            with predicted class label
        """
        # path to save intermediate prediction results
        pred_path = self.step_local_staging_dir / "predictions"
        pred_path.mkdir(exist_ok=True)
        my_classifier = ProjectTester(save_model_output=False)
        df_pred = my_classifier.run_tester_csv(dataset, pred_path, return_df=True)

        self.manifest = df_pred
        manifest_save_path = self.step_local_staging_dir / "manifest.csv"
        df_pred.to_csv(manifest_save_path, index=False)

        return manifest_save_path
