#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
import traceback
from pathlib import Path
import pandas as pd
from lkaccess import LabKey, contexts

###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
)
log = logging.getLogger(__name__)

###############################################################################
# Args


class Args(argparse.Namespace):
    def __init__(self):
        self.__parse()

    def __parse(self):
        # Setup parser
        p = argparse.ArgumentParser(
            prog="download_aics_dataset",
            description=(
                "Retrieve a dataset ready for processing from the internal "
                "AICS database."
            ),
        )

        # Arguments
        p.add_argument(
            "--sample",
            type=float,
            default=1.0,
            help=(
                "Percent how much data to download. Will be split across cell lines. "
                "Ex: 1.0 = 100 percent of each cell line, "
                "0.05 = 5 percent of each cell line."
            ),
        )
        p.add_argument(
            "--instance",
            default="PROD",
            help="Which database instance to use for data retrieval. (PROD or STAGING)",
        )
        p.add_argument(
            "--save_path",
            type=Path,
            default=Path("aics_p4_data.parquet"),
            help="Path to save the dataset to.",
        )
        p.add_argument(
            "--debug",
            action="store_true",
            help="Show traceback if the script were to fail.",
        )

        # Parse
        p.parse_args(namespace=self)


###############################################################################
# Retrieve and prepare AICS dataset


def create_aics_dataset(args: Args):
    # Try running the download pipeline
    try:
        # Get instance context
        instance_context = getattr(contexts, args.instance.upper())

        # Create connection to instance
        lk = LabKey(instance_context)
        log.info(f"Using LabKey instance: {lk}")

        # Get integrated cell data
        log.info("Retrieving pipeline integrated cell data...")
        data = pd.DataFrame(lk.dataset.get_pipeline_4_production_data())

        # Get cell line data
        log.info("Retrieving cell line data...")
        cell_line_data = pd.DataFrame(
            lk.select_rows_as_list(
                schema_name="celllines",
                query_name="CellLineDefinition",
                columns=[
                    "CellLineId",
                    "CellLineId/Name",
                    "StructureId/Name",
                    "ProteinId/Name",
                ],
            )
        )

        # Merge the data
        data = data.merge(cell_line_data, how="left", on="CellLineId")
        data = data.drop_duplicates(subset=["CellId"], keep="first")
        data = data.reset_index(drop=True)

        # Temporary until datasets 83 and 84 have structure segmentations
        data = data.loc[~data["DataSetId"].isin([83, 84])]

        # create a fov data frame
        df_fov = data.copy()
        df_fov.drop_duplicates(subset=["FOVId"], keep="first", inplace=True)
        df_fov.drop(["CellId", "CellIndex"], axis=1, inplace=True)

        # add two new colums 
        df_fov.assign(index_to_id_dict=None)
        df_fov.assign(id_to_index_dict=None)

        for row in df_fov.itertuples():
            fov_id = row.FOVId
            df_one_fov = data.query("FOVId==@fov_id")

            # collect all cells from this fov, and create mapping
            fov_index_to_id_dict = dict()
            fov_id_to_index_dict = dict()
            for cell_row in df_one_fov.itertuples():
                # Cast to string so that the values can be valid dictionary keys
                fov_index_to_id_dict[str(cell_row.CellIndex)] = str(cell_row.CellId)
                fov_id_to_index_dict[str(cell_row.CellId)] = str(cell_row.CellIndex)  

            # add dictioinary back to fov dataframe
            df_fov.at[row.Index, 'index_to_id_dict'] = [fov_index_to_id_dict]
            df_fov.at[row.Index, 'id_to_index_dict'] = [fov_id_to_index_dict]

        # The next statement is a tested assumption as of 14 July 2020 that this is a
        # valid drop statement. No data is different between the "FOV" and "Cell"
        # dataset besides the "CellId" and "CellIndex" columns
        #
        # from lkaccess import LabKey, contexts
        # import pandas as pd
        # lk = LabKey(contexts.PROD)
        # data = pd.DataFrame(lk.dataset.get_pipeline_4_production_data())
        #
        # for name, group in data.groupby("FOVId"):
        #     for col in group.columns:
        #         if (len(group[col].unique()) != 1
        #           and col not in ["CellId", "CellIndex"]):
        #               print(
        #                   f"FOVId: {name}", col,
        #                   len(group[col].unique()), len(group)
        #               )

        # Sample the data
        if args.sample != 1.0:
            log.info(f"Sampling dataset with frac={args.sample}...")
            df_fov = df_fov.groupby("CellLineId", group_keys=False)
            df_fov = df_fov.apply(pd.DataFrame.sample, frac=args.sample)
            df_fov = df_fov.reset_index(drop=True)

        # Save to Parquet
        df_fov.to_parquet(args.save_path)
        log.info(f"Saved dataset manifest to: {args.save_path}")

    # Catch any exception
    except Exception as e:
        log.error("=============================================")
        if args.debug:
            log.error("\n\n" + traceback.format_exc())
            log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)


###############################################################################
# Runner


def main():
    args = Args()
    create_aics_dataset(args)


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == "__main__":
    main()
