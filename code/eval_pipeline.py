#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import json
import numpy as np
import pandas as pd
import pydash
import shutil

from pathlib import Path

# Remember to update the PYTHON_PATH to
# export PYTHONPATH=`pwd`:`pwd`/conabio_ml_text/conabio_ml:`pwd`/conabio_ml_text

from conabio_ml.pipeline import Pipeline
from conabio_ml.assets import AssetTypes

from conabio_ml_text.datasets.dataset import Dataset, Partitions, PredictionDataset

from conabio_ml.evaluator.generic.evaluator import Evaluator, Metrics

from conabio_ml.utils.logger import get_logger, debugger
from conabio_ml.utils.report_params import languages

from conabio_ml.utils.logger import get_logger, debugger
from conabio_ml.utils.report_params import languages

from model  import pre_eval_converter
log = get_logger(__name__)
debug = debugger.debug


def multilabel_samples(dataset: Dataset.DatasetType,
                       dest_path: str,
                       process_args: dict,
                       **kwargs):
    dest_samples_path = Path(dest_path) / "samples_all_classes.csv"
    num_labels = len(dataset.params["labelmap"])

    res = {
        "samples_all_classes": {"asset": dest_samples_path,
                                "type": AssetTypes.FILE}
    }

    try:
        tmp = dataset.get_partition("test")
        groups = tmp.groupby("item")

        ixs = np.where(groups.apply(lambda x: len(x) > 1).values)

        group_list = list(groups)

        samples = pydash.chain(ixs[0])\
            .map(lambda x: tmp.loc[group_list[x][1].index])\
            .value()

        pd.concat(samples).to_csv(dest_samples_path)
    except Exception as ex:
        log.error("There was an error on reporting multilabel_samples")
        log.error(ex)

    return res


def mixed_samples(dataset: PredictionDataset.DatasetType,
                  dest_path: str,
                  process_args: dict,
                  **kwargs):
    dest_samples_path = Path(dest_path) / "mixed_samples.csv"
    th = process_args.get("decision_threshold", 0.5)

    res = {}
    try:
        unique_samples = dataset.data.groupby("item").indices.values()
        samples = []

        for sample in unique_samples:
            g_than = len(
                np.where(dataset.data.iloc[sample]["score"] > th)[0]) > 1
            if g_than:
                samples.append(dataset.data.iloc[sample])

        pd.concat(samples).to_csv(dest_samples_path)

        res = {
            "mixed_samples": {"asset": dest_samples_path,
                              "type": AssetTypes.FILE}
        }
    except Exception as ex:
        log.error("There was an error on reporting mixed_samples")
        log.error(ex)

    return res



def run(config_file: str):
    config = {}
    with open(config_file) as _f:
        config = json.load(_f)

    eval_path = config.get("eval_path", ".")
    dataset_filepath = config["dataset_filepath"]
    pred_dataset_filepath = config["pred_dataset_filepath"]
    eval_threshold = config.get("eval_threshold", 0.5)

    eval_path = Path(eval_path)
    results_path = Path(eval_path)
    
    dataset_filepath = eval_path / dataset_filepath
    pred_dataset_filepath = eval_path / pred_dataset_filepath

    pipeline = Pipeline(results_path,
                        name=f"eval_{eval_threshold}")\
        .add_process(name="dataset",
                     action=Dataset.from_csv,
                     reportable=True,
                     report_functions=[multilabel_samples],
                     args={
                         "info": {
                             "version": "0.1",
                             "description": (f"Abstracts of PLOS with 34813 abstracts for classes"
                                             f"[coastal, earth-sciences, freshwater, health-sciences, life-sciences, marine, terrestrial] "
                                             f"from 2021-01-06"),
                             "year": 2021,
                             "contributor": "CONABIO",
                             "collection": "-"
                         },
                         "source_path": dataset_filepath
                     })\
        .add_process(name="prediction_dataset",
                     action=PredictionDataset.from_csv,
                     reportable=True,
                     report_functions=[mixed_samples],
                     args={
                         "source_path": pred_dataset_filepath,
                         "decision_threshold": 0.4
                     })\
        .add_process(name="evaluate",
                     action=Evaluator.eval,
                     inputs_from_processes=["dataset",
                                            "prediction_dataset"],
                     reportable=True,
                     args={
                         "eval_config": {
                             "dataset_partition": Partitions.TEST,
                             "metrics_set": {
                                 Metrics.Sets.MULTILABEL: {
                                     'per_class': True,
                                     'average': 'macro',
                                     "zero_division": 1.0,
                                     "pre_eval_func": pre_eval_converter(eval_threshold)
                                 }
                             }}
                     })

    pipeline.run(report_lang=languages.ES)
    shutil.copy(config_file, Path(pipeline.path) / "config.json")
    shutil.copy("code/eval_pipeline.py", Path(pipeline.path) / "eval_pipeline.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("-f", "--config_file",
    #                     help="Config file for the queries. See TODO:")
    parser.add_argument("-d", "--debug",  action='store_true',
                        help="Enables debug mode to trace the progress of your searching")
    parser.add_argument("-b", "--breakpoint",  action='store_true',
                        help="Enables breakpoint inside the debugger wrapper")
    parser.add_argument("-c", "--config",
                        help="Config file for evaluation")

    ARGS = parser.parse_args()
    debugger.create(ARGS.debug, ARGS.breakpoint)

    run(ARGS.config)
