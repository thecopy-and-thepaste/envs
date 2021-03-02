#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import os
from typing import Union
import pandas as pd
import numpy as np
import shutil

from datetime import datetime

import tensorflow as tf

from pathlib import Path

# Remember to update the PYTHON_PATH to
# export PYTHONPATH=`pwd`:`pwd`/conabio_ml_text/conabio_ml:`pwd`/conabio_ml_text

from conabio_ml.pipeline import Pipeline

from conabio_ml_text.datasets.dataset import Dataset, Partitions
from conabio_ml_text.preprocessing.preprocessing import Tokens
from conabio_ml_text.preprocessing.transform import Transform

from conabio_ml_text.trainers.bcknds.tfkeras import TFKerasTrainer, TFKerasTrainerConfig
from conabio_ml_text.trainers.bcknds.tfkeras import CHECKPOINT_CALLBACK, TENSORBOARD_CALLBACK

from conabio_ml.evaluator.generic.evaluator import Evaluator, Metrics

from conabio_ml.utils.logger import get_logger, debugger
from conabio_ml_text.utils.constraints import TransformRepresentations, LearningRates

from conabio_ml_text.trainers.builders import create_learning_rate

from model import EnvironmentClassifier,  multilabel_topk, HammingLoss

log = get_logger(__name__)
debug = debugger.debug


def run(config_file: str):

    config = {}
    with open(config_file) as _f:
        config = json.load(_f)

    results_path = Path(f"pipelines/results")
    dataset_filepath = Path(f"code/{config['dataset']}")
    vocab_filepath = Path(f"code/{config['vocab']}")

    # Model layers
    layers = config["layers"]

    # Params

    # Learning rate
    initial_lr = config["params"]["initial_learning_rate"]
    # Decay steps for the lr
    decay_steps = config["params"]["decay_steps"]
    # Batch size
    batch_size = config["params"]["batch_size"]
    # Epochs of training
    epochs = config["params"]["epochs"]
    # Threshold of multilabel
    hamming_loss_th = config["params"]["hamming_loss_threshold"]

    multilabel_k_classes = config["params"]["multilabel_classes"]

    TRAIN_REPRESENTATION = TransformRepresentations.DATA_GENERATORS
    lr_schedule = create_learning_rate({"initial_learning_rate": initial_lr,
                                        "decay_steps": decay_steps},
                                       learning_rate_name=LearningRates.EXPONENTIAL_LR)
    tag = datetime.now().strftime("%Y-%m-%d_%H-%M")

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # tag = ""

    # dummy = Path('dummy.csv')
    # dataset = pd.DataFrame()

    # dummy_dataset = pd.read_csv(dataset_filepath)
    # labels = pd.unique(dummy_dataset["label"]).tolist()

    # for label in labels:
    #     df_label = dummy_dataset[dummy_dataset["label"] == label]
    #     for part in ["test", "train", "validation"]:
    #         total_rows = len(df_label[df_label["partition"] == part])
    #         debug(f"TOTOAL: {total_rows}, part: {part}, label:{label}")
    #         label_rows = df_label[df_label["partition"] == part]\
    #             .head(total_rows // 10)
    #         dataset = pd.concat([dataset, label_rows])

    # dataset.to_csv(dummy)

    # region pipeline definition
    hl = HammingLoss(threshold=hamming_loss_th)

    pipeline = Pipeline(results_path,
                        name=f"LSTM_{tag}")\
        .add_process(name="build_trainer_config",
                     action=TFKerasTrainerConfig.create,
                     args={
                         "config": {
                             "strategy": None,
                             "callbacks": {
                                 CHECKPOINT_CALLBACK: {
                                     "filepath": os.path.join(results_path, "checkpoints"),
                                     "save_best_only": True
                                 },
                                 TENSORBOARD_CALLBACK: {
                                     "log_dir": os.path.join(results_path, "tb_logs")
                                 }
                             }
                         }
                     })\
        .add_process(name="create_env_classifier",
                     action=EnvironmentClassifier.create,
                     args={
                         "model_config": {
                             "ENV_CLASSIFIER": {
                                 "layers": {
                                     "input": layers["input"],
                                     "embedding": layers["embedding"],
                                     "lstm_1": layers["lstm1"],
                                     "lstm_2": layers["lstm2"],
                                     "dense_1": layers["dense_1"],
                                     "dense_2": layers["dense_2"]
                                 }
                             }}
                     })\
        .add_process(name="dataset_loading",
                     action=Dataset.from_csv,
                     reportable=True,
                     args={
                         "source_path": dataset_filepath
                     })\
        .add_process(name="transform_to_datagen",
                     action=Transform.as_data_generator,
                     inputs_from_processes=["dataset_loading"],
                     args={
                         "vocab": vocab_filepath,
                         "shuffle": True,
                         "categorical_labels": True,
                         "transform_args": {
                             "pad_length": layers["input"]["T"],
                             "unk_token": Tokens.UNK_TOKEN
                         }
                     })\
        .add_process(name="train_classifier",
                     action=TFKerasTrainer.train,
                     reportable=True,
                     inputs_from_processes=["transform_to_datagen",
                                            "create_env_classifier",
                                            "build_trainer_config"],
                     args={
                         "train_config": {
                             "ENV_CLASSIFIER": {
                                 "representation": TRAIN_REPRESENTATION,
                                 'optimizer': tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                                 'loss': tf.keras.losses.BinaryCrossentropy(),
                                 "batch_size": batch_size,
                                 "epochs": epochs,
                                 "metrics": [hl, 'accuracy']
                             }}
                     })\
        .add_process(name="predict_classifier",
                     action=EnvironmentClassifier.predict,
                     inputs_from_processes=["train_classifier",
                                            "transform_to_datagen"],
                     reportable=True,
                     args={
                         "execution_config": None,
                         "prediction_config": {
                             "sparse_predictions": True
                         }
                     })

    # # endregion

    pipeline.run(report_pipeline=False)
    shutil.copy(config_file, Path(pipeline.path) / "config.json")
    shutil.copy("code/pipeline.py", Path(pipeline.path) / "pipeline.py")
    shutil.copy("code/model.py", Path(pipeline.path) / "model.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("-f", "--config_file",
    #                     help="Config file for the queries. See TODO:")
    parser.add_argument("-d", "--debug",  action='store_true',
                        help="Enables debug mode to trace the progress of your searching")
    parser.add_argument("-b", "--breakpoint",  action='store_true',
                        help="Enables breakpoint inside the debugger wrapper")
    parser.add_argument("-c", "--config",
                        help="Config file")

    ARGS = parser.parse_args()
    debugger.create(ARGS.debug, ARGS.breakpoint)

    run(ARGS.config)
