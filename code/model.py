#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import string
import unidecode

import tensorflow as tf
from tensorflow.python.keras.metrics import MeanMetricWrapper
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, Input, Embedding, Bidirectional, GlobalMaxPooling1D, Dropout, LayerNormalization

# Remember to update the PYTHON_PATH to
# export PYTHONPATH=`pwd`:`pwd`/conabio_ml_text/conabio_ml:`pwd`/conabio_ml_text

from conabio_ml_text.preprocessing.preprocessing import Tokens
from conabio_ml_text.trainers.bcknds.tfkeras_models import TFKerasRawDataModel
from conabio_ml_text.trainers.bcknds.tfkeras import TFKerasBaseModel

from conabio_ml.utils.logger import get_logger, debugger
from conabio_ml_text.utils.constraints import TransformRepresentations, LearningRates, Optimizers

log = get_logger(__name__)
debug = debugger.debug

re_numbers = re.compile('^[-+]?[\d.]+(?:e-?\d+)?$')

# region preprocessing methods


def simple_preprocess(preproc_args: dict = {},
                      item: str = ""):
    tokens = []
    item = unidecode.unidecode(item)
    item = item.lower()
    item = re.sub(r'https?:\/\/.*[\r\n]*', '', item)

    item = item.translate(
        str.maketrans(string.punctuation, ' '*len(string.punctuation))
    )

    for token in item.split():
        if re.findall(re_numbers, token):
            tokens.append(Tokens.NUM_TOKEN)
            continue

        tokens.append(token)

    item = item.translate(
        str.maketrans(string.punctuation, ' '*len(string.punctuation))
    )

    return " ".join(tokens)

# endregion

# region custom eval metric


def multilabel_threshold(threshold: float):

    def metric(y, y_hat):
        y_pred = tf.cast(tf.greater(y_hat, threshold), tf.float32)
        tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
        fp = tf.cast(tf.math.count_nonzero(
            y_pred * (1 - y), axis=0), tf.float32)
        fn = tf.cast(tf.math.count_nonzero(
            (1 - y_pred) * y, axis=0), tf.float32)
        f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        macro_f1 = tf.reduce_mean(f1)

        return macro_f1

    return metric


def multilabel_topk(classes: int):
    k_classes = classes

    def metric(y, y_hat):
        return tf.keras.metrics.top_k_categorical_accuracy(y, y_hat, k_classes)
    return metric


def multilabel_converter(threshold: float):
    th = threshold

    def wrapper(y: np.array,
                y_hat: np.array,
                *args):

        res = (y_hat > th) * 1.
        return res
    return wrapper


def hamming_loss(y_true: np.array, y_pred: np.array, threshold):
    if threshold is None:
        threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
        # make sure [0, 0, 0] doesn't become [1, 1, 1]
        # Use abs(x) > eps, instead of x != 0 to check for zero
        y_pred = tf.logical_and(y_pred >= threshold,
                                tf.abs(y_pred) > 1e-12)

    else:
        y_pred = y_pred > threshold

    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.int32)

    nonzero = tf.cast(
        tf.math.count_nonzero(y_true - y_pred, axis=-1), tf.float32)
    return nonzero / y_true.get_shape()[-1]


class HammingLoss(MeanMetricWrapper):
    def __init__(self, name='hamming_loss', threshold=None, dtype=tf.float32):
        super(HammingLoss, self).__init__(
            hamming_loss, name, dtype=dtype, threshold=threshold)

# endregion

# region custom loss function


@tf.function
def macro_soft_f1(y, y_hat):
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - soft_f1
    macro_cost = tf.reduce_mean(cost)  # average on all labels
    return macro_cost
# endregion


# model
class EnvironmentClassifier(TFKerasRawDataModel):

    @classmethod
    def create_model(cls,
                     layer_config: dict) -> TFKerasBaseModel.TFKerasModelType:
        try:

            layers = layer_config["layers"]

            input_layer = layers["input"]
            embedding = layers["embedding"]

            dense_1 = layers["dense_1"]
            dense_2 = layers["dense_2"]

            lstm_1 = layers["lstm_1"]
            lstm_2 = layers["lstm_2"]

            # region Simple LSTM

            # i = Input(shape=(input_layer["T"], ))
            # x = Embedding(input_dim=embedding["V"],
            #              output_dim=embedding["D"],
            #              mask_zero=True)(i)

            # x = LSTM(units=lstm_1["M"],
            #          return_sequences=True
            #   )(x)
            # if lstm_2:
            #     x = LSTM(units=lstm_2["M"],
            #              return_sequences=True
            #   )(x)
            #     x = Dropout(lstm_2["dropout"])(x)
            #     x = LayerNormalization()(x)

            # x = GlobalMaxPooling1D()(x)
            # x = Dense(units=dense_2["K"],
            #           activation='sigmoid')(x)

            # endregion

            # region Bidirectional LSTM
            i = Input(shape=(input_layer["T"], ))
            x = Embedding(input_dim=embedding["V"],
                          output_dim=embedding["D"],
                          mask_zero=True)(i)

            x = Bidirectional(LSTM(units=lstm_1["M"],
                                #    return_sequences=True  # Just used when we have 2 LSTMS
                                   ))(x)
            x = Dropout(lstm_1["dropout"])(x)

            if lstm_2:
                x = LSTM(units=lstm_2["M"],
                         return_sequences=True
                         )(x)
                x = Dropout(lstm_2["dropout"])(x)

            x = Dense(dense_1["M"],
                      activation='relu')(x)
            x = Dropout(dense_1["dropout"])(x)
            x = Dense(units=dense_2["K"],
                      activation='sigmoid')(x)
            # endregion

            model = Model(i, x)

            return model
        except Exception as ex:
            log.exception(ex)
            raise

# endregion
