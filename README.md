
# envs

# Overview

**envs** (environment classifier), is an experiment that workds with a model based in LSTMs, whose main objective is to classify abstracts gathered from the site [PLOS](https://plos.org/) into the following classes: freshwater, coastal, marine, terrestrial and others.

The complete experiment relies on the end-to-end library, [CONABIO_ML_TEXT](https://bitbucket.org/conabio_cmd/conabio_ml_text/src/master/), which also takes on the [CONABIO_ML](https://bitbucket.org/conabio_cmd/conabio_ml/src/master/).

**Note:** Both of the [CONABIO_ML](https://bitbucket.org/conabio_cmd/conabio_ml/src/master/) and [CONABIO_ML_TEXT](https://bitbucket.org/conabio_cmd/conabio_ml_text/src/master/) libraries are in early stages of development. So, the usage of **dev branches** is assumed.

## Workflow

The following resources are intended to be executed sequentially and represent the complete workflow.

1. **eda.ipynb**

    Contains the exploratory analysis of the abstracts obtained. Some statistics are drown to be used in further steps, like samples per class / partition.

    In this notebook, we also performed  the class contraction analysis.

2. **envs_preproc.ipynb**

    We rely on two methods of preprocessing: a simple preprocessing step and the BPE preprocessing algorithm.
    In this notebook, we perform both of the preprocessing methods and persist the resources related.

3. **pipeline.py/envs.ipynb**

    In this script/notebook, we actually perform the training stage.

    We use the [CONABIO_ML_TEXT](https://bitbucket.org/conabio_cmd/conabio_ml_text/src/master/) library to draw the inner steps of the experiment. Such as dataset loading, model instantiation, and model prediction.

## Other resources

The following scripts are needed in the pipeline/envs resources to work properly.

- **model.py**

    Contains the following resources:

    1. The model `EnvironmentClassifier`, based on [tfkeras 2.2+](https://www.tensorflow.org/api_docs/python/tf/keras) to be wrapped by a `conabio_ml_text.trainer.model`.
    2. The custom metrics [`HammingLoss`, `multilabel_topk`],to be measured in the training step.
    3. An alternative loss function `macro_soft_f1`.

- **preprocessing.py**

    The implementation of the BPE algorithm.

# Requirements

Python 3.7.7+

[CONABIO_ML_TEXT](https://bitbucket.org/conabio_cmd/conabio_ml_text/src/master/)

[docker](https://www.docker.com/)

# Configuration

## Set up the environment

The fastest (and the recommended) way to set up the environment is vía docker.

- **docker-compose**

    You can simply start your environment using the `docker-compose` file located in `code/docker-compose.yml` and build/start with:

    ```python
    docker-compose build
    docker-compose start
    ```

    This method relies on the image of the [CONABIO_ML_TEXT](https://bitbucket.org/conabio_cmd/conabio_ml_text/src/master/) library and only shares the directory `code` with the container.

    Note: This methods DOES NOT map the GPU with the container.

- **docker**

    If you want to take advantage of an environment with GPU ready, you can create the image manually and share the code folder with the following commands:

    ```python
    docker image build -t TAG:v0 conabio_ml_text/images/tf2
    docker run -it --gpus NUM_GPUS --name TAG -d -v host_path/envs:/lib/code_environment/code -p 9000:8888 -p 9001:6006 TAG:v0 bash
    ```

    Where:

    **host_path**: Path where the **envs** project is located

    **TAG**: Tag to recognize the container

    -p (option): Are the mapped ports of **jupyter notebook** and **tensorboard**.

## pipeline

Once you finished the **envs_preproc.ipynb** notebook execution, you will have the folder `code/config/XXX` with the following resources:

- config.json. Contains the basic configuration of the model. It has to be alike to the following:

```python
{
    "vocab": "configs/simple_proc_multilabel_50K_600/vocab.json",
    "dataset": "configs/simple_proc_multilabel_50K_600/dataset.csv",
    "layers": {
        "input": {
            "T": 600
        },
        "embedding": {
            "V": 50000,
            "D": 400
        },
        "lstm1": {
            "M": 128
        },
        "lstm2": {
            "M": 128
        },
        "dense": {
            "K": 5
        }
    },
    "params": {
        "initial_learning_rate": 0.004,
        "decay_steps": 600,
        "batch_size": 32,
        "epochs": 5,
        "multilabel_threshold": 0.3,
        "multilabel_classes": 3
    }
}
```

These parameters will be taken for the `EnvironmentClassifier.create_model` method under `layer_config` property. And for the `pipeline.run` method in the `config` property.

- vocab.json: After the preprocessing step finishes, the vocabulary to train the model will be persisted in this file.
- dataset.csv: Is the dataset to train the model

Assuming you already created the `config` file. To execute the training step, use:

```python
python code/pipeline.py -c code/configs/simple_proc_multilabel/config.js

Where:
-c file_path of the file produced by envs_preproc.ipynb
```

# Results

After finishing the `pipeline.py` script execution, the folder `pipelines/results/LSTM_XXX` will be created with the following resources/folders:

- dataset_loading: Corresponds to the process `name="dataset_loading"` of the pipeline and contains the dataset partitioned and some default statistics of the dataset.
- train_classifier: Corresponds to the process `name="train_classifier"` of the pipeline and contains the model architecture reported by Keras, as same as the train history.
- predict_classifier: Corresponds to the process `name="predict_classified"` of the pipeline and contains the dataset predicted with all the classes and its related score.
- tb_logs: The log of the training step to be displayed by **tensorboard**.
- checkpoints: The model checkpoints, for further loading.