# MAML-CNN-Text-Classifier
Model-agnostic meta-learning framework adapted to CNN text classifier
## Model Overview
This repository contains an implementation of [Model-Agnostic Meta-Learning Algorithm](https://arxiv.org/abs/1703.03400) on a [CNN Text Classifier](https://www.aclweb.org/anthology/D14-1181/). Each sub-task is defined to be a binary classification task.
## Dependencies
- python 3.x
- tensorflow == 1.14.0
- numpy == 1.17.3
- nltk == 3.4.5
## Methodology
### MAML
Meta-learning is aimed at enabling a model to quickly adapt to a new task with few data samples. Specifically, MAML attempts to learn a better initialization for each subtask, such that it can quickly fit into the new data in a few training epochs. 

![maml diagram](https://github.com/Higgsboson-X/maml-cnn-text-classifier/blob/master/images/maml_fig.png "MAML Diagram")

A model consists of a meta-learning and a set of sub-learners.
1. In the meta-learning stage, each update for the meta-learner requires few updates in the sub-task learners. Specifically, for all sub-tasks in this model, each sub-learner starts from the same set of parameters as the meta-learner, and updates its parameters by gradient descent using the samples drawn from the task-specific data. Each sub-task update contains a support step, which is used to update the sub-learner's parameters, and a query step, which is performed on a task-specific test set. The update for each sub-task is performed for a predefined number of steps. The loss from the last query step is used to calculate the proposed direction by this sub-task using gradient descent. Note that this gradient is calculated with respect to the initial meta-learner's parameters. The final direction that meta-learner takes is an average among all directions proposed by all sub-tasks.
2. In the fine-tuning stage, the model initializes from the pretrained meta-leaner's parameters and is trained using a task-specific small set of data. Here, the dataset used in the meta-learning stage and the fine-tuning stage are the same for each task.

The MAML algorithm presented in the original paper is shown in the following figure.

![maml algorithm](https://github.com/Higgsboson-X/maml-cnn-text-classifier/blob/master/images/maml_alg.png "MAML Algorithm")

### CNN Classifier
The CNN text classifier is based on the following architecture (as is proposed in the original paper).

![cnn arch](https://github.com/Higgsboson-X/maml-cnn-text-classifier/blob/master/images/cnn_arch.png "CNN Architecture")

## Folder Structure
```
maml-cnn-text-classifier
+-- ckpt (save model checkpoints)
|    +-- translations (sample dataset)
+-- code (all source codes)
|    +-- main.py
|    +-- arguments.py
|    +-- config
|          +-- model_config.py
|    +-- models
|          +-- maml_cnn_classifier.py
|    +-- utils
|          +-- vocab.py
+-- data
|    +-- translations (sample data, corpus name in arguments)
|          +-- tasks (data in the correct format for each task)
|          +-- processed (pickle files for storing processed data)
+-- images
+-- vocab
|    +-- translations (directory for storing vocabularies)
```
All the data should be in the correct format, see source code for details.

## Sample Corpus
The sample corpus is from a set of collected translations for different translators. Most literature works are gathered from [original literature translations](http://gen.lib.rus.ec/). The full dataset is not contained in this repository.

|Index|Translator | Total Data Size |
|:---:|:---------:|:---------------:|
|1|Alban Kraisheimer|15604|
|2|Andrew R. MacAndrew|39454|
|3|David Hawkes|62924|
|4|H. T. Lowe-Porter|34471|
|5|Ian C. Johnston|25076|
|6|Isabel Florence Hapgood|44507|
|7|John E. Woods|35660|
|8|Julie Rose|35161|
|9|Michael R. Katz|22123|
|10|Richard Pevear|121217|
|11|Robert Fagles|11911|
|12|Yang Xianyi|62802|

The data sizes for each task is presented in the following table.

|Task ID|Support Data Size|Query Data Size|Index Pair|
|:-----:|:---------------:|:-------------:|:--------:|
|1|18949|1889|1 & 6|
|2|19488|1951|2 & 10|
|3|19113|1894|3 & 12|
|4|19544|1948|4 & 7|
|5|19314|1937|5 & 11|
|6|18887|1883|6 & 8|
|7|19710|1973|9 & 10|

The default size is 18000 for support data and 1800 for query data.

## Run
### Prepare Data
Include all the text and label files in the `data/[corpus]/tasks/` directory, including a text containing all the sentences in the file for preparing the vocabulary. Name the files as `t[id]_text.[type]`, `t[id]_label.[type]`, where `[id]` is the index for each task, and `[type]` is the type of operation (train, val, test, infer).
### Options
```
MAML-CNN-Text-Classifier

optional arguments:
  -h, --help            show this help message and exit
  --max-seq-length MAX_SEQ_LENGTH
                        maximum sequence length for sentences
  --vocab-cutoff VOCAB_CUTOFF
                        maximum word count to ignore in vocabulary
  --filter-stopwords    whether to filter stopwords
  --embedding-size EMBEDDING_SIZE
                        dimension of sentence embedding
  --num-labels NUM_LABELS
                        total number of labels per task
  --support-data-size-per-task SUPPORT_DATA_SIZE_PER_TASK
                        support data size for each task
  --query-data-size-per-task QUERY_DATA_SIZE_PER_TASK
                        query data size for each task
  --num-filters NUM_FILTERS
                        number of filters for each size
  --l2-reg-weight L2_REG_WEIGHT
                        l2 regularization weight for output kernel and bias
  --dropout-keep-prob DROPOUT_KEEP_PROB
                        keep probability of dropout layer, only for training
  --meta-lr META_LR     learning rate for meta learner in meta-training
  --train-lr TRAIN_LR   learning rate for fine-tuning
  --sub-lr SUB_LR       learning rate for sub-tasks in meta-training
  --corpus CORPUS       name of corpus, for defining directions and paths
  --maml-epochs MAML_EPOCHS
                        meta-training epochs, 0 if normal training scheme
  --train-epochs TRAIN_EPOCHS
                        fine-tuning epochs
  --maml-num-batches MAML_NUM_BATCHES
                        number of batches for maml-training
  --train-num-batches TRAIN_NUM_BATCHES
                        number of batches for fine-tuning
  --num-tasks NUM_TASKS
                        total number of sub-tasks
  --num-updates NUM_UPDATES
                        number of updates for each sub-task
  --train-task-id TRAIN_TASK_ID
                        task id for fine-tuning, 0 for all tasks available
  --test-task-id TEST_TASK_ID
                        task id for test after meta-training, -1 for none, 0
                        for all
  --infer-task-id INFER_TASK_ID
                        task id for inference after meta-training, -1 for
                        none, 0 for all
  --epochs-per-eval EPOCHS_PER_EVAL
                        number of epochs per evaluation in fine-tuning
  --infer               whether to perform an inference after fine-tuning
  --test                whether to perform a test after fine-tuning
  --restore-ckpt-path RESTORE_CKPT_PATH
                        checkpoint path for saved model if restore from saved
```
Run the program with options and arguments, e.g.,
```
python3 main.py \
    --maml-epochs=10 \
    --train-epochs=5 \
    --support-data-size-per-task=18000 \
    --query-data-size-per-task=1800 \
    --train-task-id=1 \
    --infer-task-id=2 \
    --test-task-id=3 \
    --infer --test \
    --maml-num-batches=64 \
    --train-num-batches=64 \
```
The command above trains the meta-learner for 10 epochs with 64 batches, fine-tunes for task 1 with 5 epochs and 64 batches, performs inference for task 2, performs test for task 3, and performs both inference and test after fine-tuning.
## Results on the Sample Corpus
Results performed on the sample corpus are shown below. Performance might or might not get improved with more training epochs. Note that the classification is relatively more difficult comparing to other text classification tasks, since the writing "style" is a subtle feature.

|Task|Meta Epochs|Tune Epochs|Support Loss| Query Loss| Support Acc.| Query Acc.|
|:--:|:---------:|:---------:|:----------:|:---------:|:-----------:|:---------:|
|Meta|10|-|0.815|0.504|0.538|0.786|
|T1|10|5|0.056|1.1|0.992|0.746|
|T2|10|5|0.235|0.743|0.908|0.759|
|T3|10|5|0.165|0.066|0.949|0.765|
|T4|10|5|0.145|0.902|0.962|0.685|
|T5|10|5|0.058|0.351|0.976|0.894|
|T6|10|5|0.142|0.811|0.971|0.723|
|T7|10|5|0.218|0.73|0.925|0.752|
|T1|-|5|0.193|0.745|0.92|0.746|
|T2|-|5|0.187|0.737|0.938|0.744|
|T3|-|5|0.197|0.691|0.928|0.747|
|T4|-|5|0.234|0.995|0.914|0.643|
|T5|-|5|0.056|0.375|0.993|0.878|
|T6|-|5|0.174|0.82|0.937|0.711|
|T7|-|5|0.194|0.789|0.925|0.733|

## TODO
Higher level MAML APIs...
