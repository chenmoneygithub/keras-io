"""
Title: Evaluate Your Model with GLUE Benchmark
Author: Chen Qian
Date created: 2023/01/01
Last modified: 2023/01/04
Description: Evaluate Your Model with GLUE Benchmark.
Accelerator: GPU
"""
"""
## What's This Guide

This guide shows how to evaluate your model using GLUE benchmark. This guide
covers the following topics:
- Overview of GLUE benchmark.
- Preprocessing GLUE dataset to unify the data format.
- Making a model that works for all glue tasks.
- Basic setup of the training (finetuning if you load a pretrained model) workflow.
- Generate submission files and submit it to GLUE leaderboard.
"""

"""
## Why Do I Write This Guide

I was trying to evaluate my model with GLUE benchmark, but surprisingly I found that
despite the popularity of GLUE benchmark, there is no handy tool/tutorial that shows me
how that can be achieved. One big question I have is - can I have a unified script that
"just works" for all GLUE tasks? A followup is - can I generate the GLUE leaderboard
submission file with the same script? Finally I wrote this script, and put it in KerasNLP
github repo, please check it out at
[https://github.com/keras-team/keras-nlp/tree/master/examples/glue_benchmark](https://gith
ub.com/keras-team/keras-nlp/tree/master/examples/glue_benchmark). You can plug in any
custom model into the script following the guidance, and it's fully compatible with GPU
and TPU.

While a runnable script is good, it cannot cover enough details without massive, tedious
and unreadable comments. That's the reason for me to write this post.
"""

"""
## Overview of GLUE Benchmark


"""

"""
[GLUE benchmark](https://gluebenchmark.com/) is commonly used to test a model's
performance at text understanding. It consists of 10 tasks:

1. [CoLA](https://nyu-mll.github.io/CoLA/) (Corpus of Linguistic Acceptability): Predict
if the sentence is grammatically correct.

1. [SST-2](https://nlp.stanford.edu/sentiment/index.html) (Stanford Sentiment Treebank):
Predict the sentiment of a given sentence.

1. [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398) (Microsoft
Research Paraphrase Corpus): Predict whether a pair of sentences are semantically
equivalent.

1. [QQP](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) (Quora
Question Pairs2): Predict whether a pair of questions are semantically equivalent.

1. [MNLI](http://www.nyu.edu/projects/bowman/multinli/) (Multi-Genre Natural Language
Inference): Predict if the premise entails the hypothesis (entailment), contradicts the
hypothesis (contradiction), or neither (neutral).

1. [QNLI](https://rajpurkar.github.io/SQuAD-explorer/)(Question-answering Natural
Language Inference): Predict if the context sentence contains the answer to the question.

1. [RTE](https://aclweb.org/aclwiki/Recognizing_Textual_Entailment)(Recognizing Textual
Entailment): Predict if a sentence entails a given hypothesis or not.

1. [WNLI](https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html)(Winograd
Natural Language Inference): Predict if the sentence with the pronoun substituted is
entailed by the original sentence.

1. [AX](https://gluebenchmark.com/diagnostics)(Diagnostics Main): Evaluate sentence
understanding through Natural Language Inference (NLI) problems.

1. [STSB](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark)(Semantic Textual
Similarity Benchmark): Predict the similarity score between 2 sentences.

Each task has a dataset split as train, validation and testing, with the exception that
MNLI and AX share the same training set.

***All except "STSB" can be viewed as a text classification task, while "STSB" is a
text regression task (output a float number in range [0, 5]).***

The common approach to use GLUE benchmark is to build your model and train/finetune it on
the training set, and evaluate locally with the validation set. Once you are satisfied
with the training/validation results, generate predictions on the testing set, and write
your predictions to `*.tsv` file (e.g., mrpc.tsv) with the required format. Then you
submit a zip file with all `.tsv` files to GLUE leaderboard, then the web will tell you
the actual performance on testing dataset. You cannot evaluate on testing dataset locally
because the testing label is not publicized.
"""

"""
## Install/Import Dependencies
"""

"""shell
!pip install -q keras-nlp
"""

import os
import csv

import keras_nlp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras

"""
## Get GLUE Dataset and Preprocess Data
This section discusses about how to download GLUE dataset in a Tensorflow runtime, and
preprocess the dataset so that it's ready for training.

Let's first define the task we will evaluate. In this guide I am using `mrpc` as a
showcase, but you can change it to any glue task.
"""

task_name = "mrpc"

"""
Then we define a few hyperparameters necessary for data preprocessing.
"""

batch_size = 32
sequence_length = 512

"""
### Download GLUE Dataset

We download GLUE dataset from Tensorflow Datasets (TFDS). The nice thing is the
downloaded dataset is already of type `tf.data.Dataset`, which has good support for
parallelism, accelerator optmization and etc. Relative materials can be found
[here](https://www.tensorflow.org/datasets/overview).

***AX, MNLI_Matched and MNLI_Mismatched*** are special because they share the same
training dataset, while they all have its own testing dataset. MNLI_Matched and
MNLI_Mismatched also have their own validation dataset, while AX provides no validation
dataset.
"""

if task_name in ("ax", "mnli_matched", "mnli_mismatched"):
    train_ds, validation_ds = tfds.load(
        "glue/mnli",
        split=["train", "validation_matched"],
    )
    if task_name == "ax":
        test_ds = tfds.load(
            "glue/ax",
            split="test",
        )

    if task_name == "mnli_matched":
        test_ds = tfds.load(
            "glue/mnli_matched",
            split="test",
        )

    if task_name == "mnli_mismatched":
        validation_ds, test_ds = tfds.load(
            "glue/ax",
            split=[
                "validation",
                "test",
            ],
        )
else:
    train_ds, test_ds, validation_ds = tfds.load(
        f"glue/{task_name}",
        split=["train", "test", "validation"],
    )

"""
### Save The Testing Data Index Order

This is required for generating leaderboard submission file. You can skip reading this
code for now.
"""

idx_order = test_ds.map(lambda data: data["idx"])

"""
### Data Unification

GLUE datasets come in dictionary format, and each task has its own feature name, such as
"sentence1" and "premise". This data format discrepancy creates complexity on our
training, so we standardize the format to simplify the training.

For all tasks, after standardization, each data record will have the following format:
`(features, label)`, while `features` is a tuple of one element if the task has only one
feature, e.g., "COLA", and is a tuple of 2 elements if the tasks has 2 featyres, e.g.,
"MRPC". There are no GLUE tasks having >2 features.
"""

"""
Get the feature names for our selected task.
"""

FEATURES = {
    "cola": ("sentence",),
    "sst2": ("sentence",),
    "mrpc": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "mnli_matched": ("premise", "hypothesis"),
    "mnli_mismatched": ("premise", "hypothesis"),
    "ax": ("premise", "hypothesis"),
    "qnli": ("question", "question"),
    "qqp": ("question1", "question2"),
}

feature_names = FEATURES[task_name]

"""
Define the function doing the standardization - convert the dictionary into (features,
label) format. Then use `map` function to apply the standardization.
"""


def split_features(x):
    features = tuple([x[name] for name in feature_names])
    label = x["label"]
    return (features, label)


train_ds = train_ds.map(split_features, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(split_features, num_parallel_calls=tf.data.AUTOTUNE)
validation_ds = validation_ds.map(split_features, num_parallel_calls=tf.data.AUTOTUNE)

"""
### Tokenization and Packing

NLP models cannot directly work on text input, we need to convert the text input to float
vectors. Here we use `keras_nlp.models.BertTokenizer` to do the conversion.

Remember our `feature` can be a tuple of 2 strings, we need some way to combine them
together. A common approach is to have a `[SEP]` token between two sentences, and put two
special token separately at the beginning and end of the combined sentence. For a unified
workflow, if the feature has only one string, we skip the `[SEP]` token, but still pad
the start and end token. This can be easily approached by
`keras_nlp.layers.MultiSegmentPacker`, as shown by the code below.
"""

tokenizer = keras_nlp.models.BertTokenizer.from_preset("bert_base_en_uncased")

packer = keras_nlp.layers.MultiSegmentPacker(
    start_value=tokenizer.cls_token_id,
    end_value=tokenizer.sep_token_id,
    pad_value=tokenizer.pad_token_id,
    sequence_length=sequence_length,
)


def preprocess_fn(feature, label):
    tokenized_data = [tokenizer(x) for x in feature]
    token_ids, _ = packer(tokenized_data)
    padding_mask = token_ids != tokenizer.pad_token_id
    return {"token_ids": token_ids, "padding_mask": padding_mask}, label


"""
After applying the `preprocess_fn`, for all GLUE tasks, each data record is a tuple
`(features, label)`, and `features` is a dictionary of format
```
{
    "token_ids": a tf.Tensor representing the token ids.
    "padding_mask": a tf.Tensor representing the mask (0 means the position is masked).
}

```
"""

train_ds_processed = (
    train_ds.map(preprocess_fn).batch(batch_size).prefetch(tf.data.AUTOTUNE)
)
validation_ds_processed = (
    validation_ds.map(preprocess_fn).batch(batch_size).prefetch(tf.data.AUTOTUNE)
)
test_ds_processed = (
    test_ds.map(preprocess_fn).batch(batch_size).prefetch(tf.data.AUTOTUNE)
)

"""
## Define The Model And Set Up Training
"""

"""
Let's define some hyperparameters for our model.
"""

if task_name == "stsb":
    num_classes = 1
elif task_name in (
    "mnli",
    "mnli_mismatched",
    "mnli_matched",
    "ax",
):
    num_classes = 3
else:
    num_classes = 2

feature_dim = 128
transformer_intermediate_dim = 128
vocab_size = tokenizer.vocabulary_size()
learning_rate = 5e-5
num_epochs = 6

"""
Then we define the classification model, it's a simple Transformer encoder with one
transformer layer and one dense layer. We can build this model with a few lines with
KerasNLP offerings.

We build the model using Keras functional API, at a high level it's to define a symbolic
input, and define your graph to compute the symbolic output, then tell a `keras.Model`
the input and output information. For more details, please refer to [this
guide](https://keras.io/guides/functional_api/).
"""

token_id_input = keras.Input(shape=(None,), dtype="int32", name="token_ids")
padding_mask = keras.Input(shape=(None,), dtype="int32", name="padding_mask")
x = keras.layers.Embedding(tokenizer.vocabulary_size(), feature_dim)(token_id_input)
x = keras_nlp.layers.TransformerEncoder(
    transformer_intermediate_dim, 4, activation="tanh"
)(x, padding_mask=padding_mask)[:, 0, :]
x = keras.layers.Dense(num_classes, activation="tanh")(x)

inputs = {
    "token_ids": token_id_input,
    "padding_mask": padding_mask,
}
outputs = x
classification_model = keras.Model(inputs=inputs, outputs=outputs)

"""
Define loss function and metrics to track training.
"""

if task_name == "stsb":
    loss = keras.losses.MeanSquaredError()
    metrics = [keras.metrics.MeanSquaredError()]
else:
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [keras.metrics.SparseCategoricalAccuracy()]

classification_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=loss,
    metrics=metrics,
)

classification_model.fit(
    train_ds_processed,
    validation_data=train_ds_processed,
    epochs=num_epochs,
)

"""
## Generate GLUE Leaderboard Submission Files

Now we have a trained model! Let's use it to evaluate the testing dataset, and generate
the leaderboard submission file.

First we define the corresponding file name and label names for each task.
"""

filenames = {
    "cola": "CoLA.tsv",
    "sst2": "SST-2.tsv",
    "mrpc": "MRPC.tsv",
    "qqp": "QQP.tsv",
    "stsb": "STS-B.tsv",
    "mnli_matched": "MNLI-m.tsv",
    "mnli_mismatched": "MNLI-mm.tsv",
    "qnli": "QNLI.tsv",
    "rte": "RTE.tsv",
    "wnli": "WNLI.tsv",
    "ax": "AX.tsv",
}

labelnames = {
    "mnli_matched": ["entailment", "neutral", "contradiction"],
    "mnli_mismatched": ["entailment", "neutral", "contradiction"],
    "ax": ["entailment", "neutral", "contradiction"],
    "qnli": ["entailment", "not_entailment"],
    "rte": ["entailment", "not_entailment"],
}


"""
Create an empty file now, we will fill the content soon.
"""

submission_directory = "glue_submissions"
if not os.path.exists(submission_directory):
    os.makedirs(submission_directory)
filename = submission_directory + "/" + filenames[task_name]
labelname = labelnames.get(task_name)

"""
Use our model to generate the predictions, then we map the prediction to the right index
order. We previously created `idx_order`, now it's coming to the stage!
"""

predictions = classification_model.predict(test_ds_processed)
if task_name == "stsb":
    predictions = np.squeeze(predictions)
else:
    predictions = np.argmax(predictions, -1)

# Map the predictions to the right index order.
idx_order = list(idx_order.as_numpy_iterator())
contents = ["" for _ in idx_order]

"""
The last step is to do the right formatting. Some tasks have label in integers, while
some tasks have its special string labels such as "entailment" and "not_entailment" in
QNLI. We also write the required headline.
"""

for idx, pred in zip(idx_order, predictions):
    if labelname:
        pred_value = labelname[int(pred)]
    else:
        pred_value = pred
        if task_name == "stsb":
            pred_value = min(pred_value, 5)
            pred_value = max(pred_value, 0)
            pred_value = f"{pred_value:.3f}"
    contents[idx] = pred_value

with tf.io.gfile.GFile(filename, "w") as f:
    # GLUE requires a format of index + tab + prediction.
    writer = csv.writer(f, delimiter="\t")
    # Write the required headline for GLUE.
    writer.writerow(["index", "prediction"])

    for idx, value in enumerate(contents):
        writer.writerow([idx, value])

"""
Assume you have `task_name=mrpc`, now you can check its content with the command below.
"""

"""shell
!head -10 glue_submissions/MRPC.tsv
"""

"""
For a real submission, you have to make a zip file including all tasks. If you just want
to evaluate on a single task on testing dataset, you can download the sample submission,
and replace the corresponding submission file.
"""

"""shell
!curl -O https://gluebenchmark.com/assets/CBOW.zip
!unzip -d sample_submissions/ CBOW.zip
!cp glue_submissions/MRPC.tsv sample_submissions/
!zip -r submission.zip . -i sample_submissions/*.tsv
"""

"""
You can download the generated `submission.zip` file to your local disk, and submit it
via the [official portal](https://gluebenchmark.com/submit). The score will be available
~30s after submission.
"""

"""
Congrats!! You have reached the end of the guide, hope you now have a good understanding
of GLUE benchmark and how to use it. Again if you are looking for something just working,
please check out the [GLUE
script](https://github.com/keras-team/keras-nlp/tree/master/examples/glue_benchmark)
available in KerasNLP.
"""
