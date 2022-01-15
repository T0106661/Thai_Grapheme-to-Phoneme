# Grapheme-to-Phoneme Conversion for Thai using Neural Regression Models

This repository contains the Keras implementation of our paper.
The implementation can reproduce the main result.

You shall obey the license.
We make the ownership of the resources clarified after the paper is accepted and officially published.

# How to use

## Setup

Install package as

```shell
pip install -r requirements.txt
```

## Prediction

To predict the similarity using the pre-trained model, run predict.py as

```shell
python predict.py --input test_tha.txt --ids tha.pkl --model pretrained_tha.h5 --output output.txt
```

The ids file should be the one when the model was trained
because this file contains the information of syllables and phonemes. 

The input file needs to be represented as

```csv
characters1 [tab] syllables1 [newline]
characters2 [tab] syllables2 [newline]
...
```

Syllables need to be separated by spaces.

## Training

To train a model, run train.py as

```shell
python train.py --input train1000.txt --ids ids.pkl --model model.h5
```

After training, the ids file and the model file are created.
The ids file contains the information of syllables and phonemes.

This script can be applied to languages other than Thai.
The input file needs to be represented as

```csv
characters1 [tab] syllables1 [tab] similarity1 [newline]
characters2 [tab] syllables2 [tab] similarity2 [newline]
...
```

Syllables need to be separated by spaces.
