# Automatic Speech Recognition Using Pytorch 

## Table of contents

- [General info](#general-info)
- [Repository Description](#repository-description)
- [Requirement](#Requirement)
- [References](#References)
- [Notes](#Notes)

## General info

In this project, I implemented an end-to-end automatic speech recognition (ASR) model for the recognition of digits. The acoustic model is based on Wav2Vec 2.0 and was fine-tuned on a given dataset. We report the WER, and CER on the train and validation set for three different decoder types (CTC decoder with language model, without language model, and greedy decoder). Finally, we predict on the test set.

## Repository Description

| Filename                    | description                                                                                       |
| --------------------------- | ------------------------------------------------------------------------------------------------- |
| `dataset.py` | The custom dataset implementation python file |
| `AcousticModel.py`    | The training of the acoustic model                                                                   |
| `decoding.py`          | The decoding part (report WER, CER, in addition to the prediction on the test set)                                                         |
| `CreateLM.py`                | The code for creating KenLM language model                                 |
| `requirement.txt`           | File containing all the packages we used in this project                                          |
| `lang_model.arpa`           | The KenLM language model                                          |
| `train_transcription`           | The text file we creating the language model on                                         |
| `lexicon.txt`           | The given lexicon of the dataset                                       |
| `report.pdf`           | The report of the project                                          |


## Requirement

To run this project, you need to install several packages. For convenience, we created a `requirement.txt` file consists of all the packages used in this projcet.

In order to install all the packages in the `requirement.txt` file, simply use to command `pip install -r requirements.txt`.

## Dataset
The dataset can be found in the [link](https://drive.google.com/drive/folders/1SogBbv-odBmNC4vXP9tC1uEtY4BbQZsS?usp=sharing). *The TIDIGITS* dataset where a sequence of up to 7 digits is pronounced in each recording.


## Notes
In order to run the code look at the `report.pdf` file.
