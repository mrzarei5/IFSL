# Interpretable Few-shot Learning with Online Attribute Selection

This repository contains the code for ['Interpretable Few-shot Learning with Online Attribute Selection'](https://doi.org/10.1016/j.neucom.2024.128755).

## Requirements
### Installation
Create a conda environment and install dependencies:
```bash
conda create --name IFSL --file requirements.txt
conda activate IFSL
```

## Example Instructions

To train the framework on the CUB dataset, follow these steps:

### Prepare Data

1. Create the directory `./datasets/CUB/data`.
2. Download the dataset from [CUB Dataset](https://www.vision.caltech.edu/datasets/cub_200_2011/) and extract it into the created directory.
3. Run the following command to prepare the data:

```bash
python ./write_CUB_filelist.py --dataset=./datasets/CUB/data
```

## Training the Main Framework


### Train Attribute Predictor

Run the following command:

```bash
python ./att_predictor.py --dataset=CUB --dataset_dir=./datasets/CUB/data/CUB_filelist
```

### Train Selector Network

After the end of attribute predictor network training procedure, run the following command with appropriate parameters:

```bash
python ./att_selector.py --alpha=1 --gamma=0 --dataset=CUB --dataset_dir=./datasets/CUB/data/CUB_filelist --n_support=1 --n_query=16
```

## Experiment related to Automatically Balancing Accuracy and Interpretability

Before executing these commands, the attribute predictor and attribute selector networks should be trained using the previous steps.

### Train Unknown Attribute Learner

```bash
python ./att_learning_unknown.py --dataset=CUB --dataset_dir=./datasets/CUB/data/CUB_filelist --n_support=1
```
### Train Unknown Attributes Participation Detector

```bash
python ./unknown_participation_detector.py --alpha=1 --gamma=0 --beta=0.7 --dataset=CUB --dataset_dir=./datasets/CUB/data/CUB_filelist --n_support=1 --n_query=16
```

