Airbus Ships Detection Using U-Net Model

## Overview

This project implements a semantic U-Net model for detection of ships in satellite photos provided by Airbus company. The dataset consists of 192557
images with a shape of (768, 768, 3). The majority of photos (150000) does not contain ships whereas the remaining 42556 photos contains one ship or more (up to 15 ships in one image). 

## Dataset Details

##### Total images: 192557
##### Image shape: (768, 768, 3)
##### Images without ships: 150000
##### Images with ships: 42556


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Model Training](#model-training)
- [Inference](#inference)
- [Results](#results)


## Installation

1. Clone the repository:

    git clone https://github.com/NataliiaLimanska/airbus_ships_detection

2. Install the required dependencies:

	python3 -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt

## Usage
To train the model, data (train set and .csv file with RLE masks) should be loaded from the Kaggle competition page: https://www.kaggle.com/competitions/airbus-ship-detection/data


### Directory Structure

After downloading data, the directory structure should be like:

![image](https://github.com/NataliiaLimanska/airbus_ships_detection/assets/124721890/a1d6a59d-2d9f-4741-99e8-ecc08dbed96c)


## Model Training


Taking into account the heavily unbalanced dataset, the next strategy was implemented to increase the effectiveness of training for the segmentation model. A separate dataset was created which included all images with ships on them (42556) plus 0.5% of the images without ships randomly selected (750). Totally the working dataset represented: 43306 images.

Taking into account the impossibility of testing on the Kaggle test dataset because it was created for competition and did not contain the RLE-masks, an independent dataset was created from train images by excluding 10% of images from the train_v2 dataset. The remained train dataset was splitted on train and validation data.


To train the U-Net model, use the

python ships_model_training.py script


## Model Inference

Inference for the model was conducted on the test dataset described above, and the Dice coefficient was evaluated.

To perform inference on test images, use the

python ship_model_inference.py script


## Results


The predictions and Dice coefficients for each test image are saved in the predictions.csv file. The mean Dice coefficient for all test images is also printed out.
