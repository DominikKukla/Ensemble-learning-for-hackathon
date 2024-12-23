# Ensemble Fine-Tuned CNN Models for Polish Christmas Dishes Classification

## Description

This repository demonstrates the use of ensemble learning with fine-tuned CNN models for the classification of traditional Polish Christmas dishes.

The dataset consists of images from the following categories:

- Mushroom Soup (*Zupa Grzybowa*)
- Cheesecake (*Sernik*)
- Dumplings (*Pierogi*)
- Gingerbread (*Pierniki*)
- Poppy Seed Cake (*Makowiec*)
- Kutia (*Kutia*)
- Hunter’s Stew (*Bigos*)
- Beetroot Soup (*Barszcz*)

The models used in this project are:

- efficientnet_b0
- efficientnet_b1
- mobilenet_v3_large
- shufflenet_v2_x2_0

Each model has been fine-tuned to adapt to the nuances of the dataset. An ensemble strategy is applied to improve classification accuracy by leveraging the strengths of each individual model. The models are combined using **weighted voting**, where each model's prediction is weighted based on its performance during validation.

The dataset was preprocessed and augmented with a [transform pipeline](/src/utils/dataset.py). Examples of transformations include resizing, rotation, and color adjustments, as shown below:

<img src="data/images/transform_pierogi.png" width="700" />
<img src="data/images/transform_bigos.png" width="700" />

<br>

## Setup and Configuration

To clone the repository, use the command:
```bash
git clone https://github.com/DzmitryPihulski/ensemble-finetuned-models.git
