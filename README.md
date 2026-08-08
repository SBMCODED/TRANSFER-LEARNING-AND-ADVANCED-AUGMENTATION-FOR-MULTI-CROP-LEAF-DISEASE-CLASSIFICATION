## Augmentation vs. Imbalance Handling in Multi-Crop Plant Disease Classification

A Major Research Project (M.Sc. Data Science and Analytics, Toronto Metropolitan University) that isolates the effect of data augmentation from the choice of class-imbalance handling strategy on CNN transfer-learning models for multi-crop plant disease classification.

Author: Oladele Yusuf Olaide (Student ID: 501372558) Supervisor: Prof. Mucahit Cevik
## Overview

Real-world plant disease datasets are almost always class-imbalanced, and the two standard remedies for this — data augmentation and class-imbalance handling — are usually evaluated together in prior work, so their individual effects can't be separated. This project fixes that with a controlled, non-confounded experiment: four CNN backbones are each trained under four conditions that cross augmentation (on/off) with imbalance strategy (class weighting vs. oversampling), under an identical training regime throughout, producing sixteen independently trained and evaluated models.

A custom weighted loss function was also built to resolve a real conflict between MixUp/CutMix augmentation (which produces soft, blended labels) and standard class weighting (which expects one hard label per sample) — something that had previously made that specific combination infeasible.
## Dataset
Source: [CCMT (Crop Pest and Disease Detection) dataset](https://data.mendeley.com/datasets/bwh3zbpkpv/1) , hosted on Mendeley Data
Crops: Maize, Cassava, Cashew, Tomato
Classes: 22 disease and healthy categories
Size: 25,170 validated images (after PIL-based corrupt/truncated file filtering)
Split: 80% / 10% / 10% train/validation/test, stratified by class, fixed seed (42)
The dataset is naturally imbalanced (~13:1 between the largest and smallest classes), which is the central challenge this project addresses.

## Methodology

Backbone --->	Role
VGG16	--->Baseline reference
MobileNetV2	---->Lightweight / efficient
EfficientNetB0--->	Efficient / compound-scaled
Hybrid (ResNet50 + EfficientNetB0)---> Dual-backbone feature fusion
