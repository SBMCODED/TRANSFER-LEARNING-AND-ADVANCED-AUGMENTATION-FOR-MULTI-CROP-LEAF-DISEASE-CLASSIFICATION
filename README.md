
# Augmentation vs. Imbalance Handling in Multi-Crop Plant Disease Classification

A Major Research Project (M.Sc. Data Science and Analytics, Toronto Metropolitan University) that isolates the effect of **data augmentation** from the choice of **class-imbalance handling strategy** on CNN transfer-learning models for multi-crop plant disease classification.

**Author:** Oladele Yusuf Olaide (Student ID: 501372558)
**Supervisor:** Prof. Mucahit Cevik

---

## Overview

Real-world plant disease datasets are almost always class-imbalanced, and the two standard remedies for this — **data augmentation** and **class-imbalance handling** — are usually evaluated together in prior work, so their individual effects can't be separated. This project fixes that with a controlled, non-confounded experiment: four CNN backbones are each trained under **four conditions that cross augmentation (on/off) with imbalance strategy (class weighting vs. oversampling)**, under an identical training regime throughout, producing sixteen independently trained and evaluated models.

A custom weighted loss function was also built to resolve a real conflict between MixUp/CutMix augmentation (which produces soft, blended labels) and standard class weighting (which expects one hard label per sample) — something that had previously made that specific combination infeasible.

## Dataset

- **Source:**https://data.mendeley.com/datasets/bwh3zbpkpv/1 , hosted on Mendeley Data
- **Crops:** Maize, Cassava, Cashew, Tomato
- **Classes:** 22 disease and healthy categories
- **Size:** 25,170 validated images (after PIL-based corrupt/truncated file filtering)
- **Split:** 80% / 10% / 10% train/validation/test, stratified by class, fixed seed (42)

The dataset is naturally imbalanced (~13:1 between the largest and smallest classes), which is the central challenge this project addresses.

## Methodology

### Backbones (transfer learning, ImageNet-pretrained, frozen)
| Backbone | Role |
|---|---|
| VGG16 | Baseline reference |
| MobileNetV2 | Lightweight / efficient |
| EfficientNetB0 | Efficient / compound-scaled |
| Hybrid (ResNet50 + EfficientNetB0) | Dual-backbone feature fusion |

Every backbone shares the same classification head: `GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.4) → Dense(22, softmax)`.

### The four experimental arms
| Arm | Augmentation | Imbalance strategy |
|---|---|---|
| **A** | None | Class weighting |
| **B** | None | Oversampling |
| **C / C+** | Geometric, or full Cutout + MixUp + CutMix | Oversampling |
| **D / D+** | Geometric, or full Cutout + MixUp + CutMix | Class weighting (via custom weighted loss) |

Every model in every arm is trained under the same single-stage, frozen-backbone regime (Adam optimizer, up to 30 epochs, EarlyStopping + ReduceLROnPlateau) — this is what makes the augmentation/imbalance-strategy comparison valid.

## Key Results

| Backbone | Arm A | Arm B | Arm C+ | Arm D+ |
|---|---|---|---|---|
| VGG16 | 83.8% | 84.9% | 81.5% | 78.6% |
| MobileNetV2 | 67.5% | 61.6% | 57.1% | 56.4% |
| EfficientNetB0 | 80.2% | 77.1% | 75.0% | 73.9% |
| **Hybrid (ResNet50+EfficientNetB0)** | 87.5% | 86.2% | **88.5%** | 86.7% |

- **Best single result:** Hybrid backbone + Arm C+ (augmentation + oversampling) — **88.52%**
- **Best method on average across all backbones:** Arm A (no augmentation + class weighting) — **79.75%** mean accuracy
- Strong augmentation *reduced* accuracy for every backbone **except** the hybrid model
- MobileNetV2 was the weakest backbone in every condition
- `tomato_verticulium_wilt` was the hardest class across all 16 models, most often confused with other tomato diseases

See the full write-up (Chapters 3–5) for detailed per-model classification reports, confusion matrices, training curves, and ROC/AUC analysis.

## Repository Structure

```
├── ano.py                                       # Data loading, validation, EDA
├── plant_disease_detection.ipynb                # Full experiment notebook (all 16 runs)
├── arm_A_*_no_aug_class_weight.py               # Arm A: no augmentation + class weighting
├── arm_B_*_no_aug_oversampling.py               # Arm B: no augmentation + oversampling
├── arm_C_*_aug_oversampling.py                  # Arm C/C+: augmentation + oversampling
├── arm_D_*_aug_class_weight.py                  # Arm D/D+: augmentation + class weighting
│   (one script per backbone: vgg16, mobilenetv2, efficientnetb0, hybrid_resnet50_effnetb0)
├── docs/
│   ├── Chapter_Three_Methodology.docx
│   ├── Chapter_Four_Results.docx
│   ├── Chapter_Five_Conclusion.docx
│   ├── Abstract.docx
│   └── Acknowledgement.docx
└── README.md
```

## Getting Started

### Requirements
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn pillow
```

### Data setup
1. Download the CCMT dataset and place the four crop folders (`maize/`, `cassava/`, `cashew/`, `tomato/`) under a local `data/` directory.
2. Run `ano.py` to validate images and build the unified dataframe.

### Running an experiment
Each arm/backbone combination is a standalone script:
```bash
python arm_A_vgg16_no_aug_class_weight.py
python arm_C_hybrid_resnet50_effnetb0_aug_oversampling.py
# ...etc, one per backbone × arm combination
```
Each script trains the model, evaluates it on the held-out test set, and produces: test accuracy/AUC/loss, a full classification report, a confusion matrix, training/validation accuracy-and-loss curves, and a one-vs-rest ROC curve — all logged to `experiment_results.csv` for cross-arm comparison.

Alternatively, open `plant_disease_detection.ipynb` to run (or review) all sixteen experiments in one place.

## Limitations

- Each backbone × arm combination was trained with a single random seed (no repeated-trial confidence intervals)
- All models are feature-extraction-only (frozen backbone); none were fine-tuned
- Augmentation hyperparameters used literature-standard defaults, not dataset-specific tuning
- Train/validation/test partitions were not audited for near-duplicate images

See Chapter 3 (Methodology) and Chapter 5 (Conclusion) for full discussion.

## License

*(Add your preferred license here — e.g. MIT, Apache 2.0.)*

## Acknowledgements

This project was completed as part of the M.Sc. in Data Science and Analytics program at Toronto Metropolitan University, under the supervision of Prof. Mucahit Cevik.
