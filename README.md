# 🩸 Leukemia Detection from Blood Smear Images using Deep Learning

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TIMM-1.0.26-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p align="center">
  <b>Automated detection of Acute Lymphoblastic Leukemia (ALL) from microscopic blood smear images using ConvNeXt Large and transfer learning.</b><br>
  Trained on C-NMC 2019 | 99% Recall on holdout | 96.13% ROC-AUC | Grad-CAM Explainability
</p>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Baseline Comparison](#baseline-comparison)
- [Grad-CAM Explainability](#grad-cam-explainability)
- [Limitations](#limitations)
- [Project Structure](#project-structure)
- [How to Reproduce](#how-to-reproduce)
- [Tech Stack](#tech-stack)
- [Disclaimer](#disclaimer)

---

## 🔬 Overview

Acute Lymphoblastic Leukemia (ALL) is the most common childhood cancer. Early and accurate detection from blood smear microscopy is critical — but manual diagnosis is time-consuming, expensive, and requires expert pathologists.

This project builds an **AI-assisted screening system** that classifies single-cell blood smear images into:

| Class | Description |
|-------|-------------|
| **ALL** | Leukemia cells (cancerous) |
| **HEM** | Healthy cells (normal) |

The system is designed to **maximize recall** — in cancer screening, missing a positive case is far more dangerous than a false alarm. The model achieves **99% recall on completely unseen holdout data**, missing only 1 leukemia case out of 100.

---

## 📊 Dataset

**C-NMC 2019 — Cancer Imaging Archive**

The dataset contains segmented single-cell microscopic blood smear images from the 2019 ISBI Challenge on ALL detection.

| Property | Value |
|----------|-------|
| Total Images | 10,661 |
| ALL (Leukemia) | 7,272 (68.2%) |
| HEM (Healthy) | 3,389 (31.8%) |
| Class Imbalance | 2.15× |
| Image Format | BMP |
| Pre-defined Folds | 3 (fold_0, fold_1, fold_2) |

**Data Split Strategy:**

| Split | Folds Used | Images | Purpose |
|-------|-----------|--------|---------|
| Train | fold_0 + fold_1 | 6,974 | Model learning |
| Validation | fold_2 | 3,487 | Hyperparameter decisions |
| Holdout Test | Stratified sample | 200 | Final evaluation only — never touched during training |

> The holdout test set was locked away before any training began and used **only once** at the very end. This ensures unbiased final performance reporting.

---

## ⚙️ Methodology

### 1. Class Imbalance Handling
The dataset has a 2.15× imbalance (ALL vs HEM). Two complementary strategies were applied:
- **WeightedRandomSampler** — oversamples HEM during training so the model sees balanced batches
- **Class-weighted loss** — applies higher penalty (1.5665×) for misclassifying HEM samples

### 2. Advanced Augmentation Pipeline
Standard augmentations (flip, rotate) are insufficient for medical imaging. We added:

| Augmentation | Why |
|-------------|-----|
| ElasticTransform | Simulates real cell membrane deformation |
| GridDistortion | Mimics microscope lens distortion |
| ColorJitter + HueSaturationValue | Handles staining variability across labs |
| GaussianBlur + GaussNoise | Simulates different scanner/microscope qualities |
| Affine (scale, shift, rotate) | Geometric invariance |

### 3. Gradual Unfreezing
Training in two phases prevents destroying pretrained ImageNet weights early:
- **Epochs 1–2:** Backbone frozen, only classification head trains (warm-up)
- **Epoch 3+:** Full model unfreezes with differential learning rates

### 4. Differential Learning Rates
- **Backbone:** 3×10⁻⁵ (fine-tune carefully — already learned features)
- **Head:** 3×10⁻⁴ (learn fast — new task)

### 5. Label Smoothing
Loss function uses label smoothing (ε=0.1) to prevent overconfident predictions and improve calibration.

### 6. Test Time Augmentation (TTA)
At inference, each image is processed through 5 different augmented versions. Probabilities are averaged for a more robust final prediction. TTA improves AUC by +0.21%.

### 7. Threshold Optimization
Instead of defaulting to 0.5, the optimal classification threshold was found using Youden's J statistic on the ROC curve, maximizing the balance between sensitivity and specificity.

---

## 🧠 Model Architecture

**ConvNeXt Large** — a modernized CNN architecture that matches Vision Transformer accuracy with CNN efficiency.

| Property | Value |
|----------|-------|
| Architecture | ConvNeXt Large |
| Pretrained On | ImageNet-22k → ImageNet-1k |
| Total Parameters | 196.23M |
| Input Resolution | 384 × 384 |
| Drop Path Rate | 0.2 (stochastic depth regularization) |
| Output Classes | 2 (ALL / HEM) |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingWarmRestarts (T₀=10) |
| Loss Function | CrossEntropyLoss + Label Smoothing (ε=0.1) + Class Weights |
| Mixed Precision | ✅ (torch.amp) |
| Gradient Clipping | max_norm=1.0 |
| Early Stopping | patience=7 (monitored on Val F1) |
| GPU | NVIDIA RTX PRO 6000 Blackwell (102GB VRAM) |
| Training Time | ~17 minutes |

---

## 📈 Results

### Validation Performance (3,487 images — fold_2)

| Metric | Score |
|--------|-------|
| F1 Score (with TTA) | **90.98%** |
| ROC-AUC | 90.17% |
| Recall | 96.99% |
| Precision | 85.68% |
| Accuracy | 86.23% |

### Holdout Test Performance (200 completely unseen images)

> These numbers are reported from a **locked holdout set** — never seen during training or validation. This is the honest generalization performance.

| Metric | Score |
|--------|-------|
| **F1 Score** | **90.00%** |
| **ROC-AUC** | **96.13%** |
| **Recall** | **99.00%** |
| Precision | 82.50% |
| Accuracy | 89.00% |

### Confusion Matrix (Holdout Test Set — with TTA)

| | Predicted HEM | Predicted ALL |
|--|--------------|--------------|
| **Actual HEM** | 79 ✅ | 21 ❌ |
| **Actual ALL** | **1** ❌ | 99 ✅ |

> Only **1 leukemia case missed** out of 100. The model is optimized for screening — high recall is the clinical priority.

### Training Curves

![Training Curves](results/training_curves.png)

### Evaluation Results

![Evaluation](results/evaluation_results.png)

### Holdout Evaluation

![Holdout](results/holdout_evaluation.png)

---

## 🏆 Baseline Comparison

All models trained on the same data with the same loss function and augmentation pipeline. Only the architecture differs.

| Model | Holdout F1 | Holdout AUC | Recall | Params | Epochs |
|-------|-----------|-------------|--------|--------|--------|
| EfficientNet-B0 | 74.07% | 86.61% | 74.07% | ~5M | 10 |
| ResNet50 | 86.00% | 90.97% | 86.00% | ~25M | 10 |
| **ConvNeXt Large (Ours)** | **90.00%** | **96.13%** | **99.00%** | ~196M | 25 |

![Baseline Comparison](results/baseline_comparison.png)

> **Note:** Baselines were trained for 10 epochs for computational efficiency. A fair full-epoch comparison would likely narrow the F1 gap, but the AUC and Recall advantages of ConvNeXt Large are consistent with its architectural superiority for fine-grained medical image classification.

---

## 🔍 Grad-CAM Explainability

Gradient-weighted Class Activation Mapping (Grad-CAM) visualizes which regions of the cell the model focuses on when making predictions. This is critical for medical AI — the model must attend to biologically meaningful features, not image artifacts.

**Target Layer:** `stages[-1].blocks[-1]` — final convolutional block capturing highest-level semantic features.

### Correct Predictions

![Grad-CAM Correct](results/gradcam_correct_fixed.png)

**Observations:**
- **ALL (Leukemia):** Model attends to **irregular nuclear chromatin patterns** and **cell membrane protrusions** — exactly the morphological features hematologists use for diagnosis
- **HEM (Healthy):** Model attends to **cell boundary regularity** — healthy cells are round and smooth, which the model correctly identifies as the key discriminating feature

### Error Analysis

![Grad-CAM Errors](results/gradcam_wrong_fixed.png)

**Understanding model failures:**
- **Missed cancer (P(ALL)=0.231):** The leukemia cell had unusually round morphology resembling a healthy cell. Model attention scattered to background — a genuinely ambiguous case
- **False alarms:** Healthy cells with irregular boundaries or double-lobed shapes were flagged as leukemic — biologically understandable misclassifications that would challenge human annotators too

---

## ⚠️ Limitations

This project is honest about its limitations:

1. **HEM false positive rate is high** — 21 out of 100 healthy cells were flagged as leukemia. This is driven by the 2.15× class imbalance and the model's bias toward predicting ALL. Future work should address this with stronger oversampling or focal loss.

2. **Validation loss gap** — A persistent train/val loss gap (~0.18) indicates mild overfitting. More aggressive dropout or larger training data would help.

3. **Single dataset** — The model was trained and tested only on C-NMC 2019. Performance on images from different hospitals, staining protocols, or microscope types is unknown. Stain normalization (Macenko method) was considered but not implemented in the final pipeline.

4. **Baseline training epochs** — Baselines were trained for 10 epochs vs 25 for ConvNeXt Large. A fully fair comparison would train all models to convergence.

5. **Not a clinical tool** — This model is for research and educational purposes only and has not been validated for clinical use.

---

## 📁 Project Structure
Leukemia-detection-from-blood-smear-images-using-deep-learning/
│
├── Leukemia_Detection_Using_DL.ipynb    # Complete training notebook
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
├── LICENSE                               # MIT License
├── .gitignore
│
└── results/                             # All outputs (generated during training)
├── training_curves.png
├── evaluation_results.png
├── tta_results.png
├── holdout_evaluation.png
├── baseline_comparison.png
├── gradcam_correct_fixed.png
├── gradcam_wrong_fixed.png
├── classification_report.txt
├── baseline_comparison.csv
└── final_results.csv

> Model weights (~800MB) are stored on Google Drive due to GitHub file size limits.

---

## 🚀 How to Reproduce

### Requirements
```bash
pip install torch torchvision timm albumentations torchmetrics
```

### Step 1 — Get the Dataset
Download **C-NMC 2019** from the [Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758223) and place `PKG - C-NMC 2019.zip` in your Google Drive root.

### Step 2 — Get the Notebook
Open `Leukemia_Detection_Using_DL.ipynb` in Google Colab.

### Step 3 — Run Phase by Phase
The notebook is structured in clearly labeled phases. Run each phase sequentially:

| Phase | Description |
|-------|-------------|
| Phase 01 | Environment setup & GPU verification |
| Phase 02 | Dataset loading & EDA |
| Phase 03 | Holdout test set creation & train/val split |
| Phase 04 | Advanced augmentation pipeline |
| Phase 05 | ConvNeXt Large model architecture |
| Phase 06 | Training loop with gradual unfreezing |
| Phase 07 | Evaluation & threshold optimization |
| Phase 08 | Test Time Augmentation (TTA) |
| Phase 09 | Holdout test set evaluation |
| Phase 10 | Grad-CAM explainability |
| Phase 11 | Baseline comparison |

### Step 4 — Download Pretrained Model
Pretrained model weights (`best_convnext_large_v2.pth`) available on request via Google Drive.

### Hardware Used
- GPU: NVIDIA RTX PRO 6000 Blackwell (102GB VRAM)
- Training time: ~17 minutes
- The notebook also runs on A100/V100 (adjust BATCH_SIZE accordingly)

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.12 | Core language |
| PyTorch | 2.10 | Deep learning framework |
| TIMM | 1.0.26 | ConvNeXt Large pretrained model |
| Albumentations | latest | Advanced image augmentation |
| scikit-learn | latest | Metrics, class weights |
| Matplotlib / Seaborn | latest | Visualization |
| Google Colab | — | Training environment |
| Google Drive | — | Dataset and model storage |
| NumPy / Pandas | latest | Data processing |

---

## 📜 Disclaimer

This project is for **educational and research purposes only**.

It is **not** a validated clinical diagnostic system and must **not** be used for medical decision-making. Always consult qualified medical professionals for diagnosis and treatment decisions.

The C-NMC 2019 dataset is publicly available through the Cancer Imaging Archive for research use.

---

## 👤 Author

**Niteesh014**

[![GitHub](https://img.shields.io/badge/GitHub-Niteesh014-181717?style=for-the-badge&logo=github)](https://github.com/Niteesh014)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

