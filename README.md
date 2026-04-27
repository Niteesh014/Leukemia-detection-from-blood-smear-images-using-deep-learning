# Leukemia Detection from Blood Smear Images using Deep Learning

## Overview

This project focuses on automated detection of **Acute Lymphoblastic Leukemia (ALL)** from microscopic blood smear images using deep learning.

A transfer learning approach was implemented using **ConvNeXt Large** to classify blood cell images into:

- **ALL** — Leukemia Cells  
- **HEM** — Healthy Cells

The objective of this project is to support intelligent medical image analysis through accurate binary classification.

---

## Dataset

**C-NMC 2019 Leukemia Dataset**

The dataset contains segmented single-cell microscopic blood smear images belonging to two classes:

- ALL (Leukemia)
- HEM (Healthy)

---

## Model Used

- ConvNeXt Large
- Transfer Learning
- Input Resolution: 384 × 384
- Mixed Precision Training
- AdamW Optimizer
- Cosine Learning Rate Scheduler

---

## Workflow

- Data preprocessing and image normalization  
- Holdout testing set creation  
- Model training on GPU  
- Validation and performance evaluation  
- Real image upload prediction system

---

## Performance

The trained model achieved strong classification performance on unseen validation and holdout samples with high recall and strong overall discrimination ability.

---

## Features

- Blood smear image classification  
- Leukemia probability prediction  
- Custom image upload testing  
- Saved trained model  
- ROC Curve and Confusion Matrix evaluation

---

## Tech Stack

- Python  
- PyTorch  
- timm  
- torchvision  
- scikit-learn  
- Google Colab  
- Matplotlib  
- Seaborn

---

## Future Improvements

- Streamlit Web Application  
- Grad-CAM Explainability  
- External Dataset Validation  
- Threshold Optimization

---

## Disclaimer

This project is for educational and research purposes only. It is not a clinical diagnostic system.

---

## Author

**Niteesh014**
