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

## Model Performance

- ROC-AUC: 92.04%
- Accuracy: 87.98%
- Precision: 88.90%
- Recall: 94.51%
- F1 Score: 91.62%

The model was optimized to prioritize high recall for leukemia detection, which is important in medical screening applications.

---

## Unseen Test Performance

- Accuracy: 89.50%
- F1 Score: 89.95%

These results indicate strong generalization performance on previously unseen blood smear cell images.

---

## Confusion Matrix Summary

- Correct Healthy Predictions: 777
- Healthy predicted as Leukemia: 286
- Leukemia predicted as Healthy: 133
- Correct Leukemia Predictions: 2291


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
