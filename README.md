# AI Screenshot & Visual Organizer - ML Pipeline (v1 Baseline)

This repository contains the machine learning and deep learning pipeline for the **AI Screenshot & Visual Organizer** project. It focuses on image preprocessing, exploratory data analysis (EDA), transfer learning via Convolutional Neural Networks (CNN), evaluation metrics, and a business logic mapping layer.

---

## Project Overview & Problem Definition
Categorizing images and screenshots manually in digital archives is a tedious task. This `v1` baseline model aims to automatically classify visual inputs into core e-commerce categories using deep learning, followed by a **Taxonomy Mapping** layer that converts granular model outputs into user-friendly high-level folders.

---

## Dataset
* **Source:** [Kaggle - Ecommerce Product Images for Product Categorization](https://www.kaggle.com/datasets/fatihkgg/ecommerce-product-images-18k)
* **Subset Optimization:** To ensure robust training while managing computational constraints (Colab T4 GPU), a balanced subset of **500 samples per class** (Total: 4,500 images across 9 classes) was extracted and resized to `224x224` pixels.
* **Classes:** 
  `BABY_PRODUCTS`, `BEAUTY_HEALTH`, `CLOTHING_ACCESSORIES_JEWELLERY`, `ELECTRONICS`, `GROCERY`, `HOBBY_ARTS_STATIONERY`, `HOME_KITCHEN_TOOLS`, `PET_SUPPLIES`, `SPORTS_OUTDOOR`.

---

## Model Architecture & Training
* **Backbone:** **MobileNetV2** (Pretrained on ImageNet) using a *Feature Extraction* approach (frozen backbone, trainable classifier head).
* **Optimization:** `Adam` optimizer (lr=0.001), `CrossEntropyLoss`.
* **Data Splitting:** 80% Training, 20% Validation with Data Augmentation (RandomHorizontalFlip, Rotation, ColorJitter).
* **Validation Accuracy:** ~57% (Baseline performance, boosting to ~66% post-mapping).

---

## Business Logic: Taxonomy Mapping
Since raw e-commerce classes are too granular for personal gallery organization, an inference mapping layer (`TAXONOMY_MAP`) was implemented to bridge raw predictions to high-level system folders:
* *Example:* `BABY_PRODUCTS` -> `Retail & Personal Care`
* *Example:* `ELECTRONICS` -> `Consumer Tech`

---

## 📂 Repository Structure
```text
├── v1_ecommerce_baseline.ipynb   # Complete ML pipeline (EDA -> Training -> Evaluation -> Inference)
├── saved_model.pth               # Serialized PyTorch model weights & class labels
└── README.md                     # Project documentation
```
---

## Roadmap (Future Versions)
* **v2**: Integrate OCR (Optical Character Recognition) and Keyword Fallback routing to handle noisy, real-world phone screenshots (OOD data).
* **v3**: Implement Visual-Language Models (VLM) for zero-shot semantic understanding and automated information extraction.
