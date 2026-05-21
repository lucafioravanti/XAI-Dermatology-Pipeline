# XAI Dermatology Pipeline: Skin Lesion Classification

## Executive Summary
This repository contains a deep learning pipeline for the classification of skin lesions, aimed at supporting Explainable AI (XAI) in Healthcare. The project utilizes the **HAM10000 dataset**, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.

The original goal of this project was a comparative analysis across several architectures (ViT, ResNet, SwinT, and AutoKeras) as part of a Master's program in **Explainable AI in Healthcare Management**. This repository has since been refactored into a professional, modular pipeline to demonstrate the end-to-end management of clinical data, from exploratory analysis and preprocessing to model engineering and training.

## Dataset: HAM10000
The HAM10000 ("Human Against Machine with 10000 training images") dataset consists of 10,015 dermatoscopic images. The task is to classify images into one of 7 diagnostic categories:
- Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec)
- Basal cell carcinoma (bcc)
- Benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl)
- Dermatofibroma (df)
- Melanoma (mel)
- Melanocytic nevi (nv)
- Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc)

A significant challenge with clinical data such as HAM10000 is the severe class imbalance (e.g., Melanocytic nevi represent over 60% of the dataset).

## Comparative Results
As part of the academic project, we benchmarked multiple models on this classification task:
- **ResNet-152**: Achieved the highest validation accuracy at **79.08%** (ResNet-50 at 78.86%, ResNet-18 at 76.54%).
- **AutoKeras**: Reached an average weighted accuracy of **80%** (using 2 hidden layers, ReLU, dropout 0.5), though with some fluctuations in the loss curve.
- **Swin Transformer (SwinT)**: Reached a test accuracy of **67.75%**.
- **Vision Transformer (ViT)**: Showed a training accuracy of 75% and a validation accuracy between **60-65%**. The fluctuations indicate early signs of overfitting, highlighting the need for more aggressive regularization and data augmentation in future iterations.

## Personal Contribution
This repository highlights my specific contributions to the group project, focusing on data engineering and the implementation of the Vision Transformer. The core of my work includes:

1. **Exploratory Data Analysis (EDA) & Preprocessing**: Analyzing the clinical metadata and implementing strategies to handle missing values and data formatting (`notebooks/eda/`).
2. **Class Imbalance Management (SMOTE)**: Engineering a pipeline module to dynamically apply Synthetic Minority Over-sampling Technique (SMOTE) to balance the diagnostic classes prior to model training.
3. **ViT Engineering & Fine-Tuning**: Implementing, fine-tuning, and evaluating the Vision Transformer (ViT) architecture using `vit-keras` and TensorFlow, complete with learning rate scheduling and early stopping.
4. **Pipeline Refactoring**: Transforming a monolithic academic script into a modular, production-ready ETL and Training pipeline (`src/`).

*(Note: The exploratory models and scripts developed by my peers are preserved in `notebooks/group_work/` for reference).*

## Next Steps
To improve the robustness and generalizability of the Vision Transformer model, future steps include:
- **Enhanced Data Augmentation**: Applying heavier geometric and color transformations to prevent overfitting.
- **K-Fold Cross-Validation**: Implementing robust validation strategies to better evaluate model performance on the imbalanced clinical data.
- **Advanced Feature SMOTE**: Refining the oversampling strategy on latent feature spaces rather than raw pixel/tabular combinations.

## How to Run the Pipeline

### 1. Setup Environment
Ensure you have Python installed. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Download Data
Download the HAM10000 dataset (e.g., from Kaggle or Harvard Dataverse) and place it in the `data/` directory. The structure should look like this:
```text
data/
├── HAM10000_metadata.csv
└── HAM10000_all_images/
    ├── ISIC_0024306.jpg
    ├── ISIC_0024307.jpg
    └── ...
```

### 3. Run the Training Pipeline
You can execute the entire pipeline (data loading, preprocessing, model building, training, and evaluation) by running:
```bash
python main.py
```

The model architecture and hyperparameters can be configured in `src/config.py`.
