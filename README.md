
# Breast Cancer Classification — Machine Learning Project  
A professionally structured end‑to‑end supervised learning workflow using multiple classification models to identify malignant and benign breast tumors.

---

## 1. Project Overview
This project develops, evaluates, and compares five widely used machine learning classification algorithms using the **Breast Cancer Wisconsin Diagnostic Dataset** from `scikit‑learn`.  
The primary objective is to determine which model provides the best balance between accuracy, stability, and generalization performance.

The project includes:
- Exploratory understanding of dataset structure  
- Clean preprocessing pipeline  
- Proper feature scaling with no data leakage  
- Model training and evaluation  
- Comparative performance analysis  
- Final model selection with statistical justification  

---

## 2. Dataset Description
The dataset (`load_breast_cancer`) contains **569 samples**, each with **30 continuous numerical features** extracted from digitized images of breast cell nuclei.

### Key characteristics:
- **Features:** Measurements of cell radius, texture, perimeter, symmetry, smoothness, concavity, fractal dimension, etc.  
- **Target variable:**  
  - `0` → Malignant  
  - `1` → Benign  
- **No missing values**  
- **All numerical features** → No label encoding required  

The dataset is slightly imbalanced (Benign 357 / Malignant 212), but still suitable for standard modeling without resampling.

---

## 3. Preprocessing Pipeline

### 3.1 Data Integrity Checks  
- Verified absence of missing values  
- Confirmed feature data types  
- Examined class distribution  

### 3.2 Train–Test Split  
A standard 80/20 split ensures objective evaluation:
```
train_test_split(X, y, test_size=0.2, random_state=42)
```

### 3.3 Feature Scaling  
Because the dataset contains measurements with large differences in magnitude, scaling is essential.

**StandardScaler** was applied **only after splitting** to avoid data leakage:
```
scaler.fit(X_train)
scaler.transform(X_test)
```

This improves performance for distance‑based models (SVM, KNN) and stabilizes optimization for Logistic Regression.

---

## 4. Models Implemented
The following models were trained with consistent cross‑comparable settings:

### Logistic Regression  
Uses a linear decision boundary with a sigmoid function to estimate class probabilities.

### Decision Tree Classifier  
Creates hierarchical decision rules; highly interpretable but prone to overfitting.

### Random Forest Classifier  
An ensemble of decision trees using **bagging** and **bootstrapping**, reducing variance but sometimes overfitting on clean datasets.

### Support Vector Classifier (SVC)  
Finds the optimal margin between classes; performs strongly with scaled features.

### K‑Nearest Neighbors (KNN)  
Classifies based on the majority vote of nearest neighbors; sensitive to scaling.

---

## 5. Model Performance Comparison (Updated)

Using correctly scaled data, the following performance was obtained:

| Model | Train Accuracy | Test Accuracy | Observations |
|-------|----------------|---------------|--------------|
| **Logistic Regression** | **0.9868** | **0.9737** | Strong generalization; stable and consistent |
| Decision Tree | 1.0000 | 0.9386 | Clear overfitting; training accuracy too high |
| Random Forest | 1.0000 | 0.9561 | High performance but still overfits; less stable |
| **Support Vector Machine (SVC)** | **0.9890** | **0.9825** | Best performance; excellent generalization on scaled features |
| KNN | 0.9802 | 0.9473 | Good performance but sensitive to noise; less stable |

---

## 6. Final Model Selection – With Observations

###  **Best Overall Model: Support Vector Machine (SVC)**

####  Why SVC is the best:
- Highest **test accuracy: 0.9825**
- No signs of overfitting  
- Works exceptionally well with scaled numerical features  
- Strong boundary formation → robust generalization  
- Balanced precision, recall, and F1‑score  

####  Why Logistic Regression wasn't selected:
- Excellent model (0.9737 test accuracy)
- Slightly weaker generalization than SVC
- Still a strong, stable alternative

#### Why Decision Tree & Random Forest were rejected:
- Both show **train accuracy = 1.0**
- Clear overfitting and lower test accuracy than SVC

####  Why not KNN:
- Good performance but:
  - Lower test accuracy than SVC & Logistic Regression
  - Sensitive to distance scaling and noise  
  - Less stable across dataset variations
 

---

## 7. How to Run the Project

### Requirements
```
pip install numpy pandas scikit-learn matplotlib
```

### Run Notebook
```
jupyter notebook
```

Open:
```
Classification Algorithms Model Building.ipynb
```
and execute all cells.

---

## 8. Repository Contents
```
├── Classification Algorithms Model Building.ipynb
└── README.md
```



