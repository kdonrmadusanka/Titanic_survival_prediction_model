# 🚢 Titanic Survival Prediction – Machine Learning Project

## 📌 Project Overview

This project focuses on predicting passenger survival on the Titanic using **machine learning**. It demonstrates a **complete end‑to‑end ML workflow**, including data cleaning, feature engineering, preprocessing with pipelines, model building, and evaluation.

The project follows **industry best practices** using **pandas** and **scikit‑learn**, ensuring clean, reproducible, and leakage‑free model training.

---

## 📊 Dataset Description

The dataset contains passenger information with the following key columns:

* **Target Variable**:

  * `Survived` → 0 (Did not survive), 1 (Survived)

* **Numerical Features**:

  * `Age`, `Fare`, `SibSp`, `Parch`

* **Categorical Features**:

  * `Sex`, `Embarked`, `Cabin`

---

## 🧹 Data Cleaning & Preprocessing

### 1️⃣ Target Separation

To avoid data leakage, the target variable was separated before preprocessing:

* **X (features)** → All columns except `Survived`
* **y (target)** → `Survived`

---

### 2️⃣ Handling Missing Values

| Column   | Strategy          | Reason                         |
| -------- | ----------------- | ------------------------------ |
| Age      | Mean Imputation   | Continuous numeric feature     |
| Fare     | Median Imputation | Robust to outliers             |
| Embarked | Most Frequent     | Small categorical domain       |
| Cabin    | Dropped / Binary  | High cardinality, lost meaning |

---

### 3️⃣ Feature Engineering

#### 👨‍👩‍👧 Family Size

A new feature was created:

```
FamilySize = SibSp + Parch + 1
```

The original columns `SibSp` and `Parch` were dropped after feature creation.

---

### 4️⃣ Encoding Categorical Variables

* **Sex** → Binary encoding (`male = 0`, `female = 1`)
* **Embarked** → One‑Hot Encoding with `drop_first=True`

This avoids artificial ordering and multicollinearity.

---

### 5️⃣ Feature Scaling

| Feature | Transformation         |
| ------- | ---------------------- |
| Age     | StandardScaler         |
| Fare    | log1p → StandardScaler |

**Reasoning:**

* `Fare` is highly right‑skewed
* Log transformation makes it closer to a bell‑shaped distribution
* Scaling improves performance for linear models

---

### 6️⃣ Pipeline & ColumnTransformer

All preprocessing steps were implemented using **Pipeline** and **ColumnTransformer**, ensuring:

* Correct column‑wise transformations
* No data leakage
* Automatic reuse for validation and test sets

---

## 🤖 Model Building

### Algorithm Used

* **Logistic Regression**

**Why Logistic Regression?**

* Well‑suited for binary classification
* Interpretable
* Performs well with scaled features

---

### Model Pipeline

The full pipeline consists of:

1. Preprocessing (imputation, scaling, encoding)
2. Logistic Regression classifier

This ensures that raw data can be passed directly into the model.

---

## 🧪 Model Evaluation

### 1️⃣ Validation Accuracy

Model performance was evaluated using a validation set.

* **Accuracy Score**: Measures overall correctness of predictions

---

### 2️⃣ Confusion Matrix

The confusion matrix was used to analyze prediction errors:

|            | Predicted No    | Predicted Yes   |
| ---------- | --------------- | --------------- |
| Actual No  | True Negatives  | False Positives |
| Actual Yes | False Negatives | True Positives  |

This helps understand:

* Survival prediction errors
* Model bias toward any class

---

### 3️⃣ Classification Report

The following metrics were analyzed:

* **Precision** → Correct positive predictions
* **Recall** → Ability to detect survivors
* **F1‑Score** → Balance between precision and recall

---

## 📈 Results Summary

* The model successfully learned survival patterns from passenger data
* Feature engineering (FamilySize, Fare log transform) improved performance
* Pipeline‑based preprocessing ensured consistent results on unseen data

---

## 🧠 Key Learnings

* Proper data cleaning is crucial for model performance
* Log transformation helps handle skewed numerical features
* Pipelines prevent data leakage and simplify ML workflows
* Confusion matrices provide deeper insight than accuracy alone

---

## 🚀 Future Improvements

* Try advanced models (Random Forest, XGBoost)
* Hyperparameter tuning
* Cross‑validation
* Feature importance analysis

---

## 🛠 Technologies Used

* Python
* pandas
* NumPy
* scikit‑learn
* Matplotlib

---

## 📌 Conclusion

This project demonstrates a **complete and professional machine learning workflow**, from raw data cleaning to model evaluation. The use of pipelines and transformers ensures scalability, reproducibility, and correctness, making this project suitable for academic submission and real‑world ML applications.

---

⭐ *Feel free to fork this repository and experiment with different models!*
